# -*- coding: utf-8 -*-
# ============================================================
# APDD-DNV - YOLOv8n / DyHead(alpha=.50) / MEND-Net(alpha=.75)
# Original Trainv11-UP hyperparameters; each model trains 300 epochs
# from the SAME original YOLOv8n.pt (NOT from a trained DyHead).
#
# ONLY EDIT THE FIVE CONFIGURATION ENTRIES BELOW for each run.
# The architecture and fixed alpha are read from MODEL_YAML.
# ============================================================

# 1. Environment variables (before torch/ultralytics imports)
import os

SEED = 42  # Exact seed in the user-supplied Trainv11-UP (20261103)
# The original script selected physical GPU 1. Override in PowerShell with
# $env:CUDA_VISIBLE_DEVICES="0" if your RTX 4090 is physical GPU 0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["MENDNET_E11_LEARNABLE"] = "0"
os.environ["MENDNET_E12_ASYM"] = "0"
os.environ.pop("MENDNET_E12_ALPHA_FINE", None)
os.environ.pop("MENDNET_E12_ALPHA_COARSE", None)

# 2. Imports
import argparse
import hashlib
import json
import platform
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO

# ============================================================
# 3. Editable configuration -- EXACTLY the 5 entries requested
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# MODEL_YAML = BASE_DIR / "yaml-series" / "APDD_MENDNet_alpha075.yaml"
# PRETRAINED_WEIGHTS = BASE_DIR / "Model" / "yolov8n.pt"
# DATA_YAML = BASE_DIR / "APDD-DNV_night.yaml"
# PROJECT_DIR = BASE_DIR / "runs" / "APDD_DNV_RenewMulti"
# EXPERIMENT_NAME = "MENDNet_MLLA_LSKA_DyHead-a075_Night_seed42_300e"


MODEL_YAML = BASE_DIR / "yaml-series" / "APDD_MENDNet_alpha075.yaml"
PRETRAINED_WEIGHTS = BASE_DIR / "Model" / "yolov8n.pt"
DATA_YAML = BASE_DIR / "APDD-DNV_mix.yaml"
PROJECT_DIR = BASE_DIR / "runs" / "APDD_DNV_MultiIllum_V2"
EXPERIMENT_NAME = "MENDNet_a075_Mix_seed42_300e"


# ============================================================
# 4. Original Trainv11-UP training hyperparameters -- UNCHANGED
# ============================================================
EPOCHS = 300
BATCH = 16
IMGSZ = 896
LR0 = 0.003
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
COS_LR = False
MOSAIC = 0.5
CLOSE_MOSAIC = 50
PATIENCE = 0
WORKERS = 4
AMP = True

# Independent validation protocol used in the prior APDD-Night experiments.
# This does NOT change the 300-epoch training recipe.
EVAL_IMGSZ = 896
EVAL_BATCH = 32
EVAL_RECT = True
EVAL_HALF = True
EVAL_CONF = 0.001
EVAL_IOU = 0.7
EVAL_MAX_DET = 300

CLASS_NAMES = [
    "Joint_deterioration", "Corner_break", "Crack",
    "Grouting_patching", "Corner_pothole", "Localized_patching",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def set_random_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[INFO] Random seed: {seed}", flush=True)
    print("[INFO] Deterministic mode enabled", flush=True)


def read_yaml(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    return value


def config_architecture():
    cfg = read_yaml(MODEL_YAML)
    nodes = cfg.get("backbone", []) + cfg.get("head", [])
    if cfg.get("nc") != 6 or cfg.get("scale") != "n":
        raise RuntimeError(f"MODEL_YAML requires nc: 6 and scale: n: {MODEL_YAML}")
    if len(nodes) == 23 and nodes[-1][2] == "Detect":
        if nodes[-1][0] != [15, 18, 21]:
            raise RuntimeError("Incorrect native YOLOv8n Detect inputs")
        if "dyhead_alpha" in cfg:
            raise RuntimeError("YOLOv8n baseline must not set dyhead_alpha")
        return "YOLOv8n", None, 23
    if len(nodes) == 27 and nodes[8][2] == "C2f" and nodes[26][2] == "DynamicHead":
        if nodes[26][0] != [19, 22, 25]:
            raise RuntimeError("DyHead-only must use P3/P4/P5 [19,22,25]")
        kind = "DyHead-only"
    elif (len(nodes) == 28 and nodes[8][2] == "C2fMLLAHybrid"
          and nodes[26][2] == "LSKA" and nodes[27][2] == "DynamicHead"):
        if nodes[27][0] != [19, 22, 26] or nodes[26][3] != [11, True, 0.05]:
            raise RuntimeError("MEND-Net must be P5 MLLA + residual LSKA k11 + DyHead")
        if nodes[8][3] != [1024, True, 1, 0.5, 0.05]:
            raise RuntimeError("MEND-Net C2fMLLAHybrid arguments do not match E10")
        kind = "MEND-Net"
    else:
        raise RuntimeError(f"Unknown or incompatible 300-epoch architecture: {MODEL_YAML}")
    if "dyhead_alpha" not in cfg:
        raise RuntimeError("Custom YAML must contain dyhead_alpha: 0.5 or 0.75")
    alpha = float(cfg["dyhead_alpha"])
    expected = 0.50 if kind == "DyHead-only" else 0.75
    if abs(alpha - expected) > 1e-8:
        raise RuntimeError(f"{kind}: required alpha={expected}, YAML specifies {alpha}")
    return kind, alpha, len(nodes)


def check_data_yaml():
    cfg = read_yaml(DATA_YAML)
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(6)]
    if names != CLASS_NAMES:
        raise RuntimeError(f"Dataset names/order incorrect: {names}")
    if cfg.get("nc", 6) != 6:
        raise RuntimeError("Dataset nc is not 6")
    for name in ("train", "val", "test"):
        if name not in cfg:
            raise RuntimeError(f"Dataset YAML missing {name} split")


def get_train_model(trainer):
    m = trainer.model
    return m.module if hasattr(m, "module") else m


def check_graph_and_alpha(nn_model, kind, target_alpha, where):
    nn_model = nn_model.module if hasattr(nn_model, "module") else nn_model
    if nn_model is None:
        return None
    layers = list(nn_model.model)
    layer_count = {"YOLOv8n":23, "DyHead-only":27, "MEND-Net":28}[kind]
    if len(layers) != layer_count:
        raise RuntimeError(f"{where}: expected {layer_count} model layers; got {len(layers)}")
    if kind == "YOLOv8n":
        if type(layers[-1]).__name__ != "Detect":
            raise RuntimeError(f"{where}: expected original Detect head")
        return None
    if type(layers[-1]).__name__ != "DynamicHead":
        raise RuntimeError(f"{where}: missing DynamicHead")
    expected_inputs = [19,22,25] if kind == "DyHead-only" else [19,22,26]
    if list(layers[-1].f) != expected_inputs:
        raise RuntimeError(f"{where}: wrong DyHead inputs: {layers[-1].f}")
    tower = list(layers[-1].dyhead_tower)
    if len(tower) != 3:
        raise RuntimeError(f"{where}: expected 3 DyConv blocks, got {len(tower)}")
    alphas = []
    for b in tower:
        if any(hasattr(b, attr) for attr in (
                "cross_scale_alpha_logit", "cross_scale_alpha_fine", "cross_scale_alpha_coarse")):
            raise RuntimeError(f"{where}: E11/E12 or asymmetric DyHead is not allowed")
        a = float(getattr(b, "cross_scale_alpha", -999))
        if abs(a-target_alpha) > 1e-8:
            raise RuntimeError(f"{where}: DyHead alpha {a} != required {target_alpha}")
        alphas.append(a)
    if kind == "MEND-Net":
        if type(layers[8]).__name__ != "C2fMLLAHybrid":
            raise RuntimeError(f"{where}: MiGLA-CPE missing at layer 8")
        if type(layers[26]).__name__ != "LSKA":
            raise RuntimeError(f"{where}: LSKA missing at layer 26")
        if getattr(layers[26], "residual", False) is not True:
            raise RuntimeError(f"{where}: LSKA must be residual")
        for idx in (8, 26):
            if getattr(layers[idx], "gamma", None) is None:
                raise RuntimeError(f"{where}: missing module gate at layer {idx}")
    return alphas


def source_integrity_check(kind):
    root = Path(ultralytics.__file__).resolve().parent.parent
    if root != BASE_DIR.resolve():
        raise RuntimeError(f"Wrong installed ultralytics: {root}; expected: {BASE_DIR}. "
                           "Run from isolated repo root with PYTHONPATH set to it.")
    if ultralytics.__version__ != "8.4.3":
        raise RuntimeError(f"Expected custom Ultralytics 8.4.3, got {ultralytics.__version__}")
    if kind != "YOLOv8n":
        dh = BASE_DIR/"ultralytics"/"nn"/"Addmodules"/"DynamicHead.py"
        if not dh.is_file() or "MEND_E10_SCALE_ALPHA_V1" not in dh.read_text(encoding="utf-8-sig"):
            raise RuntimeError("E10-only DynamicHead.py missing; use the existing E10 isolated project")
        if kind == "MEND-Net":
            from ultralytics.nn import tasks
            for name in ("C2fMLLAHybrid", "LSKA", "DynamicHead"):
                if not hasattr(tasks, name):
                    raise RuntimeError(f"tasks.py missing: {name}")
    print("[FORK_PASS]", root, ultralytics.__version__, flush=True)


def check_environment():
    print("\n========== Environment ==========")
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. If GPU is physical GPU 0, run PowerShell: "
                           "$env:CUDA_VISIBLE_DEVICES='0' before starting Python.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Seed:", SEED)
    for f in (MODEL_YAML, PRETRAINED_WEIGHTS, DATA_YAML):
        if not f.is_file():
            raise FileNotFoundError(f)
        print("[OK]", f)
    kind, alpha, count = config_architecture()
    check_data_yaml()
    source_integrity_check(kind)
    if alpha is not None:
        os.environ["MENDNET_DYHEAD_ALPHA"] = str(alpha)
    print("[ARCHITECTURE_PASS]", kind, "layers=", count, "alpha=", alpha, flush=True)
    print("[MODEL_YAML_SHA256]", sha256(MODEL_YAML), flush=True)
    print("[PRETRAIN_SHA256]", sha256(PRETRAINED_WEIGHTS), flush=True)
    print("[DATA_YAML_SHA256]", sha256(DATA_YAML), flush=True)
    return kind, alpha


def confirm_pretrain(model):
    # Early backbone layers are structurally identical in all three YAMLs.
    source = YOLO(str(PRETRAINED_WEIGHTS))
    src = source.model.state_dict()
    dst = model.model.state_dict()
    checked = 0
    for key, val in src.items():
        if not key.startswith(tuple(f"model.{i}." for i in range(8))):
            continue
        if key not in dst or tuple(val.shape) != tuple(dst[key].shape):
            raise RuntimeError(f"Original YOLOv8n backbone did not transfer: {key}")
        if not torch.allclose(val.float().cpu(), dst[key].float().cpu(), atol=1e-7, rtol=1e-6):
            raise RuntimeError(f"Pretrained weight mismatch: {key}")
        checked += 1
    del source
    if checked < 30:
        raise RuntimeError("Unexpectedly few matching backbone tensors")
    print("[YOLOV8N_PRETRAIN_PASS]", checked, "early-backbone tensors", flush=True)


def next_run_name():
    base = EXPERIMENT_NAME
    if not base or any(x in base for x in ("/", "\\", "..")):
        raise ValueError(f"Invalid EXPERIMENT_NAME: {base}")
    candidate = base
    n = 2
    while (PROJECT_DIR / candidate).exists():
        candidate = f"{base}_run{n:02d}"
        n += 1
    print("[RUN_NAME]", candidate, flush=True)
    return candidate


def save_experiment_info(save_dir, run_name, kind, alpha):
    save_dir = Path(save_dir)
    out = save_dir / "experiment_configs"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_YAML, out / MODEL_YAML.name)
    shutil.copy2(DATA_YAML, out / DATA_YAML.name)
    script_path = Path(__file__).resolve()
    if script_path.is_file():
        shutil.copy2(script_path, out / script_path.name)
    source_files = [
        BASE_DIR / "ultralytics" / "nn" / "tasks.py",
        BASE_DIR / "ultralytics" / "nn" / "Addmodules" / "DynamicHead.py",
        BASE_DIR / "ultralytics" / "nn" / "Addmodules" / "LSKA.py",
        BASE_DIR / "ultralytics" / "nn" / "Addmodules" / "MLLABlock.py",
    ]
    info = {
        "experiment_name": run_name, "date": datetime.now().isoformat(),
        "architecture": kind, "dyhead_alpha": alpha,
        "dataset": str(DATA_YAML), "model_yaml": str(MODEL_YAML),
        "pretrained_weights": str(PRETRAINED_WEIGHTS),
        "model_yaml_sha256": sha256(MODEL_YAML),
        "data_yaml_sha256": sha256(DATA_YAML),
        "pretrained_weights_sha256": sha256(PRETRAINED_WEIGHTS),
        "source_code_sha256": {str(f.relative_to(BASE_DIR)): sha256(f) for f in source_files if f.is_file()},
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ,
        "optimizer": "SGD", "lr0": LR0, "lrf": LRF,
        "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "cos_lr": COS_LR, "mosaic": MOSAIC, "close_mosaic": CLOSE_MOSAIC,
        "amp": AMP, "patience": PATIENCE, "workers": WORKERS,
        "deterministic": True, "python": platform.python_version(),
        "pytorch": torch.__version__, "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_source": "YOLO(str(MODEL_YAML)); model.load(str(PRETRAINED_WEIGHTS))",
        "not_dyhead_best_warmstart": True,
        "evaluation_selection_note": "train best.pt chosen by Val fitness; report independent Test separately",
    }
    (out / "experiment_info.json").write_text(
        json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
    print("[EXPERIMENT_INFO_SAVED]", out / "experiment_info.json", flush=True)


def make_alpha_guard(kind, alpha, run_name):
    saved = {"done": False}
    def guard(trainer):
        train = check_graph_and_alpha(get_train_model(trainer), kind, alpha, "train")
        ema_obj = getattr(getattr(trainer, "ema", None), "ema", None)
        if kind != "YOLOv8n" and ema_obj is not None:
            ema = check_graph_and_alpha(ema_obj, kind, alpha, "ema")
        else:
            ema = None
        if not saved["done"]:
            save_experiment_info(trainer.save_dir, run_name, kind, alpha)
            print("[TRAIN_GRAPH_ALPHA_PASS]", train, "ema=", ema, flush=True)
            saved["done"] = True
    return guard


def inspect_checkpoint_alpha(path, kind, alpha):
    path = Path(path)
    if not path.is_file():
        return None
    # This only unpickles the checkpoint. It does NOT execute a model forward on CPU.
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        checkpoint = torch.load(str(path), map_location="cpu")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise RuntimeError("Missing model/ema in checkpoint " + str(path))
    values = check_graph_and_alpha(model, kind, alpha, f"checkpoint:{path.name}")
    print("[CHECKPOINT_ALPHA_PASS]", path.name, values, "sha256=", sha256(path), flush=True)
    return values


def train():
    set_random_seed(SEED)  # CRITICAL: seed before YOLO(MODEL_YAML) initializes random new modules.
    kind, alpha = check_environment()
    run_name = next_run_name()
    print("\n========== Loading model ==========")
    # Deliberately EXACTLY the supplied Trainv11-UP source-init sequence.
    model = YOLO(str(MODEL_YAML))
    model.load(str(PRETRAINED_WEIGHTS))
    check_graph_and_alpha(model.model, kind, alpha, "initial")
    confirm_pretrain(model)
    print("Custom architecture:", MODEL_YAML.name)
    print("Pretrained weights:", PRETRAINED_WEIGHTS.name)
    print("Alpha:", alpha)
    print("\n========== Training started ==========")
    def on_epoch_end(trainer):
        check_graph_and_alpha(get_train_model(trainer), kind, alpha, "epoch-end")
        if kind != "YOLOv8n":
            ema_nn = getattr(getattr(trainer, "ema", None), "ema", None)
            if ema_nn is None:
                raise RuntimeError("Missing EMA before validation; alpha cannot be guaranteed")
            values = check_graph_and_alpha(ema_nn, kind, alpha, "ema-before-val")
            if trainer.epoch in (0, 4, 49, 149, 249, 299):
                print("[TRAIN_EMA_ALPHA_PASS]", "epoch=", trainer.epoch + 1,
                      "alpha=", values, flush=True)
    model.add_callback("on_pretrain_routine_end", make_alpha_guard(kind, alpha, run_name))
    model.add_callback("on_train_epoch_start", lambda tr: check_graph_and_alpha(get_train_model(tr),kind,alpha,"epoch-start"))
    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=0,
        workers=WORKERS,
        optimizer="SGD",
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        cos_lr=COS_LR,
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,
        amp=AMP,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        val=True,
        save=True,
        save_period=-1,
        plots=True,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=False,
        verbose=True,
    )
    save_dir = Path(model.trainer.save_dir)
    save_experiment_info(save_dir, run_name, kind, alpha)
    for fname in ("best.pt", "last.pt"):
        inspect_checkpoint_alpha(save_dir / "weights" / fname, kind, alpha)
    print("\n========== Training completed ==========")
    print("Experiment directory:", save_dir)
    print("Best weights:", save_dir / "weights" / "best.pt")
    print("Last weights:", save_dir / "weights" / "last.pt")
    print("Results CSV:", save_dir / "results.csv")
    print("Results figure:", save_dir / "results.png")
    print("Experiment information:", save_dir / "experiment_configs" / "experiment_info.json")
    print("========================================\n")


def independent_val(weights):


    # ============================================================
    # Windows DyHead validation fix
    # Disable THOP FLOPs profiling only.
    # Does NOT modify DyHead forward or trained weights.
    # ============================================================
    import ultralytics.utils.torch_utils as torch_utils

    torch_utils.get_flops = lambda *args, **kwargs: 0.0

    from ultralytics.nn.tasks import BaseModel

    BaseModel.info = lambda self, *args, **kwargs: None

    set_random_seed(SEED)
    kind, alpha = check_environment()
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    inspect_checkpoint_alpha(weights, kind, alpha)
    model = YOLO(str(weights))
    check_graph_and_alpha(model.model, kind, alpha, "val-loaded")
    folder = PROJECT_DIR / "independent_val" / (weights.parent.parent.name + "_aligned_val")
    suffix = 2
    root = folder
    while folder.exists():
        folder = Path(str(root) + f"_run{suffix:02d}")
        suffix += 1
    result = model.val(
        data=str(DATA_YAML), split="val", imgsz=EVAL_IMGSZ,
        batch=EVAL_BATCH, rect=EVAL_RECT, half=EVAL_HALF,
        conf=EVAL_CONF, iou=EVAL_IOU, max_det=EVAL_MAX_DET,
        device=0, workers=0, augment=False, plots=False, save_json=False,
        project=str(folder.parent), name=folder.name, exist_ok=False,
    )
    ids = sorted(int(i) for i in result.box.ap_class_index)
    if ids != list(range(6)):
        raise RuntimeError(f"Not all six classes evaluated: {ids}")
    report = {
        "kind":kind, "alpha":alpha, "weights":str(weights),
        "weights_sha256":sha256(weights),
        "protocol":{"split":"val","imgsz":EVAL_IMGSZ,"batch":EVAL_BATCH,
                    "rect":EVAL_RECT,"half":EVAL_HALF,"conf":EVAL_CONF,
                    "iou":EVAL_IOU,"max_det":EVAL_MAX_DET,"seed":SEED},
        "evaluated_class_ids":ids,
        "P":float(result.box.mp),"R":float(result.box.mr),
        "mAP50":float(result.box.map50),"mAP50_95":float(result.box.map),
        "per_class_AP50_95":{name:float(result.box.maps[i]) for i,name in enumerate(CLASS_NAMES)},
    }
    dest=folder/"aligned_val_metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("[ALIGNED_VAL_COMPLETE]", json.dumps({k:report[k] for k in ("kind","alpha","P","R","mAP50","mAP50_95")}),flush=True)
    print("[SAVED]", dest, flush=True)


def preflight():
    set_random_seed(SEED)
    kind, alpha = check_environment()
    model=YOLO(str(MODEL_YAML))
    model.load(str(PRETRAINED_WEIGHTS))
    value=check_graph_and_alpha(model.model,kind,alpha,"check-only")
    confirm_pretrain(model)
    print("[CHECK_PASS]",kind,"alpha=",value,"no training",flush=True)


def main():
    p=argparse.ArgumentParser(description="Trainv11-UP style 300-epoch three-architecture comparison")
    p.add_argument("--check",action="store_true",help="Preflight only. Does not train.")
    p.add_argument("--val-only",action="store_true",help="Independent, aligned APDD-Night Val; requires --weights")
    p.add_argument("--weights",type=str,default=None,help="Path to a trained best.pt for --val-only")
    a=p.parse_args()
    if a.check and a.val_only:
        p.error("Choose --check OR --val-only")
    if a.check:
        preflight()
    elif a.val_only:
        if not a.weights:
            p.error("--val-only needs --weights path")
        independent_val(a.weights)
    else:
        train()


if __name__ == "__main__":
    main()

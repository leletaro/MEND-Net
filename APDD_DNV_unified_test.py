# -*- coding: utf-8 -*-
"""Unified read-only APDD-DNV TEST for three trained models x Night/Mix/Day.

Designed for Windows, Ultralytics 8.4.3, isolated E10 fork.
Run from/put in the ultralytics-mendnet-e10-isolated project root.
No training, model selection, weight editing or changes to source annotations.

Default: ALL THREE trained datasets: Night TEST_ALL_192, Mix TEST_ALL_215
with REAL_NIGHT_60 and subdomains, and Day TEST_ALL_155.
All evaluation is split=test only; no training or checkpoint modification.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# ---------- User-editable paths / run names ----------
DATA_ROOTS = {
    "night": Path(r"F:\Yipeng_File\Data\APDD-DNV\APDD-Night"),
    "mix":   Path(r"F:\Yipeng_File\Data\APDD-DNV\APDD-MultiIllum"),
    "day":   Path(r"F:\Yipeng_File\Data\APDD-DNV\APDD-Day"),
}
RUN_DIRS = {
    "night": BASE_DIR / "runs" / "APDD_DNV_RenewMulti",
    "mix": BASE_DIR / "runs" / "APDD_DNV_MultiIllum_V2",
    "day": BASE_DIR / "runs" / "APDD_DNV_Day_V2",
}
RUN_NAMES = {
    "night": {
        "yolov8": "yolov8n_Night_seed123_300e",
        "dyhead": "yolov8-dyhead-a050_Night_seed123_300e",
        "mendnet": "MENDNet_MLLA_LSKA_DyHead-a075_Night_seed123_300e",
    },
    "mix": {
        "yolov8": "yolov8n_Mix_seed42_300e",
        "dyhead": "DyHead_a050_Mix_seed42_300e",
        "mendnet": "MENDNet_a075_Mix_seed42_300e",
    },
    "day": {  # Day seed42/300ep training completed
        "yolov8": "yolov8n_Day_seed42_300e",
        "dyhead": "DyHead_a050_Day_seed42_300e",
        "mendnet": "MENDNet_a075_Day_seed42_300e",
    },
}
# Explicit checkpoint overrides, if a run used a different name or _run02 suffix.
WEIGHT_OVERRIDES = {
    # ("mix", "dyhead"): r"F:\...\weights\best.pt",
}
REPORT_ROOT = BASE_DIR / "runs" / "APDD_DNV_3Models_Unified_TEST"
# ---------- End of editable configuration ----------

EXPECTED_SPLITS = {"night": (846, 192, 192), "mix": (935, 80, 215), "day": (672, 155, 155)}
CLASSES = ["Joint_deterioration", "Corner_break", "Crack",
           "Grouting_patching", "Corner_pothole", "Localized_patching"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIX_VIEWS = {
    "TEST_ALL": 215,
    "TEST_REAL_NIGHT": 60,
    "TEST_NIGHT_AB": 20,
    "TEST_NIGHT_C": 40,
    "TEST_DAY_AB": 155,
}
OVERALL_FIELDS = ["Dataset", "Model", "View", "Status", "Images", "Instances", "ClassesEvaluated",
                  "P", "R", "F1_macro", "mAP50", "mAP50_95", "Inference_ms_per_image",
                  "Checkpoint", "CheckpointSHA256", "Alpha", "Protocol", "Provenance", "Error"]
PERCLASS_FIELDS = ["Dataset", "Model", "View", "ClassID", "Class", "GTInstances", "PositiveImages",
                   "P", "R", "F1", "AP50", "AP50_95", "Checkpoint"]
_HASH_CACHE = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    p = Path(path)
    stat = p.stat()
    key = (str(p.resolve()).casefold(), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(2**20), b""):
                h.update(block)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def images(folder):
    p = Path(folder)
    require(p.is_dir(), f"Missing image directory: {p}")
    return sorted((x.resolve() for x in p.rglob("*") if x.is_file() and x.suffix.lower() in EXTS),
                  key=lambda x: x.as_posix().casefold())


def mix_domain(filename):
    t = filename.lower()
    if t.startswith(("apdd-day_a_", "apdd-day_b_")):
        return "TEST_DAY_AB"
    if t.startswith(("apdd-night_a_", "apdd-night_b_")):
        return "TEST_NIGHT_AB"
    if t.startswith("apdd-night_c_"):
        return "TEST_NIGHT_C"
    return None


def load_yaml(file):
    import yaml
    with Path(file).open("r", encoding="utf-8-sig") as f:
        obj = yaml.safe_load(f)
    require(isinstance(obj, dict), f"YAML is not a mapping: {file}")
    return obj


def resolve_yaml_images(cfg_path, split):
    cfg = load_yaml(cfg_path)
    p = Path(str(cfg.get("path", cfg_path.parent)))
    if not p.is_absolute():
        p = (cfg_path.parent / p).resolve()
    value = cfg.get(split)
    require(isinstance(value, (str, list)), f"Missing '{split}' in {cfg_path}")
    all_paths = []
    for v in [value] if isinstance(value, str) else value:
        part = Path(str(v))
        part = part if part.is_absolute() else (p / part).resolve()
        if part.is_dir():
            all_paths.extend(images(part))
        elif part.is_file() and part.suffix.lower() == ".txt":
            for line in part.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                item = Path(line.strip())
                all_paths.append((item if item.is_absolute() else part.parent/item).resolve())
        elif part.is_file() and part.suffix.lower() in EXTS:
            all_paths.append(part.resolve())
        else:
            raise FileNotFoundError(f"Cannot resolve {split}: {part} in {cfg_path}")
    require(len(set(all_paths)) == len(all_paths), f"Duplicate {split} images in {cfg_path}")
    return all_paths


def checkpoint_for(dataset, model):
    override = WEIGHT_OVERRIDES.get((dataset, model))
    if override:
        return Path(override).resolve()
    return (RUN_DIRS[dataset] / RUN_NAMES[dataset][model] / "weights" / "best.pt").resolve()


def dataset_root(dataset):
    """Prefer the actual training dataset YAML path if it exists.

    This handles local Day directory names without touching the training code.
    """
    config_names = {"night": "APDD-DNV_night.yaml", "mix": "APDD-DNV_mix.yaml",
                    "day": "APDD-DNV_day.yaml"}
    yaml_path = BASE_DIR / config_names[dataset]
    if yaml_path.is_file():
        config = load_yaml(yaml_path)
        declared = config.get("path")
        if declared:
            resolved = Path(str(declared))
            if not resolved.is_absolute():
                resolved = (yaml_path.parent / resolved).resolve()
            if resolved.is_dir():
                print(f"[DATA_ROOT_FROM_YAML] {dataset}: {resolved}", flush=True)
                return resolved
    return DATA_ROOTS[dataset].resolve()


def audit_dataset(dataset):
    root = dataset_root(dataset)
    require(root.is_dir(), f"Missing {dataset} dataset root: {root}")
    require(not (root / "BUILD_INCOMPLETE.txt").exists(), f"Incomplete dataset: {root}")
    splits = {k: images(root / "images" / k) for k in ("train", "val", "test")}
    counts = tuple(len(splits[k]) for k in ("train", "val", "test"))
    require(counts == EXPECTED_SPLITS[dataset],
            f"{dataset} wrong split counts {counts}, expected {EXPECTED_SPLITS[dataset]}")
    # Exact filename and content duplication checks among splits.
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common_names = {p.name.casefold() for p in splits[left]} & {p.name.casefold() for p in splits[right]}
        require(not common_names, f"{dataset} {left}/{right} overlap filenames: {list(common_names)[:5]}")
    sha_cache = {}
    def digest(p):
        if p not in sha_cache:
            sha_cache[p] = sha(p)
        return sha_cache[p]
    hashed = {k: {digest(p) for p in paths} for k, paths in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        require(not (hashed[left] & hashed[right]),
                f"{dataset} {left}/{right} identical image file hashes: check data leakage")
    test = splits["test"]
    require(len({p.stem.lower() for p in test}) == len(test), "Duplicate TEST stems")
    views = {"TEST_ALL": test}
    if dataset == "mix":
        groups = {name: [] for name in ("TEST_DAY_AB", "TEST_NIGHT_AB", "TEST_NIGHT_C")}
        unmatched = []
        for p in test:
            domain = mix_domain(p.name)
            if domain is None:
                unmatched.append(p.name)
            else:
                groups[domain].append(p)
        require(not unmatched, f"Unknown Mix TEST provenance prefix: {unmatched[:10]}")
        views.update(groups)
        views["TEST_REAL_NIGHT"] = sorted(groups["TEST_NIGHT_AB"] + groups["TEST_NIGHT_C"])
        for k, n in MIX_VIEWS.items():
            require(len(views[k]) == n, f"Mix {k} count mismatch: {len(views[k])} vs {n}")
        require(set(groups["TEST_DAY_AB"]) | set(views["TEST_REAL_NIGHT"]) == set(test),
                "Mix daytime+real nighttime do not partition TEST_ALL")
    # Label instances are audited without altering label content. Missing .txt allowed for true backgrounds.
    annotations = {}
    sig_rows = []
    for view, subset in views.items():
        cls_count, positive_count = Counter(), Counter()
        missing_labels = 0
        for img in subset:
            label_file = root / "labels" / "test" / f"{img.stem}.txt"
            if not label_file.is_file():
                missing_labels += 1
                labhash = "NO_LABEL_BACKGROUND"
                lines = []
            else:
                lines = label_file.read_text(encoding="utf-8-sig").splitlines()
                labhash = sha(label_file)
            if view == "TEST_ALL":
                sig_rows.append(f"{img.name}|{digest(img)}|{labhash}")
            found = set()
            for n, raw in enumerate(lines, 1):
                tokens = raw.strip().split()
                if not tokens:
                    continue
                require(len(tokens) == 5, f"Invalid YOLO annotation {label_file} line {n}")
                cid = int(tokens[0]); coords = [float(x) for x in tokens[1:]]
                require(cid in range(6), f"Bad class ID in {label_file}: {cid}")
                require(all(math.isfinite(z) and 0 <= z <= 1 for z in coords) and coords[2] > 0 and coords[3] > 0,
                        f"Bad coordinates {label_file}:{n}")
                cls_count[cid] += 1; found.add(cid)
            for cid in found:
                positive_count[cid] += 1
        annotations[view] = {
            "images": len(subset), "instances": sum(cls_count.values()),
            "missing_label_backgrounds": missing_labels,
            "gt": {str(k): cls_count[k] for k in range(6)},
            "positive_images": {str(k): positive_count[k] for k in range(6)},
        }
        print(f"[DATA_AUDIT] {dataset:5s} {view:18s} images={len(subset):3d} "
              f"objects={sum(cls_count.values()):3d} GT={[cls_count[k] for k in range(6)]}", flush=True)
    # C nighttime domain has only Joint/Corner/Crack GT in confirmed V2, as in supplied script.
    if dataset == "mix":
        cgt = annotations["TEST_NIGHT_C"]["gt"]
        require(all(cgt[str(i)] == 0 for i in (3, 4, 5)), "Unexpected Night-C class taxonomy")
        require(all(cgt[str(i)] > 0 for i in (0, 1, 2)), "Night-C missing expected GT class")
    dataset_sig = hashlib.sha256("\n".join(sig_rows).encode()).hexdigest()
    print(f"[SPLITS_PASS] {dataset} Train/Val/Test={counts}; "
          f"no identical image files; scene-level independence not established", flush=True)
    return root, views, annotations, hashed, dataset_sig


def audit_provenance(dataset, model, checkpoint, test_images):
    """Report training provenance; exact overlap blocks, unknown never called verified."""
    path = checkpoint.parent.parent / "args.yaml"
    meta = {"status": "UNVERIFIED", "args_yaml": str(path), "training_data_yaml": None,
            "train": None, "val": None, "overlap": 0, "notes": []}
    if not path.is_file():
        meta["notes"].append("Missing training args.yaml")
        return meta
    try:
        cfg = load_yaml(path)
        data_arg = cfg.get("data")
        require(data_arg, "Missing training data in args.yaml")
        data_yaml = Path(str(data_arg))
        if not data_yaml.is_absolute():
            choices = [BASE_DIR / data_yaml, checkpoint.parent.parent / data_yaml]
            data_yaml = next((p for p in choices if p.is_file()), choices[0])
        require(data_yaml.is_file(), f"Training dataset YAML unavailable: {data_yaml}")
        meta["training_data_yaml"] = str(data_yaml)
        tr = resolve_yaml_images(data_yaml, "train")
        va = resolve_yaml_images(data_yaml, "val")
        meta["train"], meta["val"] = len(tr), len(va)
        known = tr + va
        names = {p.name.lower() for p in known}
        hashes = {sha(p) for p in known}
        overlap = [p.name for p in test_images if p.name.lower() in names or sha(p) in hashes]
        meta["overlap"] = len(overlap)
        if overlap:
            meta["status"] = "OVERLAP"
            meta["notes"].append(f"TEST matches train/val: {overlap[:8]}")
        elif (len(tr), len(va)) == EXPECTED_SPLITS[dataset][:2]:
            meta["status"] = "FILE_LEVEL_NO_OVERLAP"
        else:
            meta["status"] = "SPLIT_MISMATCH"
            meta["notes"].append(f"Train/Val={len(tr)}/{len(va)} not {EXPECTED_SPLITS[dataset][:2]}")
        meta["notes"].append("Filename/hash verification does not prove scene-level independence")
    except Exception as e:
        meta["notes"].append(f"{type(e).__name__}: {e}")
        meta["status"] = "UNVERIFIED"
    return meta


def write_yaml_views(dataset, root, views, output):
    import yaml
    folder = output / "_test_domain_views" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    config_paths = {}
    for name, subset in views.items():
        if name == "TEST_ALL":
            selector = "images/test"
        else:
            txt = folder / (name + ".txt")
            content = "\n".join(p.as_posix() for p in subset) + "\n"
            if txt.is_file():
                require(txt.read_text(encoding="utf-8") == content,
                        f"Refuse to overwrite changed domain image list: {txt}")
            else:
                txt.write_text(content, encoding="utf-8")
            selector = txt.resolve().as_posix()
        path = folder / (name + ".yaml")
        cfg = {"path": root.as_posix(), "train": "images/train", "val": "images/val",
               "test": selector, "nc": 6, "names": dict(enumerate(CLASSES))}
        raw = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        if path.is_file():
            require(path.read_text(encoding="utf-8") == raw,
                    f"Refuse to overwrite changed view YAML: {path}")
        else:
            path.write_text(raw, encoding="utf-8")
        resolved = resolve_yaml_images(path, "test")
        require(set(resolved) == set(subset), f"View YAML resolves incorrectly: {path}")
        config_paths[name] = path
    return config_paths


def check_checkpoint_structure(checkpoint, model):
    import torch
    try:
        data = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(str(checkpoint), map_location="cpu")
    nn_model = data.get("ema") or data.get("model") if isinstance(data, dict) else None
    require(nn_model is not None, f"No model or EMA inside {checkpoint}")
    names = getattr(nn_model, "names", {})
    observed_names = [names.get(i, names.get(str(i))) if isinstance(names, dict)
                      else names[i] for i in range(6)]
    require(observed_names == CLASSES,
            f"{model}: checkpoint category names/order incorrect: {observed_names}")
    layers = list(nn_model.model)
    target_len = {"yolov8": 23, "dyhead": 27, "mendnet": 28}[model]
    require(len(layers) == target_len, f"{model}: expected {target_len} layers, got {len(layers)}")
    last = layers[-1]
    target_alpha = {"yolov8": None, "dyhead": .50, "mendnet": .75}[model]
    if model == "yolov8":
        require(type(last).__name__ == "Detect", "Native YOLO Detect head not found")
        return None
    require(type(last).__name__ == "DynamicHead", "DynamicHead not found")
    expected_f = [19, 22, 25] if model == "dyhead" else [19, 22, 26]
    require(list(last.f) == expected_f, f"Unexpected DyHead feature inputs {last.f}")
    blocks = list(last.dyhead_tower)
    require(len(blocks) == 3, "Expected three DyHead blocks")
    alpha = [float(getattr(b, "cross_scale_alpha", -999)) for b in blocks]
    require(all(abs(a - target_alpha) <= 1e-8 for a in alpha),
            f"Wrong alpha in {checkpoint}: {alpha} expected {target_alpha}")
    if model == "mendnet":
        require(type(layers[8]).__name__ == "C2fMLLAHybrid" and type(layers[26]).__name__ == "LSKA",
                "MEND-Net MLLA/LSKA architecture mismatch")
    return target_alpha


def metrics_to_rows(dataset, model, view, stat, metrics, checkpoint, chksha, alpha, protocol, provenance):
    b = metrics.box
    idxs = {int(v): i for i, v in enumerate(b.ap_class_index)}
    gt_present = {int(k) for k, v in stat["gt"].items() if v > 0}
    require(set(idxs) == gt_present, f"{dataset}/{view}/{model} mismatch: "
            f"AP IDs={list(idxs)}, ground-truth IDs={sorted(gt_present)}")
    per_class = []
    for cid, name in enumerate(CLASSES):
        i = idxs.get(cid)
        row = {"Dataset": dataset, "Model": model, "View": view, "ClassID": cid,
               "Class": name, "GTInstances": stat["gt"][str(cid)],
               "PositiveImages": stat["positive_images"][str(cid)],
               "P": "", "R": "", "F1": "", "AP50": "", "AP50_95": "",
               "Checkpoint": str(checkpoint)}
        if i is not None:
            p, r, ap50, ap = b.class_result(i)
            row.update({"P": float(p), "R": float(r), "F1": float(b.f1[i]),
                        "AP50": float(ap50), "AP50_95": float(ap)})
        per_class.append(row)
    if gt_present:
        require(abs(sum(float(p["AP50_95"]) for p in per_class
                        if p["AP50_95"] != "") / len(gt_present) - float(b.map)) < 1e-7,
                "Per-class AP average does not match overall map")
    speed = metrics.speed if isinstance(getattr(metrics, "speed", None), dict) else {}
    overall = {"Dataset": dataset, "Model": model, "View": view, "Status": "OK",
               "Images": stat["images"], "Instances": stat["instances"],
               "ClassesEvaluated": len(idxs), "P": float(b.mp), "R": float(b.mr),
               "F1_macro": float(sum(float(q) for q in b.f1)/len(b.f1)) if len(b.f1) else 0.0,
               "mAP50": float(b.map50), "mAP50_95": float(b.map),
               "Inference_ms_per_image": float(speed.get("inference", 0)),
               "Checkpoint": str(checkpoint), "CheckpointSHA256": chksha,
               "Alpha": alpha if alpha is not None else "N/A", "Protocol": protocol,
               "Provenance": provenance, "Error": ""}
    return overall, per_class


def save_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse():
    p = argparse.ArgumentParser(description="APDD-DNV 3-model Night/Mix/Day independent TEST")
    p.add_argument("--datasets", nargs="+", choices=DATA_ROOTS, default=["night", "mix", "day"])
    p.add_argument("--models", nargs="+", choices=("yolov8", "dyhead", "mendnet"),
                   default=["yolov8", "dyhead", "mendnet"])
    p.add_argument("--mix-views", nargs="+", choices=MIX_VIEWS, default=list(MIX_VIEWS))
    p.add_argument("--list", action="store_true", help="Only audit datasets and checkpoint paths, no GPU")
    p.add_argument("--skip-existing", action="store_true", help="Reuse identical successful result signatures")
    p.add_argument("--strict-provenance", action="store_true", help="Require verified train/val file-level independence")
    p.add_argument("--physical-gpu", default="0", help="Physical CUDA GPU index; default 0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--no-half", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    return p.parse_args()


def main():
    args = parse()
    require(args.workers >= 0 and args.batch >= 1, "Invalid workers/batch")
    require(len(set(args.datasets)) == len(args.datasets), "Duplicate dataset in --datasets")
    require(len(set(args.models)) == len(args.models), "Duplicate model in --models")
    require(len(set(args.mix_views)) == len(args.mix_views), "Duplicate view in --mix-views")
    # Keep Night+Mix and future Day reports independent; never overwrite prior summaries.
    report = args.report_root.resolve() / ("_".join(args.datasets))
    audit_data = {}
    tasks = []
    # No TEST inference until audit of all selected datasets/checkpoints succeeds.
    for ds in args.datasets:
        root, views, gt, split_hashes, dataset_sha = audit_dataset(ds)
        selected_views = args.mix_views if ds == "mix" else ["TEST_ALL"]
        audit_data[ds] = (root, views, gt, dataset_sha)
        for model in args.models:
            path = checkpoint_for(ds, model)
            if not path.is_file():
                print(f"[MISSING] {ds}/{model}: {path}", flush=True)
                if not args.list:
                    raise FileNotFoundError(f"Missing trained checkpoint for {ds}/{model}: {path}")
                continue
            provenance = audit_provenance(ds, model, path, views["TEST_ALL"])
            print(f"[WEIGHT] {ds:5s}/{model:7s} {path}")
            print(f"         provenance={provenance['status']} train/val={provenance['train']}/{provenance['val']}")
            for note in provenance["notes"]:
                print(f"         {note}")
            require(provenance["status"] != "OVERLAP", f"TEST leakage for {ds}/{model}")
            if args.strict_provenance and provenance["status"] != "FILE_LEVEL_NO_OVERLAP":
                raise RuntimeError(f"Unverified training split {ds}/{model}: {provenance}")
            for view in selected_views:
                tasks.append((ds, model, view, path, provenance))
    if args.list:
        print("[LIST_COMPLETE] No model loaded, no TEST inference, no files modified")
        return
    # CUDA env must be set before importing torch/ultralytics.
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if str(args.physical_gpu).lower() == "cpu" else str(args.physical_gpu)
    import torch
    import ultralytics
    require(Path(ultralytics.__file__).resolve().parent.parent == BASE_DIR.resolve(),
            f"Wrong Ultralytics fork: {ultralytics.__file__}")
    require(ultralytics.__version__ == "8.4.3", "Unexpected Ultralytics version")
    # THOP/DCNv2 Windows native access violation fix -- ONLY suppress FLOPs profiling.
    import ultralytics.utils.torch_utils as tu
    from ultralytics.nn.tasks import BaseModel
    tu.get_flops = lambda *a, **k: 0.0
    BaseModel.info = lambda self, *a, **k: None
    print("[FLOPS_PROFILE_DISABLED] No change to model forward or metrics", flush=True)
    from ultralytics import YOLO
    device = "cpu" if str(args.physical_gpu).lower() == "cpu" else 0
    if device == 0:
        require(torch.cuda.is_available(), "CUDA unavailable")
        print("[GPU]", torch.cuda.get_device_name(0))
    report.mkdir(parents=True, exist_ok=True)
    domain_yamls = {}
    for ds in args.datasets:
        r, v, gt, ds_sha = audit_data[ds]
        domain_yamls[ds] = write_yaml_views(ds, r, v, report)
    options = {"split": "test", "imgsz": 896, "batch": args.batch,
               "rect": True, "half": not args.no_half, "conf": 0.001,
               "iou": 0.7, "max_det": 300, "workers": args.workers,
               "augment": False, "seed": 42, "deterministic": True,
               "plots": not args.no_plots}
    dump_json(report / "TEST_protocol.json", options)
    audit_report = {ds: {"fingerprint": audit_data[ds][3], "views": audit_data[ds][2],
                         "dataset_root": str(audit_data[ds][0])} for ds in args.datasets}
    dump_json(report / "TEST_dataset_audit.json", audit_report)
    dump_json(report / "TEST_model_provenance.json", {
        f"{ds}/{model}": provenance for ds, model, _view, _path, provenance in tasks
    })
    overall_records, class_records = [], []
    def write_summaries():
        save_csv(report / "TEST_overall.csv", OVERALL_FIELDS, overall_records)
        save_csv(report / "TEST_per_class.csv", PERCLASS_FIELDS, class_records)
        # A stable long-form paper summary serves all datasets.
        paper = [{"Dataset": r["Dataset"], "Model": r["Model"], "View": r["View"],
                  "GTClasses": r.get("ClassesEvaluated", ""),
                  "mAP50(%)": round(100*r["mAP50"], 4) if r["Status"] == "OK" else "",
                  "mAP50-95(%)": round(100*r["mAP50_95"], 4) if r["Status"] == "OK" else "",
                  "Status": r["Status"]} for r in overall_records]
        save_csv(report / "TEST_paper_summary.csv",
                 ["Dataset", "Model", "View", "GTClasses", "mAP50(%)", "mAP50-95(%)", "Status"], paper)
        dump_json(report / "TEST_manifest.json", {"records": overall_records,
                  "note": "Mix TEST_ALL includes its domain subsets; REAL_NIGHT=20 A/B +40 C; "
                          "C has 3 ground-truth classes."})
    for ds, model, view, path, provenance in tasks:
        data_sha = audit_data[ds][3]
        ck_sha = sha(path)
        yml = domain_yamls[ds][view]
        ident = {"dataset": ds, "model": model, "view": view, "checkpoint_sha256": ck_sha,
                 "dataset_fingerprint": data_sha, "yaml": str(yml), "protocol": options,
                 "source": str(BASE_DIR), "ultralytics": ultralytics.__version__}
        signature = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
        output = report / ds / model / view
        record_file = output / "record.json"
        if args.skip_existing and record_file.is_file():
            cached = json.loads(record_file.read_text(encoding="utf-8"))
            if cached.get("signature") == signature and cached.get("overall", {}).get("Status") == "OK":
                print(f"[REUSED] {ds}/{model}/{view}", flush=True)
                overall_records.append(cached["overall"])
                class_records.extend(cached["per_class"])
                write_summaries()
                continue
        if output.exists():
            index = 2
            while (report / ds / model / (view + f"_rerun{index:02d}")).exists():
                index += 1
            output = report / ds / model / (view + f"_rerun{index:02d}")
        output.mkdir(parents=True, exist_ok=False)
        print(f"\n[TEST_START] {ds}/{model}/{view}: images={len(audit_data[ds][1][view])}", flush=True)
        obj = metrics = None
        try:
            alpha = check_checkpoint_structure(path, model)
            obj = YOLO(str(path))
            result = obj.val(
                data=str(yml), split="test", imgsz=options["imgsz"],
                batch=options["batch"], rect=options["rect"], half=options["half"],
                conf=options["conf"], iou=options["iou"], max_det=options["max_det"],
                device=device, workers=options["workers"], augment=False,
                seed=options["seed"], deterministic=True, plots=options["plots"],
                save_txt=False, save_json=False,
                project=str(output.parent), name=output.name, exist_ok=True,
                verbose=True)
            if result is None:
                raise RuntimeError("Ultralytics returned no metrics")
            overall, per_class = metrics_to_rows(ds, model, view, audit_data[ds][2][view],
                                                 result, path, ck_sha, alpha,
                                                 "aligned896_batch32_rect_half", provenance["status"])
            print(f"[TEST_DONE] {ds}/{model}/{view} "
                  f"P={overall['P']:.6f} R={overall['R']:.6f} "
                  f"mAP50={overall['mAP50']:.6f} mAP50-95={overall['mAP50_95']:.6f}", flush=True)
        except Exception as e:
            overall = {k: "" for k in OVERALL_FIELDS}
            overall.update({"Dataset": ds, "Model": model, "View": view, "Status": "ERROR",
                            "Checkpoint": str(path), "CheckpointSHA256": ck_sha,
                            "Error": f"{type(e).__name__}: {e}"})
            per_class = []
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[TEST_ERROR] {ds}/{model}/{view} {e}", flush=True)
        finally:
            del obj, metrics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rec = {"signature": signature, "created_at": datetime.now().isoformat(),
               "identity": ident, "overall": overall, "per_class": per_class}
        dump_json(output / "record.json", rec)
        overall_records.append(overall); class_records.extend(per_class)
        write_summaries()
    successful = sum(r["Status"] == "OK" for r in overall_records)
    print(f"\n[TEST_FINISHED] {successful}/{len(overall_records)} successful")
    print("[REPORTS]", report)
    print("[NOTE] TEST_REAL_NIGHT is a subset of TEST_ALL, NOT an independent second test set")
    if successful != len(overall_records):
        sys.exit(2)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
APDD-DNV V2 | Unified independent TEST evaluator (log-grounded checkpoint mapping).

Run from a Windows PowerShell terminal:
    python .\APDD_DNV_All_Models_Test.py --list
    python .\APDD_DNV_All_Models_Test.py
    python .\APDD_DNV_All_Models_Test_FIXED.py --skip-existing
    python .\APDD_DNV_All_Models_Test.py --models yolo26n --datasets day

Important: The pretrained weights under Model/*.pt are never evaluated.
Only each training run's weights/best.pt is accepted.
22 training run locations were transcribed from the previously provided
"原装系列-day和night(1).txt" Results saved to lines.
The two remaining entries (yolo26s-day / yolo26s-night) need discovery.

Requires: ultralytics (same source/version as training), torch, pyyaml.
Uses only the Python standard library for report files (Excel-readable CSV + JSON).
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

# Place the script in F:\Yipeng_File\Code\ultralytics-main26
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = BASE_DIR / "runs"
REPORT_DIR = DEFAULT_RUNS_DIR / "APDD_DNV_TEST_AllModels_V2"

# Fill only for an ambiguous/missing auto-discovered training run.
# Example: ("yolo26n", "day"): r"F:\...\runs\...\weights\best.pt",
CHECKPOINT_OVERRIDES = {
}

# Explicit training-run folders recovered from the user's original VAL log:
# "原装系列-day和night(1).txt". Its 22 `Results saved to` lines point
# to these folders, NOT to the initial Model/*.pt pretrained weights.
# The `day`/`night`/`mix` token immediately after the model name denotes
# the TRAINED domain. The `MultiIllum_seed42` suffix is shared by ALL
# experiment names and is not evidence of MultiIllum training.
LOG_RECORDED_RUNS = {
    # APDD-Day: seven models recorded in the provided original VAL log.
    ("yolov5nu", "day"): "YOLOv5nu-day_MultiIllum_seed42",
    ("yolov8n", "day"): "YOLOv8n-day_MultiIllum_seed42",
    ("yolov10n", "day"): "YOLOv10n-day_MultiIllum_seed42",
    ("yolo11n", "day"): "YOLO11n-day_MultiIllum_seed42",
    ("yolo12n", "day"): "YOLO12n-day_MultiIllum_seed42",
    ("yolo26n", "day"): "YOLO26n-day_MultiIllum_seed42",
    ("rtdetr-l", "day"): "rtdetr-l-day_MultiIllum_seed42",
    # APDD-Night: seven models recorded in the provided original VAL log.
    ("yolov5nu", "night"): "yolov5nu-night_MultiIllum_seed42",
    ("yolov8n", "night"): "yolov8n-night_MultiIllum_seed42",
    ("yolov10n", "night"): "yolov10n-night_MultiIllum_seed42",
    ("yolo11n", "night"): "yolo11n-night_MultiIllum_seed42",
    ("yolo12n", "night"): "yolo12n-night_MultiIllum_seed42",
    ("yolo26n", "night"): "yolo26n-night_MultiIllum_seed42",
    ("rtdetr-l", "night"): "rtdetr-l-night_MultiIllum_seed42",
    # APDD-MultiIllum: all eight models recorded in the original VAL log.
    ("yolov5nu", "multiilu"): "yolov5nu-mix_MultiIllum_seed42",
    ("yolov8n", "multiilu"): "yolov8n-mix_MultiIllum_seed42",
    ("yolov10n", "multiilu"): "yolov10n-mix_MultiIllum_seed42",
    ("yolo11n", "multiilu"): "yolo11n-mix_MultiIllum_seed42",
    ("yolo12n", "multiilu"): "yolo12n-mix_MultiIllum_seed42",
    ("yolo26n", "multiilu"): "yolo26n-mix_MultiIllum_seed42",
    ("yolo26s", "multiilu"): "yolo26s-mix_MultiIllum_seed42",
    ("rtdetr-l", "multiilu"): "rtdetr-l-mix_MultiIllum_seed42",
}
LOG_RECORDED_ROOT = DEFAULT_RUNS_DIR / "APDD_DNV_Rebuilt"

# Paths used by the supplied training script and the supplied night YAML.
# If actual file names differ, edit the three lines below.
DATA_YAMLS = {
    "day": BASE_DIR / "APDD-DNV_day.yaml",
    "night": BASE_DIR / "APDD-DNV_night.yaml",
    "multiilu": BASE_DIR / "APDD-DNV_mix.yaml",
}

# The APDD-DNV V2 independent Test splits. Fail rather than silently evaluate
# the wrong version of the dataset.
EXPECTED_TEST_IMAGES = {"day": 155, "night": 192, "multiilu": 215}
CLASS_NAMES = [
    "Joint_deterioration",
    "Corner_break",
    "Crack",
    "Grouting_patching",
    "Corner_pothole",
    "Localized_patching",
]
MODEL_IDS = [
    "yolov5nu", "yolov8n", "yolov10n", "yolo11n",
    "yolo12n", "yolo26n", "yolo26s", "rtdetr-l",
]
DOMAIN_IDS = ["day", "night", "multiilu"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

MODEL_PATTERNS = {
    "yolov5nu":  r"(?<![a-z0-9])(?:yolov?5nu|v5nu|5nu)(?![a-z0-9])",
    "yolov8n":   r"(?<![a-z0-9])(?:yolov?8n|v8n|8n)(?![a-z0-9])",
    "yolov10n":  r"(?<![a-z0-9])(?:yolov?10n|v10n|10n)(?![a-z0-9])",
    "yolo11n":   r"(?<![a-z0-9])(?:yolov?11n|v11n|11n)(?![a-z0-9])",
    "yolo12n":   r"(?<![a-z0-9])(?:yolov?12n|v12n|12n)(?![a-z0-9])",
    "yolo26n":   r"(?<![a-z0-9])(?:yolov?26n|v26n|26n)(?![a-z0-9])",
    "yolo26s":   r"(?<![a-z0-9])(?:yolov?26s|v26s|26s)(?![a-z0-9])",
    "rtdetr-l":   r"(?<![a-z0-9])(?:rt[-_]?detr[-_]?l)(?![a-z0-9])",
}

OVERALL_COLUMNS = [
    "Model", "Dataset", "Experiment", "Status", "Images", "Instances",
    "P(%)", "R(%)", "F1_macro(%)", "mAP50(%)", "mAP50-95(%)",
    "Inference(ms/image)", "Checkpoint", "DataYAML", "TestImagesDir",
    "ResultDir", "Error",
]
PER_CLASS_COLUMNS = [
    "Model", "Dataset", "ClassID", "Class", "Images", "Instances",
    "P(%)", "R(%)", "F1(%)", "AP50(%)", "AP50-95(%)",
    "Checkpoint", "ResultDir",
]


def parse_args():
    ap = argparse.ArgumentParser(description="APDD-DNV all-model independent Test validation")
    ap.add_argument("--list", action="store_true", help="Audit files/weight mapping only; do not run inference")
    ap.add_argument("--models", nargs="+", choices=MODEL_IDS, default=MODEL_IDS)
    ap.add_argument("--datasets", nargs="+", choices=DOMAIN_IDS, default=DOMAIN_IDS)
    ap.add_argument("--search-root", action="append", default=[],
                    help="Additional runs folder (can be repeated)")
    ap.add_argument("--skip-existing", action="store_true", help="Reuse successful matching per-run JSON")
    ap.add_argument("--allow-count-mismatch", action="store_true",
                    help="Disable strict V2 test image count check")
    ap.add_argument("--physical-gpu", default="auto",
                    help="auto, cpu, 0, 1, etc. (sets CUDA_VISIBLE_DEVICES before torch import)")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def load_yaml(path):
    import yaml
    with path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    return config if isinstance(config, dict) else {}


def detect_model(text):
    matches = [m for m, p in MODEL_PATTERNS.items() if re.search(p, str(text).lower())]
    return matches[0] if len(matches) == 1 else None


def detect_domain(text):
    """Read explicit training domain, not the shared _MultiIllum_seed42 suffix.

    Crucial: 'YOLO26n-day_MultiIllum_seed42' means DAY and
    'yolo26n-mix_MultiIllum_seed42' means MULTIILLUM.
    """
    # Strip Windows/POSIX parents and extensions before interpreting tokens.
    s = str(text).replace("\\", "/").rsplit("/", 1)[-1].lower()
    s = re.sub(r"\.(?:ya?ml|pt)$", "", s)
    # When available, favor explicit day/night/mix tokens (regardless of suffix).
    # `day` or `night` is only accepted as a delimited token.
    domain_tokens = re.findall(r"(?<![a-z0-9])(?:day|night|mix)(?![a-z0-9])", s)
    if domain_tokens:
        return {"day": "day", "night": "night", "mix": "multiilu"}[domain_tokens[0]]
    if re.search(r"(?<![a-z0-9])(?:multi[-_]?illum\w*|multiilu\w*|mixed)(?![a-z0-9])", s):
        return "multiilu"
    if re.search(r"(?<![a-z0-9])(?:nocturnal|daylight)(?![a-z0-9])", s):
        return "night" if "nocturnal" in s else "day"
    return None


def resolve_dataset_yaml(domain):
    p = DATA_YAMLS[domain]
    if p.is_file():
        return p
    # Case-insensitive and alias fallback (does not select multiple files arbitrarily).
    matches = [x for x in BASE_DIR.glob("*.yaml")
               if "apdd" in x.name.lower() and detect_domain(x.stem) == domain]
    matches += [x for x in BASE_DIR.glob("*.yml")
                if "apdd" in x.name.lower() and detect_domain(x.stem) == domain]
    matches = sorted(set(matches), key=lambda z: str(z).lower())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple dataset YAML candidates for {domain}: {matches}; edit DATA_YAMLS")
    raise FileNotFoundError(f"Dataset YAML for {domain} not found: {p}; edit DATA_YAMLS")


def training_run_candidates(search_roots):
    """Return candidates tagged only by train args.yaml and experiment directory name."""
    found = {}
    notes = []
    for root in search_roots:
        if not root.is_dir():
            notes.append(f"Search directory absent: {root}")
            continue
        for best in root.rglob("best.pt"):
            if best.parent.name.lower() != "weights":
                continue
            run = best.parent.parent
            # Avoid scanning reports as training runs.
            if "APDD_DNV_TEST_AllModels".lower() in str(run).lower():
                continue
            args_file = run / "args.yaml"
            train_args = {}
            if args_file.is_file():
                try:
                    train_args = load_yaml(args_file)
                except Exception as e:
                    notes.append(f"Could not parse {args_file}: {e}")
            name_model = detect_model(run.name)
            args_model = detect_model(Path(str(train_args.get("model", ""))).stem)
            if name_model and args_model and name_model != args_model:
                notes.append(f"Model name disagrees with training args; skipped: {best}")
                continue
            model_id = name_model or args_model
            # Infer dataset from the actual training YAML filename first.
            train_data = str(train_args.get("data", ""))
            args_domain = detect_domain(Path(train_data).stem)
            name_domain = detect_domain(run.name)
            if args_domain and name_domain and args_domain != name_domain:
                notes.append(f"Dataset name disagrees with training args; skipped: {best}")
                continue
            domain_id = args_domain or name_domain
            if not model_id or not domain_id:
                continue
            # The baseline comparison was trained with seed=42; do not silently
            # use runs with another seed. Runs lacking seed info are permitted.
            try:
                seed_value = train_args.get("seed")
                if seed_value is not None and int(seed_value) != 42:
                    continue
                seed_match = re.search(r"seed[-_ ]?(\d+)", run.name.lower())
                if seed_match and int(seed_match.group(1)) != 42:
                    continue
            except (TypeError, ValueError):
                notes.append(f"Unclear random seed: {run}")
                continue
            key = (model_id, domain_id)
            found.setdefault(key, set()).add(best.resolve())
    return {k: sorted(v, key=str) for k, v in found.items()}, notes


def choose_checkpoint(key, candidates):
    override = CHECKPOINT_OVERRIDES.get(key)
    if override:
        p = Path(override)
        if p.name.lower() != "best.pt" or p.parent.name.lower() != "weights":
            return None, "ERROR", "Override must point to .../weights/best.pt"
        if not p.is_file():
            return None, "ERROR", f"Override checkpoint does not exist: {p}"
        return p.resolve(), "OK", "manual override"

    log_folder = LOG_RECORDED_RUNS.get(key)
    log_best = None
    if log_folder:
        log_best = LOG_RECORDED_ROOT / log_folder / "weights" / "best.pt"
        if log_best.is_file():
            return log_best.resolve(), "OK", "exact training folder from provided VAL log"

    # Fallback discovery may handle relocated runs and the two YOLO26s
    # domain models not mentioned in the supplied log.
    options = candidates.get(key, [])
    if len(options) == 1:
        return options[0], "OK", "unique auto-discovered training best.pt"
    if len(options) > 1:
        return None, "AMBIGUOUS", ("Multiple training checkpoints; specify CHECKPOINT_OVERRIDES: "
                                    + " | ".join(map(str, options)))
    if log_best is not None:
        return None, "MISSING", ("VAL log identified this training directory, but best.pt "
                                 f"is not present: {log_best}")
    return None, "MISSING", ("No original VAL log entry and no auto-discovered "
                              "training weights/best.pt (check whether the model was trained)")


def dataset_info(domain, allow_count_mismatch):
    yml = resolve_dataset_yaml(domain)
    cfg = load_yaml(yml)
    if "test" not in cfg or not cfg["test"]:
        raise ValueError(f"No test: entry in {yml}")
    root = Path(str(cfg.get("path", yml.parent)))
    if not root.is_absolute():
        root = yml.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root} ({yml})")
    test_part = cfg["test"]
    if not isinstance(test_part, str):
        raise ValueError(f"test: must be a directory string: {yml}")
    test_dir = Path(test_part)
    if not test_dir.is_absolute():
        test_dir = root / test_dir
    test_dir = test_dir.resolve()
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test image directory not found: {test_dir}")
    for other in ("train", "val"):
        if isinstance(cfg.get(other), str):
            other_path = Path(cfg[other])
            if not other_path.is_absolute():
                other_path = root / other_path
            if other_path.resolve() == test_dir:
                raise ValueError(f"{other} and test point to the same folder: {test_dir}")

    names = cfg.get("names", {})
    if isinstance(names, list):
        name_list = [str(x) for x in names]
    elif isinstance(names, dict):
        name_list = [str(names.get(i, names.get(str(i), ""))) for i in range(6)]
    else:
        raise ValueError(f"Invalid names in YAML: {yml}")
    if name_list != CLASS_NAMES:
        raise ValueError(f"Class order differs from APDD-DNV taxonomy: {yml}\n{name_list}")

    images = sorted(p for p in test_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    count = len(images)
    expected = EXPECTED_TEST_IMAGES[domain]
    if count != expected and not allow_count_mismatch:
        raise ValueError(f"{domain} Test count={count}, expected={expected}. "
                         "Check V2 split or use --allow-count-mismatch if intentional.")
    if count == 0:
        raise ValueError(f"Empty Test image folder: {test_dir}")

    # Standard Ultralytics convention: images/test/xxx.jpg -> labels/test/xxx.txt.
    try:
        relative = test_dir.relative_to(root)
        parts = relative.parts
        if "images" in parts:
            i = parts.index("images")
            label_root = root.joinpath(*parts[:i], "labels", *parts[i + 1:])
        else:
            label_root = root / "labels" / "test"
    except ValueError:
        label_root = test_dir.parent.parent / "labels" / "test"

    objects = Counter()
    class_images = Counter()
    missing_label_files = 0
    for img in images:
        label = label_root / img.relative_to(test_dir).with_suffix(".txt")
        if not label.is_file():
            missing_label_files += 1  # An empty background image may have no label file.
            continue
        classes_here = set()
        for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            fields = line.strip().split()
            if not fields:
                continue
            try:
                cid = int(float(fields[0]))
            except ValueError:
                raise ValueError(f"Invalid class in {label}:{line_no}")
            if cid not in range(6):
                raise ValueError(f"Out-of-range class {cid} in {label}:{line_no}")
            objects[cid] += 1
            classes_here.add(cid)
        for cid in classes_here:
            class_images[cid] += 1
    if sum(objects.values()) == 0:
        raise ValueError(f"No ground-truth labels found: {label_root}")
    absent = [CLASS_NAMES[i] for i in range(6) if objects[i] == 0]
    if absent:
        raise ValueError(f"Test set missing GT for class(es): {absent} in {label_root}")
    return {
        "yaml": str(yml), "test_dir": str(test_dir), "images": count,
        "instances": sum(objects.values()), "missing_labels": missing_label_files,
        "class_counts": {str(i): objects[i] for i in range(6)},
        "class_images": {str(i): class_images[i] for i in range(6)},
    }


def pct(value):
    return round(float(value) * 100.0, 4)


def get_metrics(metrics, key, info, checkpoint, result_dir):
    box = metrics.box
    p, r = float(box.mp), float(box.mr)
    valid_f1 = [float(v) for v in box.f1]
    mean_f1 = (sum(valid_f1) / len(valid_f1)) if valid_f1 else 0.0
    speed = getattr(metrics, "speed", {}) or {}
    label = key[0] + "-" + key[1]
    overall = {
        "Model": key[0], "Dataset": key[1], "Experiment": label,
        "Status": "OK", "Images": info["images"], "Instances": info["instances"],
        "P(%)": pct(p), "R(%)": pct(r), "F1_macro(%)": pct(mean_f1),
        "mAP50(%)": pct(box.map50), "mAP50-95(%)": pct(box.map),
        "Inference(ms/image)": round(float(speed.get("inference", 0.0)), 3),
        "Checkpoint": str(checkpoint), "DataYAML": info["yaml"],
        "TestImagesDir": info["test_dir"], "ResultDir": str(result_dir), "Error": "",
    }
    class_ids = [int(x) for x in getattr(box, "ap_class_index", [])]
    index = {cid: pos for pos, cid in enumerate(class_ids)}
    rows = []
    for cid, cname in enumerate(CLASS_NAMES):
        i = index.get(cid)
        rows.append({
            "Model": key[0], "Dataset": key[1], "ClassID": cid, "Class": cname,
            "Images": info["class_images"][str(cid)],
            "Instances": info["class_counts"][str(cid)],
            "P(%)": pct(box.p[i]) if i is not None else "",
            "R(%)": pct(box.r[i]) if i is not None else "",
            "F1(%)": pct(box.f1[i]) if i is not None else "",
            "AP50(%)": pct(box.ap50[i]) if i is not None else "",
            "AP50-95(%)": pct(box.ap[i]) if i is not None else "",
            "Checkpoint": str(checkpoint), "ResultDir": str(result_dir),
        })
    return overall, rows


def save_csv(path, columns, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_reports(results):
    overall_rows = [r["overall"] for r in results]
    class_rows = [cls for r in results for cls in r.get("classes", [])]
    save_csv(REPORT_DIR / "TEST_overall_24.csv", OVERALL_COLUMNS, overall_rows)
    save_csv(REPORT_DIR / "TEST_per_class.csv", PER_CLASS_COLUMNS, class_rows)
    metrics_names = ["P(%)", "R(%)", "F1_macro(%)", "mAP50(%)", "mAP50-95(%)"]
    wide_columns = ["Model"] + [f"{dom}_{metric}" for dom in DOMAIN_IDS for metric in metrics_names]
    index = {(r["overall"]["Model"], r["overall"]["Dataset"]): r["overall"] for r in results}
    wide = []
    for m in MODEL_IDS:
        row = {"Model": m}
        for d in DOMAIN_IDS:
            source = index.get((m, d), {})
            for metric in metrics_names:
                row[f"{d}_{metric}"] = source.get(metric, "") if source.get("Status") == "OK" else ""
        wide.append(row)
    save_csv(REPORT_DIR / "TEST_paper_wide.csv", wide_columns, wide)
    (REPORT_DIR / "TEST_manifest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def run_config(args):
    return {
        "split": "test", "imgsz": args.imgsz, "batch": args.batch,
        "workers": args.workers, "seed": args.seed, "conf": args.conf,
        "iou": args.iou, "max_det": args.max_det,
        "plots": not args.no_plots, "physical_gpu": args.physical_gpu,
    }


def main():
    args = parse_args()
    if args.physical_gpu.lower() not in ("auto", "cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
    elif args.physical_gpu.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import ultralytics
    from ultralytics import YOLO, RTDETR

    # Prevent importing an unrelated global ultralytics installation.
    print(f"Python: {sys.executable}")
    print(f"Ultralytics: {ultralytics.__version__} @ {ultralytics.__file__}")
    print(f"Torch: {torch.__version__}")
    print(f"[LOG MAPPING] {len(LOG_RECORDED_RUNS)} experiments with paths from original VAL log")
    use_cpu = args.physical_gpu.lower() == "cpu"
    device = "cpu" if use_cpu else (0 if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        print(f"Evaluation GPU (logical cuda:0): {torch.cuda.get_device_name(0)}")
    else:
        print("Evaluation device: CPU")

    search_roots = [DEFAULT_RUNS_DIR] + [Path(x) for x in args.search_root]
    candidates, notes = training_run_candidates(search_roots)
    for note in notes:
        print("[DISCOVERY WARNING]", note)

    datasets = {}
    dataset_errors = {}
    for d in args.datasets:
        try:
            datasets[d] = dataset_info(d, args.allow_count_mismatch)
            info = datasets[d]
            print(f"[TEST DATA] {d}: {info['images']} images, "
                  f"{info['instances']} GT objects; YAML={info['yaml']}")
            if info["missing_labels"]:
                print(f"[NOTE] {d}: {info['missing_labels']} images without label files (possible backgrounds)")
        except Exception as e:
            dataset_errors[d] = f"{type(e).__name__}: {e}"
            print(f"[DATA ERROR] {d}: {dataset_errors[d]}")

    selections = {}
    for m in args.models:
        for d in args.datasets:
            key = (m, d)
            checkpoint, status, why = choose_checkpoint(key, candidates)
            selections[key] = (checkpoint, status, why)
            print(f"[MAPPING] {m:10s} {d:9s} {status:10s} "
                  f"{str(checkpoint) if checkpoint else why}")
            if status == "OK":
                print(f"          source: {why}")

    if args.list:
        print("\nAudit-only mode. LOG paths are checked against actual best.pt files.")
        print("The provided log does not contain yolo26s-day/night; they are searched independently.")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    config = run_config(args)
    for m in args.models:
        for d in args.datasets:
            key = (m, d)
            checkpoint, status, why = selections[key]
            folder = REPORT_DIR / f"{m}_{d}_TEST"
            folder.mkdir(parents=True, exist_ok=True)
            info = datasets.get(d)
            baseline = {
                "Model": m, "Dataset": d, "Experiment": f"{m}-{d}",
                "Status": status, "Images": info["images"] if info else "",
                "Instances": info["instances"] if info else "",
                "P(%)": "", "R(%)": "", "F1_macro(%)": "",
                "mAP50(%)": "", "mAP50-95(%)": "", "Inference(ms/image)": "",
                "Checkpoint": str(checkpoint) if checkpoint else "",
                "DataYAML": info["yaml"] if info else "",
                "TestImagesDir": info["test_dir"] if info else "",
                "ResultDir": str(folder), "Error": "",
            }
            if d in dataset_errors:
                baseline.update(Status="DATA_ERROR", Error=dataset_errors[d])
                results.append({"overall": baseline, "classes": []})
                write_reports(results)
                continue
            if status != "OK":
                baseline["Error"] = why
                results.append({"overall": baseline, "classes": []})
                write_reports(results)
                continue

            stat = checkpoint.stat()
            signature = {
                "checkpoint": str(checkpoint), "checkpoint_size": stat.st_size,
                "checkpoint_mtime_ns": stat.st_mtime_ns,
                "yaml": info["yaml"], "test_dir": info["test_dir"],
                "image_count": info["images"], "gt_count": info["instances"],
                "config": config,
            }
            record_path = folder / "record.json"
            if args.skip_existing and record_path.is_file():
                try:
                    cached = json.loads(record_path.read_text(encoding="utf-8"))
                    if cached.get("signature") == signature and cached.get("overall", {}).get("Status") == "OK":
                        results.append({"overall": cached["overall"], "classes": cached["classes"]})
                        print(f"[REUSED] {m}-{d}")
                        write_reports(results)
                        continue
                except (OSError, ValueError, KeyError, TypeError):
                    pass

            print("\n" + "=" * 90)
            print(f"[START] {m}-{d} | split=test | imgsz={args.imgsz} | {checkpoint}")
            print("=" * 90)
            try:
                model = RTDETR(str(checkpoint)) if m == "rtdetr-l" else YOLO(str(checkpoint))
                metrics = model.val(
                    data=info["yaml"], split="test",
                    imgsz=args.imgsz, batch=args.batch, device=device,
                    workers=args.workers, seed=args.seed, deterministic=True,
                    conf=args.conf, iou=args.iou, max_det=args.max_det,
                    augment=False, half=False, rect=True, plots=not args.no_plots,
                    save_json=False, save_txt=False, verbose=True,
                    project=str(REPORT_DIR), name=folder.name, exist_ok=True,
                )
                overall, per_class = get_metrics(metrics, key, info, checkpoint, folder)
                outcome = {"overall": overall, "classes": per_class}
                print(f"[DONE] {m}-{d}: P={overall['P(%)']:.3f}%, "
                      f"R={overall['R(%)']:.3f}%, mAP50={overall['mAP50(%)']:.3f}%, "
                      f"mAP50-95={overall['mAP50-95(%)']:.3f}%")
            except Exception as e:
                baseline.update(Status="ERROR", Error=f"{type(e).__name__}: {e}")
                outcome = {"overall": baseline, "classes": []}
                (folder / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"[ERROR] {m}-{d}: {baseline['Error']}")
            finally:
                try:
                    del metrics, model
                except NameError:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            record_path.write_text(json.dumps({
                "signature": signature, "evaluated_at": datetime.now().isoformat(),
                **outcome,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(outcome)
            write_reports(results)  # Incremental, safe if one model later fails.

    ok = sum(r["overall"]["Status"] == "OK" for r in results)
    failed = [r["overall"]["Experiment"] + ": " + r["overall"]["Status"]
              for r in results if r["overall"]["Status"] != "OK"]
    print("\n" + "=" * 90)
    print(f"FINISHED: {ok}/{len(results)} successful Test evaluations")
    if failed:
        print("NEEDS ATTENTION:")
        for item in failed:
            print("  " + item)
    print("REPORTS:")
    for name in ("TEST_overall_24.csv", "TEST_per_class.csv", "TEST_paper_wide.csv", "TEST_manifest.json"):
        print(" ", REPORT_DIR / name)


if __name__ == "__main__":
    main()

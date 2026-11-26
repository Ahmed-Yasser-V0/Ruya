"""Retrain the Egyptian currency model with the new polymer 20-pound note.

Usage (from the project folder, with the GPU-enabled .venv):
    python training/train_currency.py datasets/egycurrency datasets/new-egyptian-currency [...]

Each dataset is a Roboflow "YOLOv8" export (folders train/valid/test with
images/ and labels/, plus data.yaml). Class names differ between datasets, so
they are mapped to ONE list of classes (the original 13 + "20 pound new").
Background photos without money (COCO128) are added as negatives so phones,
pets and faces stop being read as banknotes.

At the end the new model is compared with the current currency.pt on the same
validation images. models/currency_new.pt is only copied over currency.pt if it
is better (the old model is kept as currency_old.pt).
"""
import os
import random
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "currency_merged"
CLASSES = ["0.25 pound m", "0.25 pound", "0.50 pound m", "0.50 pound", "1 pound m", "1 pound",
           "10 pound new", "10 pound", "100 pounds", "20 pounds", "200 pounds", "5 pounds",
           "50 pounds", "20 pound new"]


def map_name(raw):
    """Map any dataset's class name to one of CLASSES (or None to drop it)."""
    n = raw.lower().replace("_", " ").replace("-", " ")
    n = re.sub(r"^\d+\s*\(", "(", n)                       # "14 (20 new front)" -> "(20 new front)"
    coin = bool(re.search(r"\b(m|coin|metal|معدن)\b", n))
    new = bool(re.search(r"\b(new|polymer|plastic)\b", n))
    if re.search(r"\b(0?\.25|25 ?(piastres?|pt)|quarter)\b", n):
        return "0.25 pound m" if coin else "0.25 pound"
    if re.search(r"\b(0?\.5|0?\.50|50 ?(piastres?|pt)|half)\b", n):
        return "0.50 pound m" if coin else "0.50 pound"
    words = {"one": "1", "five": "5", "ten": "10", "twenty": "20", "fifty": "50", "hundred": "100"}
    for w, d in words.items():
        n = re.sub(rf"\b{w}\b", d, n)
    m = re.search(r"(?<![\d.])(200|100|50|20|10|5|1)(?![\d.])", n)
    if not m:
        return None
    v = m.group(1)
    if v == "1":
        return "1 pound m" if coin else "1 pound"
    if v == "10":
        return "10 pound new" if new else "10 pound"
    if v == "20":
        return "20 pound new" if new else "20 pounds"
    return {"5": "5 pounds", "50": "50 pounds", "100": "100 pounds", "200": "200 pounds"}[v]


def merge(datasets):
    shutil.rmtree(OUT, ignore_errors=True)
    for split in ("train", "val"):
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)
    stats = {c: 0 for c in CLASSES}
    for ds in map(Path, datasets):
        names = yaml.safe_load(open(ds / "data.yaml"))["names"]
        names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
        mapping = {i: map_name(n) for i, n in enumerate(names)}
        print(f"\n{ds.name}:")
        for i, n in enumerate(names):
            print(f"   {n!r:28} -> {mapping[i]}")
        for split, target in (("train", "train"), ("valid", "val"), ("test", "val")):
            img_dir = ds / split / "images"
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                lab = ds / split / "labels" / (img.stem + ".txt")
                lines = []
                if lab.exists():
                    for line in open(lab):
                        parts = line.split()
                        if len(parts) >= 5 and mapping.get(int(parts[0])):
                            cls = CLASSES.index(mapping[int(parts[0])])
                            lines.append(" ".join([str(cls)] + parts[1:5]))
                            stats[CLASSES[cls]] += 1
                if not lines:
                    continue
                name = f"{ds.name}_{split}_{img.name}"
                shutil.copy(img, OUT / target / "images" / name)
                open(OUT / target / "labels" / (Path(name).stem + ".txt"), "w").write("\n".join(lines) + "\n")
    return stats


def add_negatives():
    """COCO128: everyday photos with no money = 'background' examples."""
    from ultralytics.utils.downloads import download
    coco = ROOT / "training" / "coco128"
    if not coco.exists():
        download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", dir=ROOT / "training")
    imgs = sorted((coco / "images" / "train2017").glob("*.jpg"))
    random.seed(0)
    for i, img in enumerate(imgs):
        split = "val" if i % 8 == 0 else "train"
        shutil.copy(img, OUT / split / "images" / f"negative_{img.name}")
        open(OUT / split / "labels" / f"negative_{img.stem}.txt", "w").close()
    return len(imgs)


def main(datasets):
    stats = merge(datasets)
    print("\nBoxes per class:", stats)
    missing = [c for c, n in stats.items() if n == 0]
    if missing:
        print("WARNING: no examples for", missing)
    print("Negatives added:", add_negatives())
    data = OUT / "data.yaml"
    yaml.safe_dump({"path": str(OUT), "train": "train/images", "val": "val/images",
                    "names": {i: c for i, c in enumerate(CLASSES)}}, open(data, "w"), allow_unicode=True)

    from ultralytics import YOLO
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Training on:", torch.cuda.get_device_name(0) if device == 0 else "CPU (slow!)")
    model = YOLO(str(ROOT / "currency.pt"))          # start from the current model (transfer learning)
    model.train(data=str(data), epochs=int(os.environ.get("EPOCHS", 40)), imgsz=640, batch=-1,
                device=device, project=str(ROOT / "training" / "runs"), name="currency", exist_ok=True,
                patience=10, plots=False, workers=2)
    best = ROOT / "training" / "runs" / "currency" / "weights" / "best.pt"
    shutil.copy(best, ROOT / "models" / "currency_new.pt")
    compare(data)


def compare(data):
    """Same validation images, old vs new model. Swap only if the new one wins."""
    from ultralytics import YOLO
    new = YOLO(str(ROOT / "models" / "currency_new.pt")).val(data=str(data), plots=False, verbose=False)
    print(f"\nNEW model  mAP50: {new.box.map50:.3f}   mAP50-95: {new.box.map:.3f}")
    # the old model has no "20 pound new" class, so compare on the classes both know
    old = YOLO(str(ROOT / "currency.pt")).val(data=str(data), plots=False, verbose=False)
    print(f"OLD model  mAP50: {old.box.map50:.3f}   mAP50-95: {old.box.map:.3f}   (cannot see the new 20)")
    if new.box.map50 >= old.box.map50:
        shutil.copy(ROOT / "currency.pt", ROOT / "models" / "currency_old.pt")
        shutil.copy(ROOT / "models" / "currency_new.pt", ROOT / "currency.pt")
        print("RESULT: new model installed as currency.pt (old one saved in models/currency_old.pt)")
    else:
        print("RESULT: new model was NOT better - kept the current currency.pt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])

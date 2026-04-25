#!/usr/bin/env python3
"""
YOLOv11s Gun Detection — Combined Dataset Training
Datasets: muha/gun-soqmi v2 + Anvari CCTV + Monash Guns
Run     : modal run train_modal.py
Download: modal volume get gun-detect-v3-runs gun-v4/weights/best.pt .
"""
import modal

RUNS_VOL  = "gun-detect-v4-runs"
DATA_VOL  = "gun-detect-v2-data"   # reuse — muha & anvari already downloaded
RUNS_PATH = "/runs"
DATA_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install(
        "ultralytics>=8.3.0",
        "roboflow>=1.1.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    )
)

app      = modal.App("gun-detection-v3", image=image)
runs_vol = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)
data_vol = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
rf_secret = modal.Secret.from_name("roboflow")

_GUN_NAMES = {
    "gun", "guns", "pistol", "pistols", "rifle", "shotgun",
    "heavy gun", "long guns", "handgun", "handguns",
    "weapon", "weapons", "guns perspective", "firearm", "firearms",
}

def _is_gun(name: str) -> bool:
    return name.lower().strip() in _GUN_NAMES


def _remap_label_file(path, gun_ids):
    """Remap gun class IDs → 0, drop non-gun lines. Modifies file in place."""
    lines = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        if gun_ids is None or int(parts[0]) in gun_ids:
            lines.append("0 " + " ".join(parts[1:]))
    path.write_text("\n".join(lines))
    return bool(lines)


def _remap_dataset(ds_dir, yaml_cfg):
    """Remap all label files in a downloaded Roboflow dataset."""
    names    = yaml_cfg.get("names", [])
    gun_ids  = {i for i, n in enumerate(names) if _is_gun(n)} or None
    print(f"  gun_ids={gun_ids} → {[names[i] for i in (gun_ids or [])]}")
    for split in ("train", "valid", "test"):
        lbl_dir = ds_dir / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            _remap_label_file(lbl, gun_ids)


@app.function(
    timeout=3600 * 2,
    volumes={DATA_PATH: data_vol},
    secrets=[rf_secret],
    memory=16384,
    cpu=4,
)
def prepare_dataset():
    import hashlib, os, random, yaml
    from pathlib import Path
    from roboflow import Roboflow

    combined_dir = Path(DATA_PATH) / "combined_v3"
    yaml_out     = combined_dir / "data.yaml"

    if yaml_out.exists():
        print("Combined dataset already prepared — skipping.")
        return str(yaml_out)

    combined_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ["ROBOFLOW_API_KEY"]
    rf      = Roboflow(api_key=api_key)

    # ── Download datasets ─────────────────────────────────────────────────────
    datasets = [
        ("anvari", "cctv_anvari", "mohammadreza-anvari-h6ase", "cctv-gun-detector",       1),
        ("monash", "monash_guns", "arms",                      "the-monash-guns-dataset", 2),
    ]

    ds_dirs = {}
    for name, folder, workspace, project, version in datasets:
        ds_dir = Path(DATA_PATH) / folder
        ds_dirs[name] = ds_dir
        if ds_dir.exists():
            print(f"{name}: already downloaded — skipping.")
        else:
            print(f"Downloading {name}…")
            rf.workspace(workspace).project(project).version(version).download(
                "yolov11", location=str(ds_dir)
            )
        cfg = yaml.safe_load((ds_dir / "data.yaml").read_text())
        print(f"Remapping {name} labels…")
        _remap_dataset(ds_dir, cfg)

    # ── Collect all images + labels, deduplicate by MD5 ──────────────────────
    print("\nCollecting and deduplicating images…")
    seen_hashes = {}   # md5 → (img_path, lbl_path, original_split)
    duplicates  = 0

    split_map = {"train": "train", "valid": "val", "test": "test"}

    all_entries = []  # (img_path, lbl_path, original_split)

    for name, ds_dir in ds_dirs.items():
        for rf_split, yolo_split in split_map.items():
            img_dir = ds_dir / rf_split / "images"
            lbl_dir = ds_dir / rf_split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*"):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                    continue
                md5 = hashlib.md5(img_path.read_bytes()).hexdigest()
                if md5 in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes[md5] = True
                all_entries.append((str(img_path), str(lbl_path), yolo_split))

    print(f"Total unique images: {len(all_entries)} ({duplicates} duplicates removed)")

    # ── Re-split 80/10/10 ────────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(all_entries)
    n      = len(all_entries)
    n_val  = int(n * 0.10)
    n_test = int(n * 0.10)

    train_entries = all_entries[n_val + n_test:]
    val_entries   = all_entries[:n_val]
    test_entries  = all_entries[n_val:n_val + n_test]

    # Oversample Anvari CCTV images 5x in training to give them more weight
    CCTV_OVERSAMPLE = 5
    cctv_train = [e for e in train_entries if "cctv_anvari" in e[0]]
    train_entries = train_entries + cctv_train * (CCTV_OVERSAMPLE - 1)
    random.shuffle(train_entries)
    print(f"  Anvari CCTV oversampled {CCTV_OVERSAMPLE}x in training")

    split_entries = {
        "train": train_entries,
        "val":   val_entries,
        "test":  test_entries,
    }

    for split, entries in split_entries.items():
        print(f"  {split}: {len(entries)} images")
        txt = combined_dir / f"{split}.txt"
        txt.write_text("\n".join(e[0] for e in entries))

    yaml_out.write_text(yaml.dump({
        "path":  str(combined_dir),
        "train": "train.txt",
        "val":   "val.txt",
        "test":  "test.txt",
        "nc":    1,
        "names": ["gun"],
    }))

    data_vol.commit()
    print(f"\ndata.yaml → {yaml_out}")
    return str(yaml_out)


@app.function(
    gpu="H100",
    timeout=3600 * 6,
    volumes={DATA_PATH: data_vol, RUNS_PATH: runs_vol},
    memory=32768,
    cpu=8,
)
def train(data_yaml: str):
    from pathlib import Path
    from ultralytics import YOLO

    # Resume if interrupted
    last_ckpt = Path(RUNS_PATH) / "gun-v4" / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        weights = "yolo11s.pt"
        print(f"Starting from COCO pretrained {weights}")
        model = YOLO(weights)
        results = model.train(
            data=data_yaml,
            epochs=200,
            imgsz=640,
            batch=128,
            device=0,
            optimizer="AdamW",
            lr0=0.0001,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            patience=30,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            project=RUNS_PATH,
            name="gun-v4",
            save=True,
            save_period=5,
            cache="ram",
            workers=8,
            verbose=True,
        )

    runs_vol.commit()

    m = results.results_dict
    print(f"\nmAP50    : {m.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95 : {m.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Precision: {m.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall   : {m.get('metrics/recall(B)', 'N/A')}")
    return str(results.save_dir)


@app.local_entrypoint()
def main():
    print("Step 1/2 — preparing combined dataset…")
    data_yaml = prepare_dataset.remote()

    print("Step 2/2 — training on H100…")
    save_dir = train.remote(data_yaml)

    print(f"\nDone. Download weights:")
    print(f"  modal volume get {RUNS_VOL} gun-v4/weights/best.pt .")

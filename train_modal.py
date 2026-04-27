#!/usr/bin/env python3
"""
YOLOv11s Gun Detection — Final Model (gun-v3)
Datasets: Anvari CCTV + Monash Guns (Anvari oversampled 5x)
Weights : Fine-tuned from gun-v2 (muha-pretrained) → gun-v3
Run     : modal run train_modal.py
Download: modal volume get gun-detect-v3-runs gun-v3/weights/best.pt .
"""
import modal

RUNS_VOL  = "gun-detect-v3-runs"
DATA_VOL  = "gun-detect-v2-data"
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

app       = modal.App("gun-detection-v3", image=image)
runs_vol  = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)
data_vol  = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
rf_secret = modal.Secret.from_name("roboflow")

_GUN_NAMES = {
    "gun", "guns", "pistol", "pistols", "rifle", "shotgun",
    "heavy gun", "long guns", "handgun", "handguns",
    "weapon", "weapons", "guns perspective", "firearm", "firearms",
}

def _is_gun(name: str) -> bool:
    return name.lower().strip() in _GUN_NAMES

def _remap_label_file(path, gun_ids):
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
    names   = yaml_cfg.get("names", [])
    gun_ids = {i for i, n in enumerate(names) if _is_gun(n)} or None
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

    # Collect all images, deduplicate by MD5
    print("\nCollecting and deduplicating images…")
    seen_hashes = {}
    duplicates  = 0
    all_entries = []
    split_map   = {"train": "train", "valid": "val", "test": "test"}

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
                all_entries.append((str(img_path), str(lbl_path), name))

    print(f"Total unique images: {len(all_entries)} ({duplicates} duplicates removed)")

    # Re-split 80/10/10
    random.seed(42)
    random.shuffle(all_entries)
    n      = len(all_entries)
    n_val  = int(n * 0.10)
    n_test = int(n * 0.10)

    train_entries = all_entries[n_val + n_test:]
    val_entries   = all_entries[:n_val]
    test_entries  = all_entries[n_val:n_val + n_test]

    # Oversample Anvari CCTV 5x in training
    CCTV_OVERSAMPLE = 5
    cctv_train    = [e for e in train_entries if e[2] == "anvari"]
    train_entries = train_entries + cctv_train * (CCTV_OVERSAMPLE - 1)
    random.shuffle(train_entries)
    print(f"  Anvari CCTV oversampled {CCTV_OVERSAMPLE}x in training")

    for split, entries in [("train", train_entries), ("val", val_entries), ("test", test_entries)]:
        print(f"  {split}: {len(entries)} images")
        (combined_dir / f"{split}.txt").write_text("\n".join(e[0] for e in entries))

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

    last_ckpt = Path(RUNS_PATH) / "gun-v3" / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        # Fine-tune from gun-v2 (muha pretrained)
        prev_best = Path(RUNS_PATH) / "gun-v2" / "weights" / "best.pt"
        weights   = str(prev_best) if prev_best.exists() else "yolo11s.pt"
        print(f"Starting from {weights}")
        model = YOLO(weights)
        results = model.train(
            data=data_yaml,
            epochs=150,
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
            patience=25,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            project=RUNS_PATH,
            name="gun-v3",
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
    print("Step 1/2 — preparing dataset…")
    data_yaml = prepare_dataset.remote()

    print("Step 2/2 — training on H100…")
    save_dir = train.remote(data_yaml)

    print(f"\nDone. Download weights:")
    print(f"  modal volume get {RUNS_VOL} gun-v3/weights/best.pt .")

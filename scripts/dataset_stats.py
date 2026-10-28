"""Real dataset numbers for the report and the viva (no training, no GPU needed).

    python scripts/dataset_stats.py            # laptop: reads the zip's file list over HTTP
    python scripts/dataset_stats.py --local    # Colab: uses the downloaded 2.2 GB zip

What it measures, and from where:
  * Full PlantVillage (color): file list inside data.zip on Hugging Face.
  * Our split (200/40/60 per class, seed 42): the SAME function the training
    uses (plan_plantvillage_split), so the numbers match the Colab run exactly.
  * Leakage check: leaf groups shared between our train / val / test.
  * Image size, channels, leaf area, tensor shape: the images that are
    actually prepared in data/ on this machine.
  * PlantDoc: file list of the pinned GitHub commit + the prepared folder.
Writes results/dataset_stats.json (read by scripts/build_viva_pdfs.py).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import (OVERLAP_CLASSES, PLANTDOC_TO_PLANTVILLAGE, PV_CLASSES, HF_REPO,  # noqa: E402
                      LeafDataset, _leaf_group, _open_plantvillage_zip, get_transforms,
                      list_plantdoc_files, plan_plantvillage_split)
from src.utils import save_json  # noqa: E402


def crop_of(cls):
    return cls.split("___")[0]


def folder_stats(root):
    """Count images per class in root/<class>/ and look at their size/mode."""
    per_class, sizes, modes = {}, Counter(), Counter()
    if not os.path.isdir(root):
        return None
    for cls in sorted(os.listdir(root)):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder) or cls == "masks":
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        per_class[cls] = len(files)
        for f in files:
            with Image.open(os.path.join(folder, f)) as im:
                sizes[f"{im.size[0]}x{im.size[1]}"] += 1
                modes[im.mode] += 1
    return {"per_class": per_class, "total": sum(per_class.values()),
            "sizes": dict(sizes.most_common(10)), "modes": dict(modes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--local", action="store_true", help="download the full zip instead of remote reads")
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--val-per-class", type=int, default=40)
    parser.add_argument("--test-per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/dataset_stats.json")
    args = parser.parse_args()
    stats = {"source": {"plantvillage": f"huggingface.co/datasets/{HF_REPO} (data.zip, color)",
                        "plantdoc": "github.com/pratikkayal/PlantDoc-Dataset (pinned commit)"}}

    # ---- 1. Full PlantVillage, from the zip's file list -------------------------
    print("Reading PlantVillage zip file list ...")
    zf = _open_plantvillage_zip(remote=not args.local)
    names = zf.namelist()
    color = [n for n in names if n.startswith("raw/color/") and not n.endswith("/")]
    full_per_class = Counter(n.split("/")[2] for n in color)
    stats["pv_full"] = {
        "total_images": len(color),
        "num_classes": len(full_per_class),
        "num_crops": len({crop_of(c) for c in full_per_class}),
        "crops": sorted({crop_of(c) for c in full_per_class}),
        "num_healthy_classes": sum(c.endswith("healthy") for c in full_per_class),
        "per_class": {c: full_per_class[c] for c in PV_CLASSES},
        "largest_class": max(full_per_class.items(), key=lambda kv: kv[1]),
        "smallest_class": min(full_per_class.items(), key=lambda kv: kv[1]),
        "versions_in_zip": sorted({n.split("/")[1] for n in names if n.count("/") >= 2}),
    }

    # ---- 2. Our split (exactly what training uses) ------------------------------
    print("Planning our split (same code as training) ...")
    chosen, seg_of = plan_plantvillage_split(names, args.train_per_class, args.val_per_class,
                                             args.test_per_class, args.seed)
    split_counts = Counter(sp for _, sp, _ in chosen)
    per_class_split = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for _, sp, cls in chosen:
        per_class_split[cls][sp] += 1
    totals = {c: sum(v.values()) for c, v in per_class_split.items()}

    # Leakage check: which physical leaves appear in more than one split?
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(HF_REPO, "leaf_grouping/leaf-map.json", repo_type="dataset")) as f:
        leaf_map = json.load(f)
    groups = defaultdict(set)
    for path, sp, _ in chosen:
        groups[sp].add(_leaf_group(path, leaf_map))
    stats["our_split"] = {
        "config": {"train_per_class": args.train_per_class, "val_per_class": args.val_per_class,
                   "test_per_class": args.test_per_class, "seed": args.seed},
        "train": split_counts["train"], "val": split_counts["val"], "test": split_counts["test"],
        "total": len(chosen),
        "per_class": {c: per_class_split[c] for c in PV_CLASSES},
        "largest_class": max(totals.items(), key=lambda kv: kv[1]),
        "smallest_class": min(totals.items(), key=lambda kv: kv[1]),
        "classes_with_fewer_test": {c: v["test"] for c, v in per_class_split.items()
                                    if v["test"] < args.test_per_class},
        "color_photos_with_segmented_twin": len(seg_of),
        "leaf_groups": {sp: len(g) for sp, g in groups.items()},
        "shared_leaf_groups": {"train_val": len(groups["train"] & groups["val"]),
                               "train_test": len(groups["train"] & groups["test"]),
                               "val_test": len(groups["val"] & groups["test"])},
    }

    # ---- 3. Images actually prepared on this machine -----------------------------
    pv_dir = os.path.join(args.data_dir, "plantvillage")
    local = {sp: folder_stats(os.path.join(pv_dir, sp)) for sp in ("train", "val", "test")}
    info_path = os.path.join(pv_dir, "prep_info.json")
    stats["pv_local"] = {"prep_info": json.load(open(info_path)) if os.path.exists(info_path) else None,
                         "splits": local}
    areas = []
    mask_root = os.path.join(pv_dir, "masks")
    for dirpath, _, files in os.walk(mask_root):
        for f in files:
            areas.append((np.asarray(Image.open(os.path.join(dirpath, f))) > 127).mean())
    stats["pv_local"]["leaf_area"] = ({"mean": float(np.mean(areas)), "min": float(np.min(areas)),
                                       "max": float(np.max(areas)), "num_masks": len(areas)}
                                      if areas else None)
    if local["train"] and local["train"]["total"]:
        ds = LeafDataset(os.path.join(pv_dir, "train"), get_transforms("eval"))
        x, _ = ds[0]
        stats["tensor"] = {"single_image": list(x.shape), "dtype": str(x.dtype),
                           "min": float(x.min()), "max": float(x.max()),
                           "batch_example": [64] + list(x.shape)}

    # ---- 4. PlantDoc ----------------------------------------------------------------
    print("Listing PlantDoc files (git) ...")
    files = [p for p in list_plantdoc_files(args.data_dir) if p.count("/") == 2]
    pd_counts = defaultdict(lambda: {"train": 0, "test": 0})
    for p in files:
        split, folder, _ = p.split("/")
        if split in ("train", "test"):
            pd_counts[folder][split] += 1
    stats["plantdoc"] = {
        "folders": len(pd_counts),
        "train_images": sum(v["train"] for v in pd_counts.values()),
        "test_images": sum(v["test"] for v in pd_counts.values()),
        "per_folder": {k: dict(v) for k, v in sorted(pd_counts.items())},
        "mapping": PLANTDOC_TO_PLANTVILLAGE,
        "unmapped_folders": sorted(k for k in pd_counts if k not in PLANTDOC_TO_PLANTVILLAGE),
        "num_overlap_classes": len(OVERLAP_CLASSES),
        "pv_classes_not_in_plantdoc": sorted(set(PV_CLASSES) - set(OVERLAP_CLASSES)),
        "local": folder_stats(os.path.join(args.data_dir, "plantdoc")),
    }
    stats["plantdoc"]["total_images"] = stats["plantdoc"]["train_images"] + stats["plantdoc"]["test_images"]

    save_json(stats, args.out)
    s = stats["our_split"]
    print(f"PlantVillage: {stats['pv_full']['total_images']} images, {stats['pv_full']['num_classes']} classes, "
          f"{stats['pv_full']['num_crops']} crops | our split {s['train']}/{s['val']}/{s['test']} | "
          f"shared leaves {s['shared_leaf_groups']} | PlantDoc {stats['plantdoc']['total_images']} images")
    print("Saved", args.out)


if __name__ == "__main__":
    main()

"""Everything about data: downloading, splitting, transforms and DataLoaders.

Datasets
--------
PlantVillage (lab photos, one leaf on a plain background, 38 classes)
    Source: Hugging Face "mohanty/PlantVillage" (official upload by the dataset
    author, public, no login). It ships one big data.zip (~2.2 GB, contains the
    color / grayscale / segmented versions) plus the *official* train/test split
    files. The official split keeps all photos of the same physical leaf on the
    same side, so there is no leakage between train and test.

PlantDoc (real field photos: cluttered backgrounds, several leaves, shadows)
    Source: GitHub "pratikkayal/PlantDoc-Dataset" (public). We never train on
    it. We only use it as an "out-of-domain" test set, restricted to the 28
    classes that also exist in PlantVillage.

After preparation the folders look like this (one sub-folder per class):
    data/plantvillage/train/<class>/*.jpg
    data/plantvillage/val/<class>/*.jpg
    data/plantvillage/test/<class>/*.jpg
    data/plantvillage/masks/<split>/<class>/*.png   (leaf masks, white = leaf)
    data/plantdoc/<PlantVillage class name>/*.jpg

Leaf masks come from PlantVillage's own "segmented" version of each photo
(same leaf, background painted black). They are used for the background-bias
experiment and for the Grad-CAM leaf-focus score.
"""

import io
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import ensure_dir, load_json, save_json, seed_worker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO = "mohanty/PlantVillage"
PLANTDOC_REPO = "pratikkayal/PlantDoc-Dataset"
# Pin the exact commit so everyone downloads the same images.
PLANTDOC_COMMIT = "5467f6012d78d1c446145d5f582da6096f852ae8"

IMG_SIZE = 224
# Pretrained ImageNet models expect inputs normalised with these statistics.
# We use the same normalisation for the scratch CNN so both models see identical inputs.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The 38 PlantVillage classes (folder names in the official dataset), sorted.
PV_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# PlantDoc folder name -> matching PlantVillage class.
# All 28 PlantDoc classes have a PlantVillage counterpart; 10 PlantVillage
# classes (e.g. Orange HLB, Tomato Target Spot) do not appear in PlantDoc.
# ("Tomato two spotted spider mites leaf" exists only in PlantDoc's train
# folder, with just 2 images.)
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

# Indices (into PV_CLASSES) of the classes shared by both datasets.
OVERLAP_CLASSES = sorted(set(PLANTDOC_TO_PLANTVILLAGE.values()))
OVERLAP_IDX = [PV_CLASSES.index(c) for c in OVERLAP_CLASSES]


# ---------------------------------------------------------------------------
# PlantVillage download + leak-free subsampled split
# ---------------------------------------------------------------------------
def _leaf_group(path, leaf_map):
    """Return an ID for the physical leaf shown in this image.

    PlantVillage often has several photos of the same leaf. The dataset ships
    a leaf map (photo name -> leaf id). If a photo is missing from the map we
    treat it as its own group (the safe fallback).
    """
    cls = path.split("/")[2]
    key = os.path.splitext(os.path.basename(path).split("___")[-1])[0].lower()
    for entry in leaf_map.get(key, []):  # entry looks like "Tomato___healthy:::12.0"
        if entry.startswith(cls + ":::"):
            return entry
    return path


def _find_segmented_twins(zip_names):
    """Map each color photo to its segmented (leaf on black background) version.

    Usually  raw/color/<cls>/<name>.JPG -> raw/segmented/<cls>/<name>_final_masked.jpg.
    One class (Corn Common rust) names its color photos differently, so as a
    fallback we match on the photo ID after the last '___' (only if unique).
    """
    names = set(zip_names)
    by_photo_id = defaultdict(list)
    for n in zip_names:
        if n.startswith("raw/segmented/") and n.endswith("_final_masked.jpg"):
            cls = n.split("/")[2]
            photo_id = os.path.basename(n)[: -len("_final_masked.jpg")].split("___")[-1].lower()
            by_photo_id[(cls, photo_id)].append(n)

    twins = {}
    for n in zip_names:
        if not n.startswith("raw/color/") or n.endswith("/"):
            continue
        stem = os.path.splitext(n.replace("raw/color/", "raw/segmented/", 1))[0]
        if stem + "_final_masked.jpg" in names:
            twins[n] = stem + "_final_masked.jpg"
        else:
            cls = n.split("/")[2]
            photo_id = os.path.splitext(os.path.basename(n))[0].split("___")[-1].lower()
            candidates = by_photo_id.get((cls, photo_id), [])
            if len(candidates) == 1:
                twins[n] = candidates[0]
    return twins


def make_leaf_mask(segmented_image, threshold=25):
    """Leaf mask from a segmented image: leaf pixels = 255, background = 0.

    In the segmented images the background was painted black, so any pixel
    whose brightest channel is above `threshold` belongs to the leaf. The median
    filter removes JPEG noise specks at the leaf border.
    """
    gray_max = np.asarray(segmented_image.convert("RGB")).max(axis=2)
    mask = Image.fromarray(((gray_max > threshold) * 255).astype(np.uint8))
    return mask.filter(ImageFilter.MedianFilter(5))


def random_background(size, rng):
    """A random, blotchy, coloured background (no real-world content).

    A tiny 4x4 grid of random colours is blown up to full size (smooth colour
    blobs), then pixel noise is added. Every call gives a different background.
    """
    small = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    blobs = np.asarray(Image.fromarray(small).resize(size, Image.BICUBIC), dtype=np.float32)
    noise = rng.normal(0, 20, size=blobs.shape)
    return Image.fromarray(np.clip(blobs + noise, 0, 255).astype(np.uint8))


def _open_plantvillage_zip(remote):
    """Open data.zip either from a local download (fast) or remotely (laptop).

    remote=False: download the whole 2.2 GB zip once (~1-3 min on Colab).
    remote=True : read only the images we need straight from Hugging Face over
                  HTTP. Slow per image (~1-2 s) but avoids a 2.2 GB download,
                  so it is only meant for tiny smoke tests on a laptop.
    """
    from huggingface_hub import HfFileSystem, hf_hub_download

    if remote:
        f = HfFileSystem().open(f"datasets/{HF_REPO}/data.zip", "rb", block_size=1 << 20)
        return zipfile.ZipFile(f)
    local_path = hf_hub_download(HF_REPO, "data.zip", repo_type="dataset")
    return zipfile.ZipFile(local_path)


def plan_plantvillage_split(zip_names, train_per_class=200, val_per_class=40,
                            test_per_class=60, seed=42):
    """Decide which PlantVillage photos go to train / val / test (file names only).

    - test  = random sample from the OFFICIAL test split (leaf-grouped by the authors)
    - train/val = the OFFICIAL train split, divided by leaf group, so no leaf is in both
    zip_names = list of files inside data.zip (needed to find the segmented twins).
    Returns (chosen, seg_of): chosen = list of (zip path, split, class),
    seg_of = {color photo path: segmented photo path}.
    """
    from huggingface_hub import hf_hub_download

    # 1. Small metadata files: official split lists and the leaf map.
    def read_list(name):
        path = hf_hub_download(HF_REPO, f"splits/{name}", repo_type="dataset")
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    official_train = read_list("color_train.txt")
    official_test = read_list("color_test.txt")
    with open(hf_hub_download(HF_REPO, "leaf_grouping/leaf-map.json", repo_type="dataset")) as f:
        leaf_map = json.load(f)

    # 2. Find the "segmented" (leaf-only) twin of every color photo.
    #    We only keep photos that have one, because we need a leaf mask for every image.
    seg_of = _find_segmented_twins(zip_names)
    official_train = [p for p in official_train if p in seg_of]
    official_test = [p for p in official_test if p in seg_of]

    # 3. Decide which files go to which split.
    rng = random.Random(seed)
    by_class_train = defaultdict(list)
    by_class_test = defaultdict(list)
    for p in official_train:
        by_class_train[p.split("/")[2]].append(p)
    for p in official_test:
        by_class_test[p.split("/")[2]].append(p)

    chosen = []  # list of (zip path, split, class)
    for cls in PV_CLASSES:
        # Group training photos by leaf, shuffle the groups, give whole groups to val.
        groups = defaultdict(list)
        for p in sorted(by_class_train[cls]):
            groups[_leaf_group(p, leaf_map)].append(p)
        group_ids = sorted(groups)
        rng.shuffle(group_ids)

        val_files, train_files = [], []
        for g in group_ids:
            if len(val_files) < val_per_class:
                val_files.extend(groups[g])
            else:
                train_files.extend(groups[g])
        val_files = val_files[:val_per_class]
        rng.shuffle(train_files)
        train_files = train_files[:train_per_class]

        test_files = sorted(by_class_test[cls])
        rng.shuffle(test_files)
        test_files = test_files[:test_per_class]

        chosen += [(p, "train", cls) for p in train_files]
        chosen += [(p, "val", cls) for p in val_files]
        chosen += [(p, "test", cls) for p in test_files]
    return chosen, seg_of


def prepare_plantvillage(data_dir="data", train_per_class=200, val_per_class=40,
                         test_per_class=60, seed=42, remote=False):
    """Download PlantVillage and write a small, balanced, leak-free subset to disk
    (split chosen by plan_plantvillage_split), with a leaf mask for every image.
    Returns the folder that contains train/ val/ test/ masks/.
    """
    out_dir = os.path.join(data_dir, "plantvillage")
    config = dict(train_per_class=train_per_class, val_per_class=val_per_class,
                  test_per_class=test_per_class, seed=seed, masks=True)
    info_path = os.path.join(out_dir, "prep_info.json")
    if os.path.exists(info_path) and load_json(info_path)["config"] == config:
        print(f"PlantVillage already prepared in {out_dir}, skipping.")
        return out_dir
    if os.path.exists(out_dir):  # prepared with other settings: start clean
        shutil.rmtree(out_dir)

    print(f"Opening PlantVillage zip (remote={remote}) ...")
    zf = _open_plantvillage_zip(remote)
    chosen, seg_of = plan_plantvillage_split(zf.namelist(), train_per_class, val_per_class,
                                             test_per_class, seed)

    # Extract just those images from the zip, plus a leaf mask for each one.
    print(f"Extracting {len(chosen)} PlantVillage images + leaf masks ...")
    for i, (path, split, cls) in enumerate(chosen):
        name = os.path.splitext(os.path.basename(path))[0]
        target = os.path.join(out_dir, split, cls, name + ".jpg")
        ensure_dir(os.path.dirname(target))
        with open(target, "wb") as f:
            f.write(zf.read(path))
        segmented = Image.open(io.BytesIO(zf.read(seg_of[path])))
        mask_path = os.path.join(out_dir, "masks", split, cls, name + ".png")
        ensure_dir(os.path.dirname(mask_path))
        make_leaf_mask(segmented).save(mask_path)
        if (i + 1) % 500 == 0 or (remote and (i + 1) % 20 == 0):
            print(f"  {i + 1}/{len(chosen)}")

    counts = {s: sum(1 for _, sp, _ in chosen if sp == s) for s in ("train", "val", "test")}
    save_json({"config": config, "counts": counts}, info_path)
    print("PlantVillage ready:", counts)
    return out_dir


# ---------------------------------------------------------------------------
# PlantDoc download (field images, test only)
# ---------------------------------------------------------------------------
def _safe_name(name):
    """PlantDoc file names contain characters like '?' that Windows cannot store."""
    stem = os.path.splitext(name)[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem)[:80] + ".jpg"


def _download_plantdoc_image(item):
    """Download one image, shrink it (some are 4000 px wide) and save as JPEG."""
    repo_path, target = item
    if os.path.exists(target):
        return True
    url = (f"https://raw.githubusercontent.com/{PLANTDOC_REPO}/{PLANTDOC_COMMIT}/"
           + urllib.parse.quote(repo_path))
    for attempt in range(3):  # retry: network hiccups / rate limiting
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
            break
        except Exception as e:  # a few PlantDoc files are broken or not really images
            if attempt == 2:
                print(f"  skipped {repo_path}: {e}")
                return False
            time.sleep(2 * (attempt + 1))
    img.thumbnail((512, 512))
    ensure_dir(os.path.dirname(target))
    img.save(target, quality=95)
    return True


def list_plantdoc_files(data_dir="data"):
    """Paths of every file in the PlantDoc repo at the pinned commit, e.g.
    'test/Apple Scab Leaf/xyz.jpg'.

    A "blobless" clone without checkout downloads only the file list (a few
    seconds), not the 1 GB of images. We don't simply check the files out
    because some names contain '?', which Windows forbids.
    """
    listing_dir = os.path.join(data_dir, "plantdoc_git")
    if not os.path.isdir(listing_dir):
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
                        f"https://github.com/{PLANTDOC_REPO}.git", listing_dir], check=True)
    return subprocess.run(["git", "-C", listing_dir, "ls-tree", "-r", "--name-only", "-z",
                           PLANTDOC_COMMIT], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.split("\0")


def prepare_plantdoc(data_dir="data", splits=("train", "test"), max_per_class=None,
                     num_threads=16):
    """Download PlantDoc and store it under PlantVillage class names.

    We use BOTH PlantDoc's train and test folders as our field test set
    (~2,500 images), because we never train on PlantDoc and more images give
    a more reliable accuracy estimate. max_per_class limits it for smoke tests.
    """
    out_dir = os.path.join(data_dir, "plantdoc")
    info_path = os.path.join(out_dir, "prep_info.json")
    config = dict(splits=list(splits), max_per_class=max_per_class)
    if os.path.exists(info_path) and load_json(info_path)["config"] == config:
        print(f"PlantDoc already prepared in {out_dir}, skipping.")
        return out_dir

    per_class = defaultdict(list)
    for path in list_plantdoc_files(data_dir):
        parts = path.split("/")
        if len(parts) == 3 and parts[0] in splits:
            pv_class = PLANTDOC_TO_PLANTVILLAGE.get(parts[1])
            if pv_class is not None:
                # prefix with split name so train/test files with the same name don't clash
                target = os.path.join(out_dir, pv_class, parts[0] + "_" + _safe_name(parts[2]))
                per_class[pv_class].append((path, target))

    items = []
    for pv_class in sorted(per_class):
        files = sorted(per_class[pv_class])
        items += files[:max_per_class] if max_per_class else files

    print(f"Downloading {len(items)} PlantDoc images ...")
    with ThreadPoolExecutor(num_threads) as pool:  # parallel downloads, otherwise it is slow
        ok = list(pool.map(_download_plantdoc_image, items))
    save_json({"config": config, "downloaded": sum(ok), "failed": len(ok) - sum(ok)}, info_path)
    print(f"PlantDoc ready: {sum(ok)} images ({len(ok) - sum(ok)} skipped)")
    return out_dir


# ---------------------------------------------------------------------------
# Dataset, transforms, DataLoaders
# ---------------------------------------------------------------------------
BACKGROUNDS = ["original", "leaf_only", "random_bg"]


class LeafDataset(Dataset):
    """Reads images from root/<class name>/*.jpg.

    Unlike torchvision's ImageFolder, labels always come from the fixed
    PV_CLASSES list, so PlantDoc (only 28 folders) gets the same label numbers
    as PlantVillage (38 folders).

    background (PlantVillage only, needs mask_root = folder with the leaf masks):
        "original"  : the photo as it is (grey/plain lab background)
        "leaf_only" : background pixels set to black, only the leaf is left
        "random_bg" : the leaf pasted on a random blotchy coloured background
    fixed_bg_seed: if set, image i always gets the same random background
        (used for test sets, so evaluation is repeatable). If None, a new
        background is drawn every time (used for training = augmentation).
    """

    def __init__(self, root, transform=None, class_names=PV_CLASSES, background="original",
                 mask_root=None, fixed_bg_seed=None):
        assert background in BACKGROUNDS, background
        assert background == "original" or mask_root, "leaf_only/random_bg need mask_root"
        self.transform = transform
        self.class_names = class_names
        self.background = background
        self.mask_root = mask_root
        self.fixed_bg_seed = fixed_bg_seed
        self.samples = []  # list of (image path, label index)
        for label, cls in enumerate(class_names):
            folder = os.path.join(root, cls)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, name), label))

    def __len__(self):
        return len(self.samples)

    def load_mask(self, idx):
        """Leaf mask (PIL, mode 'L', 255 = leaf) of sample idx."""
        path, label = self.samples[idx]
        name = os.path.splitext(os.path.basename(path))[0] + ".png"
        return Image.open(os.path.join(self.mask_root, self.class_names[label], name))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.background != "original":
            mask = self.load_mask(idx)
            if self.background == "leaf_only":
                new_bg = Image.new("RGB", image.size)  # all black
            else:
                # np.random is seeded by set_seed / seed_worker, so training stays reproducible
                seed = (self.fixed_bg_seed + idx if self.fixed_bg_seed is not None
                        else np.random.randint(2**31))
                rng = np.random.default_rng(seed)
                new_bg = random_background(image.size, rng)
            image = Image.composite(image, new_bg, mask)  # leaf where mask=255, new_bg elsewhere
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(kind):
    """kind = "eval" | "standard" | "strong".

    eval     : just resize + normalise (no randomness) for val/test/app.
    standard : light augmentation (flips) - the usual PlantVillage setup.
    strong   : simulates field conditions: zoom/crop, rotation, lighting/colour
               changes and blur, to make the model robust to domain shift.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if kind == "eval":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])
    if kind == "standard":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    if kind == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 3.0))], p=0.3),
            transforms.ToTensor(),
            normalize,
        ])
    raise ValueError(f"unknown transform kind: {kind}")


def pv_dataset(pv_dir, split, kind="eval", background="original", fixed_bg_seed=None):
    """A PlantVillage split (train/val/test) with the given transform and background."""
    return LeafDataset(os.path.join(pv_dir, split), get_transforms(kind), background=background,
                       mask_root=os.path.join(pv_dir, "masks", split),
                       fixed_bg_seed=fixed_bg_seed)


def make_loader(dataset, batch_size=64, shuffle=False, num_workers=2, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)  # reproducible shuffling
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator)


def get_plantvillage_loaders(pv_dir, aug="standard", batch_size=64, num_workers=2, seed=42,
                             background="original"):
    """Train loader uses augmentation `aug`; val/test never use augmentation.

    `background` changes train AND val images (val must look like train, because
    it picks the best checkpoint). The test loader always uses original photos.
    """
    train = pv_dataset(pv_dir, "train", aug, background)
    val = pv_dataset(pv_dir, "val", "eval", background, fixed_bg_seed=seed)
    test = pv_dataset(pv_dir, "test", "eval")
    return (make_loader(train, batch_size, True, num_workers, seed),
            make_loader(val, batch_size, False, num_workers, seed),
            make_loader(test, batch_size, False, num_workers, seed))

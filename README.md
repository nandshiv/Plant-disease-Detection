# Plant Disease Detection with Transfer Learning, Grad-CAM and Domain-Shift Analysis

Deep Learning Lab project (PyTorch + torchvision).

We classify leaf images into **38 plant/disease classes** from PlantVillage, and we:

1. compare a **small CNN trained from scratch** with a **pretrained EfficientNet-B0 fine-tuned with transfer learning**,
2. evaluate with accuracy, per-class precision/recall/F1 and confusion matrices,
3. explain predictions with **Grad-CAM** heatmaps (correct and wrong predictions) and measure how much of the
   heatmap lies on the leaf (**leaf-focus score**, using PlantVillage's leaf masks),
4. test for **background bias**: retrain the baseline CNN on leaf-only and random-background images and test every
   model on all three backgrounds,
5. measure the **domain shift** from lab photos (PlantVillage) to real field photos (PlantDoc), and try to
   reduce it with **stronger data augmentation** (before/after),
6. add a **confidence threshold** so unsure predictions are rejected ("not sure"), chosen on the validation set,
7. provide a **Gradio app**: upload a leaf photo → disease, confidence, Grad-CAM overlay (or "not sure").

## Datasets (free, no login)

| Dataset | Role | Source | Notes |
|---|---|---|---|
| PlantVillage (color) | train / val / test | Hugging Face [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage) (official upload by the authors) | 54k lab images, 38 classes, plain background. One 2.2 GB zip + official split files. |
| PlantDoc | field test only | GitHub [`pratikkayal/PlantDoc-Dataset`](https://github.com/pratikkayal/PlantDoc-Dataset), pinned to commit `5467f60` | 2,578 real-world photos, 28 classes, all of which map onto a PlantVillage class (`PLANTDOC_TO_PLANTVILLAGE` in `src/data.py`). |

Both are downloaded automatically by `src/data.py`. Nothing needs a token or account (Hugging Face may print
an "unauthenticated requests" warning; it is harmless). `git` must be installed: it is used to list PlantDoc's files.

**Splits without leakage.** PlantVillage often has several photos of the *same physical leaf*. If one photo is in
train and another in test, test accuracy is inflated. So:

- **test** = a random sample of the authors' **official test split**, which is already grouped by leaf;
- **train / val** = the official train split, divided **by leaf ID** (from the dataset's leaf map), so all photos of a
  leaf go to the same side.
- Subset size (per class): up to 200 train, 40 val, up to 60 test → 7,430 / 1,520 / 2,244 images (three small
  classes have fewer train or test images; numbers from `scripts/dataset_stats.py`). This keeps all six training
  runs within roughly 40 minutes on a Colab T4.
- PlantDoc is never used for training. We use both of its folders (train + test, ~2.5k images) as one field test set.
- **Leaf masks.** The PlantVillage zip also has a "segmented" copy of each photo (same leaf, background painted
  black). For every image we keep, we save a mask (`make_leaf_mask` in `src/data.py`: pixel is leaf if its brightest
  channel > 25, then a 5×5 median filter). Every color photo has a segmented copy (one class, Corn Common rust, is matched by photo ID because its file
  names differ).

## Project structure

```
DL_project/
├── src/
│   ├── utils.py      # seeds, device, JSON, training-curve plots
│   ├── data.py       # download + leak-free split, leaf masks, LeafDataset (backgrounds), transforms, loaders
│   ├── models.py     # SimpleCNN, EfficientNet-B0 (transfer learning), checkpoint save/load
│   ├── train.py      # training loop (AdamW, cosine LR, mixed precision on GPU), run_experiment()
│   ├── evaluate.py   # metrics, confusion matrices, domain shift, background bias, rejection threshold
│   └── gradcam.py    # Grad-CAM implementation, leaf-focus score, example grids
├── notebooks/
│   └── train_plant_disease.ipynb   # thin Colab notebook that calls src/
├── app.py            # Gradio demo
├── scripts/
│   ├── dataset_stats.py      # real dataset numbers -> docs/viva_prep/dataset_stats.json
│   └── build_viva_pdfs.py    # regenerates the viva PDFs in docs/viva_prep/
├── docs/viva_prep/   # viva preparation PDFs
├── requirements.txt
└── results/          # created by the runs (checkpoints/, plots/, eval/, gradcam/, *.json)
```

## The experiments

| Name | Model | Train augmentation | Epochs | LR |
|---|---|---|---|---|
| `cnn_standard` | SimpleCNN (scratch, ~1.2M params) | standard: resize + flips | 12 | 1e-3 |
| `effnet_standard` | EfficientNet-B0, ImageNet-pretrained, all layers fine-tuned (~4.1M params) | standard | 6 | 5e-4 |
| `cnn_strong` | SimpleCNN | strong: RandomResizedCrop, flips, rotation ±30°, ColorJitter, GaussianBlur | 12 | 1e-3 |
| `effnet_strong` | EfficientNet-B0 | strong | 6 | 5e-4 |
| `cnn_leaf_only` | SimpleCNN, trained on leaf-only images (black background) | standard | 12 | 1e-3 |
| `cnn_random_bg` | SimpleCNN, trained with the leaf on a new random background each time | standard | 12 | 1e-3 |

All runs: seed 42, batch size 64, AdamW (weight decay 1e-4), cosine learning-rate schedule, cross-entropy loss,
best checkpoint selected on **validation** accuracy. Test sets are only used at the end.

## Results

> Placeholders. Fill in from `results/results_table.md` after the Colab run (see step 5 below).

**In-domain (PlantVillage test) and domain shift (PlantDoc), accuracy**

| Experiment | PV test acc (38) | PV test macro-F1 | PV acc (28 shared) | PlantDoc acc (28) | PlantDoc macro-F1 | Domain gap |
|---|---|---|---|---|---|---|
| cnn_standard | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| effnet_standard | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| cnn_strong | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| effnet_strong | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| cnn_leaf_only | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| cnn_random_bg | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

*Domain gap = PV acc (28 shared) − PlantDoc acc (28). Both are measured on the same 28 classes so the comparison is fair.*

**Background bias** (accuracy; from `results/background_table.md`)

| Trained on | test: original | test: leaf only | test: random bg | PlantDoc |
|---|---|---|---|---|
| cnn_standard | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| cnn_leaf_only | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| cnn_random_bg | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| effnet_standard | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| effnet_strong | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Leaf-focus score** (from `results/leaf_focus_table.md`): mean share of Grad-CAM heat on the leaf, PlantVillage
test, 300 images. cnn_standard _TBD_, effnet_standard _TBD_, effnet_strong _TBD_ (mean leaf area _TBD_).

**Rejection threshold** (from `results/rejection/<model>.json`), effnet_strong: threshold _TBD_;
PV test coverage _TBD_ / accuracy on answered _TBD_; PlantDoc coverage _TBD_ / accuracy on answered _TBD_.

**Training time on Colab T4:** cnn_standard _TBD_ min, effnet_standard _TBD_ min, cnn_strong _TBD_ min, effnet_strong _TBD_ min
(from `train_time_sec` in `results/history_*.json`).

**Figures** (in `results/` after the run):

- Training curves: `plots/history_<experiment>.png`
- Confusion matrices: `eval/<experiment>_pv_test_confusion.png`, `eval/<experiment>_plantdoc_confusion.png`
- Domain shift bar chart: `plots/domain_shift.png`
- Grad-CAM: `gradcam/<experiment>_pv_test.png`, `gradcam/<experiment>_plantdoc.png`
- Background bias heatmap: `plots/background_bias.png`
- Coverage vs accuracy: `rejection/<experiment>_coverage_accuracy.png`

**Observations:** _TBD after the run (for example: how much better transfer learning is, how large the domain gap
is, whether strong augmentation reduced it, what the Grad-CAM maps show on field images)._

## How to run

### On Google Colab (training)

1. Push this project to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Plant disease detection project"
   git branch -M main
   git remote add origin https://github.com/nandshiv/Plant-disease-Detection.git
   git push -u origin main
   ```
2. Open `https://colab.research.google.com/github/nandshiv/Plant-disease-Detection/blob/main/notebooks/train_plant_disease.ipynb`
3. `Runtime → Change runtime type → T4 GPU → Save`.
4. The first code cell already has `GITHUB_REPO_URL` set to this repository (change it if you fork the repo).
5. `Runtime → Run all`. Allow Google Drive access when asked. Checkpoints go to `MyDrive/plant_disease/checkpoints`,
   results are copied to `MyDrive/plant_disease/results` and downloaded as `results_export.zip`.

### Locally (laptop, CPU)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Tiny smoke test (downloads only ~150 PlantVillage images remotely, takes ~5–10 min, mostly download):

```bash
python -c "from src.data import *; prepare_plantvillage('data', 2, 1, 1, remote=True); prepare_plantdoc('data', ('test',), 2)"
python -m src.train --model cnn --aug standard --epochs 1
```

Gradio app (needs a checkpoint in `results/checkpoints/`, e.g. `effnet_strong.pt` copied from Drive):

```bash
python app.py
```

Then open http://127.0.0.1:7860.

## Viva cheat sheet

**Transfer learning.** Instead of starting from random weights, we start from a network already trained on
ImageNet (1.3M images, 1000 classes). Its early layers already detect edges, colours and textures, which are useful
for leaves too. We replace only the last layer (1000 → 38 outputs) and fine-tune all layers with a small learning rate.
Result: much higher accuracy from little data and few epochs, compared with the scratch CNN. (*Feature extraction*
= freeze the backbone, train only the new head; *fine-tuning* = also update the backbone. We fine-tune;
`build_efficientnet(freeze_backbone=True)` gives the other option.)

**Data augmentation.** Random, label-preserving changes to training images (flip, crop, rotate, change brightness,
blur), applied freshly every epoch. The model effectively sees more varied data, overfits less and becomes invariant
to those changes. We use it only on training images; val/test images are only resized and normalised.
"Strong" augmentation deliberately imitates field conditions (other zoom levels, angles, lighting, focus) to fight
domain shift.

**Grad-CAM.** Shows which image regions drove a prediction. Take the feature maps of the last conv layer, compute
the gradient of the predicted class score with respect to them, average each gradient map into one weight per
feature map, take the weighted sum of the maps, apply ReLU (keep only positive evidence), and upsample to image size.
Red = important. If the model is right "for the right reason", red is on the lesions; red on the background means it
learned a shortcut.

**Domain shift.** Training and test data come from different distributions. PlantVillage photos are single leaves on
a uniform background with studio lighting; PlantDoc photos are taken in fields with clutter, several leaves, shadows
and blur. A model that scores ~99% on PlantVillage can drop a lot on PlantDoc. We measure the gap on the 28 shared
classes and try to reduce it with stronger augmentation. (Other remedies, not done here: training on field images,
background removal, domain adaptation methods.)

**Background bias and leaf-focus score.** If all training photos share one kind of background, a model can use the
background as a clue. We test this by changing only the background of the test leaves (leaf-only, random) and by
training variants on leaf-only and random backgrounds. The leaf-focus score is the share of the Grad-CAM heatmap
inside the leaf mask; compare it with the leaf's share of the image (an evenly spread heatmap would score exactly that).

**Rejection threshold.** The softmax confidence of the predicted class is compared with a threshold chosen on the
validation set (the 5th percentile of confidence among correct validation predictions). Below it, the app says
"not sure". Coverage = share of images answered; the trade-off is fewer answers but higher accuracy on them.

**Reading the confusion matrix.** Rows = true class, columns = predicted class. Our matrices are row-normalised, so
each row sums to 1 and the **diagonal value is the recall** of that class. Bright off-diagonal cells show which
classes get confused (e.g. two tomato diseases with similar spots). The extra "other" column counts predictions of a
class outside the shown set (on PlantDoc: one of the 10 PlantVillage classes that PlantDoc does not have).
Precision of a class = diagonal cell divided by its column total (in the unnormalised matrix).

**Other likely questions.**
- *Why a validation set?* To choose the best epoch/checkpoint without looking at the test set.
- *Why macro-F1?* It averages over classes, so small classes count as much as big ones.
- *Why ImageNet normalisation?* The pretrained weights expect inputs with that mean/std.
- *Why mixed precision?* float16 on the T4's tensor cores is ~2× faster; the GradScaler stops small gradients from underflowing to zero.
- *Why seeds?* Same split, initial weights and shuffling every time → reproducible numbers.

## Credits

- Mohanty, Hughes, Salathé (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science.
- Singh et al. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection.* CoDS-COMAD.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.

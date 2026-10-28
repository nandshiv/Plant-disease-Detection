"""Evaluation: accuracy, per-class precision/recall/F1, confusion matrices,
the domain-shift comparison (PlantVillage test vs. PlantDoc field photos),
the background-bias test, and the confidence threshold for rejecting unsure predictions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .data import BACKGROUNDS, OVERLAP_IDX, PV_CLASSES, LeafDataset, get_transforms, pv_dataset
from .utils import ensure_dir, save_json


@torch.no_grad()
def predict(model, loader, device):
    """Run the model on a whole loader. Returns (true labels, predicted labels, probabilities)."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        all_true.append(labels)
        all_pred.append(probs.argmax(1))
        all_prob.append(probs)
    return (torch.cat(all_true).numpy(), torch.cat(all_pred).numpy(),
            torch.cat(all_prob).numpy())


def short_name(cls):
    """'Tomato___Early_blight' -> 'Tomato: Early blight' (for readable plots)."""
    crop, disease = cls.split("___")
    crop = crop.split("_(")[0].replace(",_bell", " (bell)")
    return f"{crop}: {disease.replace('_', ' ').strip()}"


def compute_metrics(y_true, y_pred, labels):
    """Accuracy + per-class precision/recall/F1 for the given label indices.

    Precision(c) = of all images PREDICTED as c, how many really are c.
    Recall(c)    = of all images that REALLY are c, how many we found.
    F1(c)        = harmonic mean of the two.
    Macro average = plain mean over classes (every class counts equally).
    """
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=[PV_CLASSES[i] for i in labels],
        output_dict=True, zero_division=0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "num_images": int(len(y_true)),
        "per_class": {PV_CLASSES[i]: report[PV_CLASSES[i]] for i in labels},
    }


def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    """Row-normalised confusion matrix: row = true class, column = predicted class.

    Each row sums to 1, so the diagonal is the recall of that class. If the
    model predicts a class outside `labels` (possible on PlantDoc, which has
    only 28 of the 38 classes), it is counted in an extra "other" column.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    other = np.array([np.sum((y_true == t) & ~np.isin(y_pred, labels)) for t in labels])
    cm = np.concatenate([cm, other[:, None]], axis=1)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # normalise each row

    names = [short_name(PV_CLASSES[i]) for i in labels]
    size = 0.32 * len(labels) + 3
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names) + 1), names + ["other"], rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def evaluate_on_folder(model, folder, device, title, out_prefix, labels=None,
                       batch_size=64, num_workers=2):
    """Evaluate on root/<class>/*.jpg; save metrics JSON + confusion matrix PNG.

    labels = class indices to report on (default: all 38). For PlantDoc we pass
    OVERLAP_IDX (the shared classes). The model still chooses among all 38,
    so predicting a class PlantDoc doesn't have simply counts as a mistake.
    """
    labels = labels if labels is not None else list(range(len(PV_CLASSES)))
    dataset = LeafDataset(folder, transform=get_transforms("eval"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers)
    y_true, y_pred, _ = predict(model, loader, device)

    keep = np.isin(y_true, labels)  # only images whose true class is in `labels`
    y_true, y_pred = y_true[keep], y_pred[keep]
    metrics = compute_metrics(y_true, y_pred, labels)
    save_json(metrics, out_prefix + "_metrics.json")
    plot_confusion_matrix(y_true, y_pred, labels, title, out_prefix + "_confusion.png")
    print(f"{title}: accuracy {metrics['accuracy']:.4f}, macro F1 {metrics['macro_f1']:.4f} "
          f"({metrics['num_images']} images)")
    return metrics


def evaluate_experiment(model, name, pv_dir, plantdoc_dir, device, results_dir="results",
                        num_workers=2):
    """Full evaluation of one trained model:
    1. PlantVillage test, all 38 classes             (in-domain)
    2. PlantVillage test, only the shared classes (in-domain, fair comparison)
    3. PlantDoc, the shared classes               (out-of-domain / field photos)
    (shared = the len(OVERLAP_IDX) = 28 classes that exist in both datasets)
    """
    out = os.path.join(results_dir, "eval")
    pv_test = os.path.join(pv_dir, "test")
    m_all = evaluate_on_folder(model, pv_test, device, f"{name} - PlantVillage test",
                               os.path.join(out, f"{name}_pv_test"), num_workers=num_workers)
    m_overlap = evaluate_on_folder(model, pv_test, device, f"{name} - PlantVillage ({len(OVERLAP_IDX)} shared)",
                                   os.path.join(out, f"{name}_pv_overlap"), OVERLAP_IDX,
                                   num_workers=num_workers)
    m_doc = evaluate_on_folder(model, plantdoc_dir, device, f"{name} - PlantDoc (field)",
                               os.path.join(out, f"{name}_plantdoc"), OVERLAP_IDX,
                               num_workers=num_workers)
    summary = {
        "pv_test_acc": m_all["accuracy"],
        "pv_test_macro_f1": m_all["macro_f1"],
        "pv_overlap_acc": m_overlap["accuracy"],
        "plantdoc_acc": m_doc["accuracy"],
        "plantdoc_macro_f1": m_doc["macro_f1"],
        # Domain gap = how much accuracy we lose going from lab photos to field photos
        "domain_gap": m_overlap["accuracy"] - m_doc["accuracy"],
    }
    save_json(summary, os.path.join(out, f"{name}_summary.json"))
    return summary


def results_table(summaries):
    """Markdown table from {experiment name: summary dict}, ready to paste into the README."""
    n = len(OVERLAP_IDX)
    lines = [f"| Experiment | PV test acc (38) | PV test macro-F1 | PV acc ({n} shared) "
             f"| PlantDoc acc ({n}) | PlantDoc macro-F1 | Domain gap |",
             "|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['pv_test_acc']:.3f} | {s['pv_test_macro_f1']:.3f} "
                     f"| {s['pv_overlap_acc']:.3f} | {s['plantdoc_acc']:.3f} "
                     f"| {s['plantdoc_macro_f1']:.3f} | {s['domain_gap']:.3f} |")
    return "\n".join(lines)


def plot_domain_shift(summaries, out_path):
    """Grouped bar chart: in-domain vs. field accuracy for every experiment."""
    names = list(summaries)
    x = np.arange(len(names))
    pv = [summaries[n]["pv_overlap_acc"] for n in names]
    doc = [summaries[n]["plantdoc_acc"] for n in names]

    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 3, 4))
    ax.bar(x - 0.2, pv, width=0.4, label="PlantVillage test (lab)")
    ax.bar(x + 0.2, doc, width=0.4, label="PlantDoc (field)")
    for i in range(len(names)):
        ax.text(x[i] - 0.2, pv[i] + 0.01, f"{pv[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, doc[i] + 0.01, f"{doc[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(f"accuracy ({len(OVERLAP_IDX)} shared classes)")
    ax.set_title("Domain shift: lab images vs. field images")
    ax.legend(loc="upper right")
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def predict_dataset(model, dataset, device, batch_size=64, num_workers=2):
    """predict() for a Dataset instead of a DataLoader."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return predict(model, loader, device)


# ---------------------------------------------------------------------------
# Background-bias experiment
# ---------------------------------------------------------------------------
def evaluate_background_bias(model, name, pv_dir, plantdoc_dir, device, results_dir="results",
                             num_workers=2, seed=42):
    """Test one model on the PlantVillage test set with each background
    (original / leaf_only / random_bg), and on PlantDoc.

    If accuracy drops a lot when only the background changes, the model was
    using the background as a clue (a "shortcut"), not only the leaf.
    """
    result = {}
    for bg in BACKGROUNDS:
        # fixed_bg_seed: every model sees exactly the same random backgrounds
        ds = pv_dataset(pv_dir, "test", "eval", bg, fixed_bg_seed=seed)
        y_true, y_pred, _ = predict_dataset(model, ds, device, num_workers=num_workers)
        result[f"pv_test_{bg}"] = float((y_true == y_pred).mean())
    doc = LeafDataset(plantdoc_dir, transform=get_transforms("eval"))
    y_true, y_pred, _ = predict_dataset(model, doc, device, num_workers=num_workers)
    keep = np.isin(y_true, OVERLAP_IDX)
    result["plantdoc"] = float((y_true[keep] == y_pred[keep]).mean())
    save_json(result, os.path.join(results_dir, "background", f"{name}.json"))
    print(f"{name}: " + ", ".join(f"{k} {v:.3f}" for k, v in result.items()))
    return result


def background_table(results):
    """Markdown table: one row per trained model, one column per test condition."""
    cols = [f"pv_test_{bg}" for bg in BACKGROUNDS] + ["plantdoc"]
    lines = ["| Trained on | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)


def plot_background_bias(results, out_path):
    """Heatmap: rows = how the model was trained, columns = how it was tested."""
    cols = [f"pv_test_{bg}" for bg in BACKGROUNDS] + ["plantdoc"]
    names = list(results)
    values = np.array([[results[n][c] for c in cols] for n in names])
    fig, ax = plt.subplots(figsize=(8, 1.0 * len(names) + 1.8))
    im = ax.imshow(values, cmap="Greens", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(cols)):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(len(cols)), ["test: original", "test: leaf only", "test: random bg",
                                     "PlantDoc (field)"])
    ax.set_yticks(range(len(names)), [f"train: {n}" for n in names])
    ax.set_title("Background-bias experiment (accuracy)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confidence threshold: reject unsure predictions
# ---------------------------------------------------------------------------
def coverage_and_accuracy(confidence, correct, threshold):
    """coverage = share of images we answer (confidence >= threshold);
    accuracy = accuracy on the answered images only."""
    accepted = confidence >= threshold
    coverage = float(accepted.mean())
    accuracy = float(correct[accepted].mean()) if accepted.any() else float("nan")
    return coverage, accuracy


def rejection_analysis(model, name, pv_dir, plantdoc_dir, device, results_dir="results",
                       keep_correct=0.95, num_workers=2):
    """Choose a confidence threshold on the VALIDATION set, then report its
    effect on the PlantVillage test set and on PlantDoc.

    Rule: the threshold is the confidence below which only 5% of the CORRECT
    validation predictions fall (keep_correct=0.95). So on lab images we keep
    ~95% of the right answers, and anything less confident is answered with
    "not sure". The test sets are never used to choose the threshold.
    Confidence = the softmax probability of the predicted class.
    """
    y_true, y_pred, probs = predict_dataset(model, pv_dataset(pv_dir, "val"), device,
                                            num_workers=num_workers)
    conf_val, correct_val = probs.max(1), y_true == y_pred
    if not correct_val.any():  # only possible in a tiny smoke test
        correct_val = np.ones_like(correct_val)
    threshold = float(np.quantile(conf_val[correct_val], 1 - keep_correct))

    sets = {"pv_test": pv_dataset(pv_dir, "test"),
            "plantdoc": LeafDataset(plantdoc_dir, transform=get_transforms("eval"))}
    result = {"threshold": threshold, "keep_correct": keep_correct}
    curves = {}
    for set_name, ds in sets.items():
        y_true, y_pred, probs = predict_dataset(model, ds, device, num_workers=num_workers)
        # PV test: all 38 classes; PlantDoc: the shared classes
        keep = np.ones(len(y_true), bool) if set_name == "pv_test" else np.isin(y_true, OVERLAP_IDX)
        conf, correct = probs.max(1)[keep], (y_true == y_pred)[keep]
        coverage, acc_accepted = coverage_and_accuracy(conf, correct, threshold)
        result[set_name] = {
            "accuracy_all": float(correct.mean()),
            "coverage": coverage,
            "accuracy_accepted": acc_accepted,
            "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
            "mean_confidence_wrong": float(conf[~correct].mean()) if (~correct).any() else None,
        }
        # Curve over many thresholds: accuracy usually rises as we answer fewer images
        curves[set_name] = [coverage_and_accuracy(conf, correct, t) for t in np.linspace(0, 0.99, 100)]
        print(f"{name} {set_name}: threshold {threshold:.2f} -> answers {coverage:.1%} of images, "
              f"accuracy on those {acc_accepted:.3f} (vs {correct.mean():.3f} answering all)")

    save_json(result, os.path.join(results_dir, "rejection", f"{name}.json"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for set_name, curve in curves.items():
        cov, acc = zip(*curve)
        ax.plot(cov, acc, label=set_name)
        ax.scatter([result[set_name]["coverage"]], [result[set_name]["accuracy_accepted"]], zorder=3)
    ax.set_xlabel("coverage (share of images answered)")
    ax.set_ylabel("accuracy on answered images")
    ax.set_title(f"{name}: coverage vs accuracy (dots = threshold {threshold:.2f})", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "rejection", f"{name}_coverage_accuracy.png"), dpi=130)
    plt.close(fig)
    return result

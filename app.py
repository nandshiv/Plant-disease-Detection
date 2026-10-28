"""Gradio demo: upload a leaf photo -> predicted disease, confidence, Grad-CAM overlay.

If the confidence is below the rejection threshold, the app says "not sure"
instead of giving a diagnosis. The threshold is read from
results/rejection/<checkpoint name>.json (made by rejection_analysis in the
notebook); without that file it falls back to DEFAULT_THRESHOLD.

Run locally (after copying a checkpoint into results/checkpoints/):
    python app.py
    python app.py --ckpt results/checkpoints/effnet_standard.pt
Inside Colab (gives a public *.gradio.live link):
    python app.py --share
"""

import argparse
import glob
import os

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.data import get_transforms
from src.evaluate import short_name
from src.gradcam import GradCAM, overlay
from src.models import get_gradcam_layer, load_checkpoint
from src.utils import get_device, load_json

# Preferred checkpoint first; the app uses the first one that exists.
DEFAULT_CKPTS = ["results/checkpoints/effnet_strong.pt",
                 "results/checkpoints/effnet_standard.pt"]
DEFAULT_THRESHOLD = 0.5  # only used if no results/rejection/<name>.json exists


def find_threshold(ckpt_path, threshold):
    """Use the threshold given on the command line, else the one chosen on the validation set."""
    if threshold is not None:
        return threshold
    name = os.path.splitext(os.path.basename(ckpt_path))[0]
    path = os.path.join("results", "rejection", f"{name}.json")
    if os.path.exists(path):
        return load_json(path)["threshold"]
    print(f"No {path} found, using default threshold {DEFAULT_THRESHOLD}")
    return DEFAULT_THRESHOLD


def find_checkpoint(path):
    if path:
        return path
    for p in DEFAULT_CKPTS + sorted(glob.glob("results/checkpoints/*.pt")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No checkpoint found. Train on Colab and copy the .pt file to "
                            "results/checkpoints/, or pass --ckpt PATH.")


def build_app(ckpt_path, threshold):
    device = get_device()
    model, model_name, class_names = load_checkpoint(ckpt_path, device)
    cam_fn = GradCAM(model, get_gradcam_layer(model, model_name))
    transform = get_transforms("eval")
    print(f"Loaded {model_name} from {ckpt_path} on {device}, rejection threshold {threshold:.2f}")

    def predict(image):
        if image is None:
            return None, None, "Please upload an image."
        image = image.convert("RGB")
        tensor = transform(image).to(device)
        cam, pred, prob = cam_fn(tensor)  # heatmap for the predicted class

        # Top-5 class probabilities for the gr.Label widget
        with torch.no_grad():
            probs = model(tensor.unsqueeze(0)).softmax(1)[0].cpu().numpy()
        top5 = np.argsort(probs)[::-1][:5]
        confidences = {short_name(class_names[i]): float(probs[i]) for i in top5}

        # Overlay the heatmap on the resized input image
        resized = np.asarray(image.resize((tensor.shape[2], tensor.shape[1]))) / 255.0
        heatmap = Image.fromarray((overlay(resized, cam) * 255).astype(np.uint8))

        if prob >= threshold:
            text = f"Prediction: {short_name(class_names[pred])}\nConfidence: {prob:.1%}"
        else:  # rejection: too unsure to give a diagnosis
            text = (f"NOT SURE (confidence {prob:.1%} is below the threshold {threshold:.0%}).\n"
                    f"Best guess: {short_name(class_names[pred])}. Try a clearer photo of a "
                    "single leaf, or ask an expert.")
        return confidences, heatmap, text

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Leaf photo"),
        outputs=[gr.Label(num_top_classes=5, label="Top-5 predictions"),
                 gr.Image(label="Grad-CAM (red = most important for the prediction)"),
                 gr.Textbox(label="Result")],
        title="Plant Disease Detection",
        description=(f"Model: {model_name} trained on PlantVillage (38 classes). "
                     f"Upload a photo of a single leaf. Predictions below {threshold:.0%} "
                     "confidence are rejected."),
        flagging_mode="never",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="path to a .pt checkpoint")
    parser.add_argument("--share", action="store_true", help="create a public link (Colab)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--threshold", type=float, default=None,
                        help="rejection threshold (default: from results/rejection/)")
    args = parser.parse_args()
    ckpt = find_checkpoint(args.ckpt)
    build_app(ckpt, find_threshold(ckpt, args.threshold)).launch(share=args.share,
                                                                server_port=args.port)

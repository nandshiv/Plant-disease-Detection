"""Build the viva-preparation PDFs in docs/viva_prep/.

    python scripts/build_viva_pdfs.py

Everything that describes the code is READ FROM THE CODE when this script runs:
layer shapes come from a real forward pass, hyper-parameters from the notebook's
settings cell, transform settings from get_transforms(), defaults from function
signatures, and code snippets via inspect. So after the code changes, just run
this script again. Results are taken from results/ only if they come from a real
(full) run; otherwise the PDFs show "[pending]" placeholders.

Inputs : src/, notebooks/train_plant_disease.ipynb, results/ (optional),
         results/dataset_stats.json (from scripts/dataset_stats.py, optional),
         data/ (optional, for real example images)
Outputs: docs/viva_prep/*.pdf   (figures are drawn into a temporary folder)
"""

import inspect
import json
import os
import re
import sys
import tempfile
import textwrap

import matplotlib

matplotlib.use("Agg")  # this script only writes files, never opens windows
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import gradcam as G  # noqa: E402
from src import models as M  # noqa: E402
from src import train as T  # noqa: E402

OUT_DIR = os.path.join("docs", "viva_prep")
RESULTS = "results"
FIG_DIR = tempfile.mkdtemp(prefix="viva_figs_")
PENDING = "[pending]"

# ===========================================================================
# 1. PDF building blocks (reportlab)
# ===========================================================================
_mpl_fonts = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for name, file in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                   ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"),
                   ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
                   ("DejaVuMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(_mpl_fonts, file)))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")

PAGE_W, PAGE_H = A4
MARGIN = 1.9 * cm
FRAME_W = PAGE_W - 2 * MARGIN
BLUE, ORANGE, GREY = "#1f5f9e", "#c0641a", "#555555"

S = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=22, leading=27,
                            spaceAfter=6, textColor=colors.HexColor("#1b3a5c")),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu", fontSize=11.5, leading=15,
                               textColor=colors.HexColor("#444444"), spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16, leading=20, spaceBefore=10,
                         spaceAfter=7, textColor=colors.HexColor("#1b3a5c")),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12.5, leading=16, spaceBefore=9,
                         spaceAfter=4, textColor=colors.HexColor("#1f5f9e")),
    "h3": ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=10.5, leading=14, spaceBefore=6,
                         spaceAfter=2),
    "p": ParagraphStyle("p", fontName="DejaVu", fontSize=10, leading=14.2, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=10, leading=14,
                             leftIndent=14, bulletIndent=3, spaceAfter=2),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.6, leading=11),
    "cellh": ParagraphStyle("cellh", fontName="DejaVu-Bold", fontSize=8.6, leading=11,
                            textColor=colors.white),
    "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=7.9, leading=9.8),
    "caption": ParagraphStyle("caption", fontName="DejaVu-Oblique", fontSize=8.6, leading=11,
                              alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                              spaceAfter=8),
    "q": ParagraphStyle("q", fontName="DejaVu-Bold", fontSize=10, leading=13.5, spaceBefore=5,
                        spaceAfter=1),
    "a": ParagraphStyle("a", fontName="DejaVu", fontSize=9.6, leading=13.2, leftIndent=12,
                        spaceAfter=4),
    "box": ParagraphStyle("box", fontName="DejaVu", fontSize=9.6, leading=13.2),
}


def fmt(text):
    """Tiny markup: escape HTML, then `code` -> monospace, **bold** -> bold."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="DejaVuMono" color="#8b1a1a">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


class Doc:
    """A list of flowables plus helpers, rendered with build()."""

    def __init__(self, title, subtitle):
        self.title = title
        self.story = [Paragraph(fmt(title), S["title"]), Paragraph(fmt(subtitle), S["subtitle"])]

    def h1(self, t):
        self.story.append(Paragraph(fmt(t), S["h1"]))

    def h2(self, t):
        self.story.append(Paragraph(fmt(t), S["h2"]))

    def h3(self, t):
        self.story.append(Paragraph(fmt(t), S["h3"]))

    def p(self, t):
        self.story.append(Paragraph(fmt(t), S["p"]))

    def bullets(self, items):
        for it in items:
            self.story.append(Paragraph(fmt(it), S["bullet"], bulletText="•"))
        self.story.append(Spacer(1, 4))

    def space(self, h=6):
        self.story.append(Spacer(1, h))

    def pagebreak(self):
        self.story.append(PageBreak())

    def code(self, text, max_lines=48):
        """Monospaced block with a border. Long lines are wrapped, long blocks split."""
        lines = []
        for line in textwrap.dedent(text).rstrip().split("\n"):
            wrapped = textwrap.wrap(line, 96, subsequent_indent="    ", drop_whitespace=False,
                                    replace_whitespace=False) or [""]
            lines += wrapped
        for i in range(0, len(lines), max_lines):
            chunk = "\n".join(lines[i:i + max_lines])
            t = Table([[Preformatted(chunk, S["code"])]], colWidths=[FRAME_W])
            t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#999999")),
                                   ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f4f4")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                   ("TOPPADDING", (0, 0), (-1, -1), 4),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
            self.story += [t, Spacer(1, 6)]

    def table(self, rows, widths=None, header=True, font_size=None):
        """rows = list of lists of strings. widths = fractions of the frame width."""
        n = len(rows[0])
        widths = widths or [1 / n] * n
        cs = S["cell"] if font_size is None else ParagraphStyle("c", parent=S["cell"],
                                                                 fontSize=font_size, leading=font_size + 2.4)
        hs = S["cellh"] if font_size is None else ParagraphStyle("h", parent=S["cellh"],
                                                                  fontSize=font_size, leading=font_size + 2.4)
        data = [[Paragraph(fmt(str(c)), hs if (header and r == 0) else cs) for c in row]
                for r, row in enumerate(rows)]
        t = Table(data, colWidths=[w * FRAME_W for w in widths], repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
                 ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                 ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f5f9e"))]
            style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef3f8"))
                      for r in range(2, len(rows), 2)]
        t.setStyle(TableStyle(style))
        self.story += [t, Spacer(1, 8)]

    def box(self, title, items, color="#fff6dd", border="#d9a400", numbered=False):
        """A highlighted box (e.g. 'numbers to memorize')."""
        paras = [Paragraph(f"<b>{fmt(title)}</b>", S["box"]), Spacer(1, 3)]
        for i, it in enumerate(items, 1):
            prefix = f"{i}. " if numbered else "• "
            paras.append(Paragraph(prefix + fmt(it), S["box"]))
        t = Table([[paras]], colWidths=[FRAME_W])
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(border)),
                               ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        self.story += [t, Spacer(1, 8)]

    def img(self, path, width_frac=1.0, caption=None, max_h_frac=0.62):
        """Image scaled to fit the page; skipped with a note if the file is missing."""
        if path and path.startswith("PENDING:"):
            self.p(f"[figure pending: `{path[8:]}` appears here after the Colab run]")
            return
        if not path or not os.path.exists(path):
            self.p(f"[figure not available: `{path}`]")
            return
        iw, ih = ImageReader(path).getSize()
        w = FRAME_W * width_frac
        h = w * ih / iw
        max_h = (PAGE_H - 2 * MARGIN) * max_h_frac
        if h > max_h:
            h, w = max_h, max_h * iw / ih
        parts = [Image(path, width=w, height=h)]
        if caption:
            parts.append(Paragraph(fmt(caption), S["caption"]))
        self.story.append(KeepTogether(parts))
        self.space(4)

    def qa(self, num, q, a, tag=None):
        tag_text = f"  [{tag}]" if tag else ""
        self.story.append(KeepTogether([Paragraph(f"Q{num}.{fmt(tag_text)} {fmt(fill(q))}", S["q"]),
                                        Paragraph(fmt(fill(a)), S["a"])]))

    def build(self, filename):
        path = os.path.join(OUT_DIR, filename)

        def decorate(canvas, doc):
            canvas.saveState()
            canvas.setFont("DejaVu", 7.8)
            canvas.setFillColor(colors.HexColor("#777777"))
            canvas.drawString(MARGIN, PAGE_H - 1.1 * cm, self.title)
            canvas.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"page {doc.page}")
            canvas.drawString(MARGIN, 1.0 * cm, "Plant disease detection - viva preparation")
            canvas.restoreState()

        doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                                topMargin=MARGIN, bottomMargin=MARGIN, title=self.title,
                                author="DL Lab group")
        doc.build(self.story, onFirstPage=decorate, onLaterPages=decorate)
        print("wrote", path)
        return path


# ===========================================================================
# 2. Facts read from the code
# ===========================================================================
def ref(obj):
    """'src/models.py: SimpleCNN (line 27)' for a function/class."""
    file = os.path.relpath(inspect.getsourcefile(obj), ROOT).replace("\\", "/")
    line = inspect.getsourcelines(obj)[1]
    return f"{file}: {obj.__qualname__} (line {line})"


def source(obj, start=None, end=None):
    """Source code of obj, optionally only from the line containing `start`
    up to (and including) the line containing `end`."""
    lines = inspect.getsource(obj).split("\n")
    i = next((k for k, l in enumerate(lines) if start in l), 0) if start else 0
    j = next((k for k, l in enumerate(lines) if k >= i and end in l), len(lines) - 1) if end else len(lines) - 1
    return textwrap.dedent("\n".join(lines[i:j + 1]))


def default(func, arg):
    return inspect.signature(func).parameters[arg].default


def notebook_settings():
    """Run the notebook's settings cell (plain Python) and return its variables."""
    nb = json.load(open("notebooks/train_plant_disease.ipynb", encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "code" and "SMOKE_TEST = False" in src:
            env = {}
            exec(src, env)
            return {k: v for k, v in env.items() if k.isupper()}
    raise RuntimeError("settings cell not found in the notebook")


def notebook_sections():
    nb = json.load(open("notebooks/train_plant_disease.ipynb", encoding="utf-8"))
    heads = []
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "markdown" and src.startswith("## "):
            heads.append(src.split("\n")[0][3:])
    return heads


def layer_table(model, input_shape=(1, 3, 224, 224), modules=None):
    """Run a real forward pass and record (name, type, output shape, #params)."""
    rows, hooks = [], []
    for name, mod in modules:
        def hook(m, i, o, name=name):
            rows.append((name, type(m).__name__, tuple(o.shape[1:]),
                         sum(p.numel() for p in m.parameters())))
        hooks.append(mod.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return rows


def shape_str(shape):
    return "×".join(str(s) for s in shape)


def lr_schedule(lr, epochs):
    """The real learning rate used in each epoch (same scheduler as train.fit)."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    out = []
    for _ in range(epochs):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def transform_lines(kind):
    return [repr(t) for t in D.get_transforms(kind).transforms]


def collect_facts():
    F = {"cfg": notebook_settings(), "sections": notebook_sections()}
    cnn = M.SimpleCNN(len(D.PV_CLASSES))
    mods = []
    for b, block in enumerate(cnn.features):
        for k, layer in enumerate(block):
            mods.append((f"features[{b}][{k}]", layer))
    mods += [("pool", cnn.pool), ("classifier[0]", cnn.classifier[0]), ("classifier[1]", cnn.classifier[1])]
    F["cnn_layers"] = layer_table(cnn, modules=mods)
    F["cnn_blocks"] = layer_table(cnn, modules=[(f"features[{b}]", blk) for b, blk in enumerate(cnn.features)])
    F["cnn_params"] = M.count_parameters(cnn)
    F["cnn_dropout"] = cnn.classifier[0].p

    eff = M.build_efficientnet(len(D.PV_CLASSES), pretrained=False)
    F["eff_stages"] = layer_table(eff, modules=[(f"features[{i}]", m) for i, m in enumerate(eff.features)]
                                  + [("avgpool", eff.avgpool), ("classifier[0]", eff.classifier[0]),
                                     ("classifier[1]", eff.classifier[1])])
    F["eff_params"] = M.count_parameters(eff)
    F["eff_trainable"] = sum(p.numel() for p in M.get_model("effnet", 38, pretrained=False).parameters()
                             if p.requires_grad)
    F["eff_dropout"] = eff.classifier[0].p
    F["eff_head_in"] = eff.classifier[1].in_features
    frozen = M.build_efficientnet(38, pretrained=False, freeze_backbone=True)
    F["eff_trainable_frozen"] = sum(p.numel() for p in frozen.parameters() if p.requires_grad)

    cfg = F["cfg"]
    F["lr_cnn"] = lr_schedule(cfg["CNN_LR"], cfg["CNN_EPOCHS"])
    F["lr_eff"] = lr_schedule(cfg["EFFNET_LR"], cfg["EFFNET_EPOCHS"])
    F["tf"] = {k: transform_lines(k) for k in ("eval", "standard", "strong")}
    F["weight_decay"] = default(T.fit, "weight_decay")
    F["mask_threshold"] = default(D.make_leaf_mask, "threshold")
    F["keep_correct"] = default(E.rejection_analysis, "keep_correct")
    F["overlay_alpha"] = default(G.overlay, "alpha")
    F["n_overlap"] = len(D.OVERLAP_CLASSES)
    F["not_in_plantdoc"] = sorted(set(D.PV_CLASSES) - set(D.OVERLAP_CLASSES))
    F["stats"] = json.load(open(os.path.join(RESULTS, "dataset_stats.json"))) \
        if os.path.exists(os.path.join(RESULTS, "dataset_stats.json")) else None
    return F


# ---- results (only from a real, full run) ---------------------------------
def load(path):
    full = os.path.join(RESULTS, path)
    return json.load(open(full)) if os.path.exists(full) else None


def results_are_real():
    """A full Colab run tests on ~2,200 PlantVillage images; the CPU smoke test on 38.
    Smoke-test numbers must never appear in the viva documents."""
    m = load("eval/cnn_standard_pv_test_metrics.json")
    return bool(m and m["num_images"] >= 500)


REAL = None  # set in main()
VALS = {}    # numbers for the {placeholders} in the question texts, set in main()


def fill(text):
    """Put the real numbers into a question/answer text."""
    return text.format_map(VALS)


def r(path, key=None, digits=3, pct=False):
    """A result number as text, or [pending]."""
    if not REAL:
        return PENDING
    obj = load(path)
    if obj is None:
        return PENDING
    for k in (key.split(".") if key else []):
        obj = obj.get(k) if isinstance(obj, dict) else None
        if obj is None:
            return PENDING
    if isinstance(obj, (int, float)):
        return f"{obj:.1%}" if pct else f"{obj:.{digits}f}"
    return str(obj)


def rfig(path):
    return os.path.join(RESULTS, path) if REAL else "PENDING:results/" + path


# ===========================================================================
# 3. Figures (diagrams drawn with matplotlib; examples made with OUR code)
# ===========================================================================
def _box(ax, x, y, w, h, text, fc="#e8f0fa", ec="#1f5f9e", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x1, y1, x2, y2, text=None, color="#333333"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, text, fontsize=7, ha="center", va="bottom",
                color="#444444")


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pipeline(F):
    cfg = F["cfg"]
    A, B, SH = ("#e3eefb", BLUE), ("#fdeedd", ORANGE), ("#eeeeee", GREY)
    fig, ax = _canvas(8.6, 10.5)
    _box(ax, 0.02, 0.90, 0.44, 0.08, "PlantVillage data.zip (Hugging Face)\ncolor + segmented photos, 38 classes", *SH)
    _box(ax, 0.54, 0.90, 0.44, 0.08, f"PlantDoc (GitHub, pinned commit)\nfield photos, {F['n_overlap']} shared classes", *SH)
    _box(ax, 0.02, 0.76, 0.44, 0.09, f"plan_plantvillage_split + prepare_plantvillage\nleaf-grouped split "
         f"{cfg['TRAIN_PER_CLASS']}/{cfg['VAL_PER_CLASS']}/{cfg['TEST_PER_CLASS']} per class\n+ leaf mask per image", *SH)
    _box(ax, 0.54, 0.76, 0.44, 0.09, "prepare_plantdoc\nresize to max 512 px\nfolders renamed to PV classes", *SH)
    _arrow(ax, 0.24, 0.90, 0.24, 0.85)
    _arrow(ax, 0.76, 0.90, 0.76, 0.85)
    _box(ax, 0.02, 0.62, 0.96, 0.09, "LeafDataset (background: original / leaf_only / random_bg)  +  "
         "get_transforms (eval / standard / strong)\nDataLoader, batch "
         f"{cfg['BATCH_SIZE']}, input tensor 3×224×224", *SH)
    _arrow(ax, 0.24, 0.76, 0.24, 0.71)
    _box(ax, 0.02, 0.46, 0.30, 0.11, f"SimpleCNN from scratch\ncnn_standard, cnn_strong\n"
         f"{cfg['CNN_EPOCHS']} epochs, lr {cfg['CNN_LR']:g}", *A, bold=False)
    _box(ax, 0.35, 0.46, 0.30, 0.11, "Background-bias runs\ncnn_leaf_only, cnn_random_bg", *A)
    _box(ax, 0.68, 0.46, 0.30, 0.11, f"EfficientNet-B0 (ImageNet)\neffnet_standard, effnet_strong\n"
         f"{cfg['EFFNET_EPOCHS']} epochs, lr {cfg['EFFNET_LR']:g}", *B)
    for x in (0.17, 0.50, 0.83):
        _arrow(ax, x, 0.62, x, 0.57)
    ax.text(0.5, 0.415, "run_experiment -> fit: AdamW + cosine LR, best checkpoint on validation accuracy -> checkpoints (Drive)",
            ha="center", fontsize=7.5, color="#333333")
    _box(ax, 0.02, 0.24, 0.22, 0.13, f"evaluate_experiment\nacc, P/R/F1, confusion\nPV test (38 & {F['n_overlap']})\n+ PlantDoc ({F['n_overlap']})", *SH)
    _box(ax, 0.265, 0.24, 0.22, 0.13, "evaluate_background\n_bias\n3 test backgrounds\n+ PlantDoc", *A)
    _box(ax, 0.51, 0.24, 0.22, 0.13, "Grad-CAM +\nleaf-focus score\n(gradcam.py)", *A)
    _box(ax, 0.755, 0.24, 0.225, 0.13, "rejection_analysis\nthreshold from\nvalidation set", *B)
    for x in (0.13, 0.375, 0.62, 0.865):
        _arrow(ax, x, 0.405, x, 0.37)
    _arrow(ax, 0.76, 0.76, 0.76, 0.71)
    _box(ax, 0.02, 0.07, 0.46, 0.12, "results/  (JSON, tables, plots, confusion\nmatrices, Grad-CAM, leaf focus, rejection)", *SH)
    _box(ax, 0.52, 0.07, 0.46, 0.12, "app.py (Gradio)\nupload -> prediction + confidence\n+ Grad-CAM, or 'NOT SURE'", *SH)
    _arrow(ax, 0.40, 0.24, 0.30, 0.19)
    _arrow(ax, 0.86, 0.24, 0.80, 0.19)
    ax.add_patch(FancyBboxPatch((0.02, 0.005), 0.03, 0.025, boxstyle="square,pad=0", fc=A[0], ec=A[1]))
    ax.text(0.06, 0.017, "Person A", fontsize=8, va="center")
    ax.add_patch(FancyBboxPatch((0.22, 0.005), 0.03, 0.025, boxstyle="square,pad=0", fc=B[0], ec=B[1]))
    ax.text(0.26, 0.017, "Person B", fontsize=8, va="center")
    ax.add_patch(FancyBboxPatch((0.42, 0.005), 0.03, 0.025, boxstyle="square,pad=0", fc=SH[0], ec=SH[1]))
    ax.text(0.46, 0.017, "shared", fontsize=8, va="center")
    return _save(fig, "pipeline.png")


def fig_cnn(F):
    blocks = F["cnn_blocks"]
    fig, ax = _canvas(9.5, 2.6)
    labels = ["Input\n3×224×224"] + [f"conv_block {i + 1}\n{shape_str(s)}" for i, (_, _, s, _) in enumerate(blocks)] \
        + ["Global avg\npool\n256", f"Dropout {F['cnn_dropout']}\n+ Linear\n38 logits"]
    n = len(labels)
    w = 0.98 / n - 0.02
    for i, lab in enumerate(labels):
        x = 0.01 + i * (w + 0.02)
        _box(ax, x, 0.25, w, 0.5, lab, fs=7.5,
             fc="#e3eefb" if 0 < i <= len(blocks) else "#eeeeee")
        if i < n - 1:
            _arrow(ax, x + w, 0.5, x + w + 0.02, 0.5)
    ax.text(0.5, 0.06, "each conv_block = Conv3×3 -> BatchNorm -> ReLU -> Conv3×3 -> BatchNorm -> ReLU -> MaxPool 2×2 "
            "(halves height and width)", ha="center", fontsize=7.5)
    return _save(fig, "cnn.png")


def fig_gradcam():
    fig, ax = _canvas(9.5, 4.2)
    _box(ax, 0.00, 0.62, 0.13, 0.25, "input image\n3×224×224")
    _box(ax, 0.18, 0.62, 0.18, 0.25, "network up to the\nLAST conv layer\n(target_layer)")
    _box(ax, 0.41, 0.62, 0.17, 0.25, "feature maps A\nK×h×w\n(saved by forward hook)", fc="#fdeedd", ec=ORANGE)
    _box(ax, 0.63, 0.62, 0.15, 0.25, "rest of network\n-> logits")
    _box(ax, 0.83, 0.62, 0.16, 0.25, "score of class c\n(predicted class)")
    for x1, x2 in [(0.13, 0.18), (0.36, 0.41), (0.58, 0.63), (0.78, 0.83)]:
        _arrow(ax, x1, 0.745, x2, 0.745)
    _arrow(ax, 0.91, 0.62, 0.91, 0.42, "backward()", color="#b22222")
    _box(ax, 0.80, 0.18, 0.19, 0.24, "gradients dScore/dA\nK×h×w\n(saved by tensor hook)", fc="#fbe3e3", ec="#b22222")
    _box(ax, 0.56, 0.18, 0.19, 0.24, "mean over h,w\n-> one weight α_k\nper feature map")
    _box(ax, 0.32, 0.18, 0.19, 0.24, "Σ_k α_k·A_k\nthen ReLU\n(keep positive)")
    _box(ax, 0.06, 0.18, 0.21, 0.24, "upsample to 224×224\nscale to [0,1]\n-> heatmap (overlay)", fc="#e6f4e6", ec="#2e7d32")
    for x1, x2 in [(0.80, 0.75), (0.56, 0.51), (0.32, 0.27)]:
        _arrow(ax, x1, 0.30, x2, 0.30)
    _arrow(ax, 0.495, 0.62, 0.42, 0.42)
    ax.text(0.5, 0.03, "leaf-focus score = Σ(heatmap × leaf mask) / Σ(heatmap)", ha="center", fontsize=9,
            fontweight="bold", color="#2e7d32")
    return _save(fig, "gradcam_steps.png")


def fig_transfer(F):
    fig, ax = _canvas(9.5, 3.6)
    _box(ax, 0.00, 0.45, 0.16, 0.35, "ImageNet\n1.28M images\n1000 classes", fc="#eeeeee", ec=GREY)
    _box(ax, 0.22, 0.45, 0.40, 0.35, "EfficientNet-B0 'features' (stages 0-8)\nedges -> textures -> parts\n"
         f"output {F['eff_head_in']}×7×7 -> avgpool -> {F['eff_head_in']}", fc="#fdeedd", ec=ORANGE)
    _box(ax, 0.68, 0.62, 0.30, 0.18, f"old head: Linear({F['eff_head_in']} -> 1000)\n(thrown away)",
         fc="#f3f3f3", ec="#aaaaaa")
    _box(ax, 0.68, 0.36, 0.30, 0.22, f"NEW head:\nDropout({F['eff_dropout']}) +\nLinear({F['eff_head_in']} -> 38)\n"
         "random init", fc="#e6f4e6", ec="#2e7d32", fs=8)
    _arrow(ax, 0.16, 0.625, 0.22, 0.625, "pretrained\nweights")
    _arrow(ax, 0.62, 0.70, 0.68, 0.71)
    _arrow(ax, 0.62, 0.55, 0.68, 0.47)
    ax.text(0.42, 0.25, "OUR runs (fine-tuning): every layer is trained, lr "
            f"{F['cfg']['EFFNET_LR']:g}, {F['eff_trainable']:,} trainable parameters", ha="center", fontsize=8.5,
            color=ORANGE, fontweight="bold")
    ax.text(0.42, 0.12, "feature extraction (option freeze_backbone=True, NOT used in our runs): "
            f"only the head is trained, {F['eff_trainable_frozen']:,} parameters", ha="center", fontsize=8,
            color="#555555")
    return _save(fig, "transfer.png")


def fig_rejection(F):
    fig, ax = _canvas(9.5, 2.8)
    _box(ax, 0.00, 0.40, 0.13, 0.30, "leaf photo")
    _box(ax, 0.18, 0.40, 0.16, 0.30, "model\n-> softmax\n(38 probabilities)")
    _box(ax, 0.39, 0.40, 0.17, 0.30, "confidence p =\nhighest probability")
    _box(ax, 0.61, 0.40, 0.13, 0.30, "p ≥ t ?", fc="#fff6dd", ec="#d9a400", fs=10, bold=True)
    _box(ax, 0.80, 0.62, 0.19, 0.25, "YES: show\nprediction", fc="#e6f4e6", ec="#2e7d32")
    _box(ax, 0.80, 0.13, 0.19, 0.25, "NO: 'NOT SURE'\n(best guess shown)", fc="#fbe3e3", ec="#b22222")
    for x1, x2 in [(0.13, 0.18), (0.34, 0.39), (0.56, 0.61)]:
        _arrow(ax, x1, 0.55, x2, 0.55)
    _arrow(ax, 0.74, 0.60, 0.80, 0.72)
    _arrow(ax, 0.74, 0.50, 0.80, 0.27)
    ax.text(0.37, 0.12, f"t is chosen on the VALIDATION set: the confidence below which only "
            f"{1 - F['keep_correct']:.0%} of the correct\nvalidation predictions fall (keep_correct={F['keep_correct']})",
            ha="center", fontsize=8)
    return _save(fig, "rejection.png")


def fig_split(F):
    cfg = F["cfg"]
    fig, ax = _canvas(9.5, 3.8)
    _box(ax, 0.00, 0.62, 0.22, 0.28, "official train list\n(color_train.txt)")
    _box(ax, 0.00, 0.12, 0.22, 0.28, "official test list\n(color_test.txt)\nalready leaf-grouped")
    _box(ax, 0.28, 0.62, 0.20, 0.28, "keep photos that\nhave a segmented\ntwin (for masks)")
    _box(ax, 0.28, 0.12, 0.20, 0.28, "keep photos that\nhave a segmented\ntwin (for masks)")
    _box(ax, 0.54, 0.62, 0.18, 0.28, "group photos\nby physical leaf\n(leaf map), shuffle\ngroups (seed 42)")
    _box(ax, 0.78, 0.78, 0.21, 0.17, f"val: first groups\nuntil {cfg['VAL_PER_CLASS']}/class", fc="#fdeedd", ec=ORANGE)
    _box(ax, 0.78, 0.55, 0.21, 0.17, f"train: other groups,\nrandom {cfg['TRAIN_PER_CLASS']}/class", fc="#e3eefb")
    _box(ax, 0.78, 0.12, 0.21, 0.28, f"test: random\n{cfg['TEST_PER_CLASS']}/class\n(fewer if the class\nis small)",
         fc="#e6f4e6", ec="#2e7d32")
    _arrow(ax, 0.22, 0.76, 0.28, 0.76)
    _arrow(ax, 0.22, 0.26, 0.28, 0.26)
    _arrow(ax, 0.48, 0.76, 0.54, 0.76)
    _arrow(ax, 0.72, 0.80, 0.78, 0.86)
    _arrow(ax, 0.72, 0.70, 0.78, 0.63)
    _arrow(ax, 0.48, 0.26, 0.78, 0.26)
    return _save(fig, "split.png")


def fig_tensor(F):
    cfg = F["cfg"]
    fig, ax = _canvas(9.5, 2.2)
    labels = ["JPEG file\n256×256 pixels\nRGB", "Resize\n(224, 224)", "ToTensor\n3×224×224\nvalues 0..1",
              "Normalize\n(ImageNet\nmean, std)", f"DataLoader\nbatch\n{cfg['BATCH_SIZE']}×3×224×224"]
    for i, lab in enumerate(labels):
        x = 0.01 + i * 0.2
        _box(ax, x, 0.2, 0.16, 0.6, lab, fs=8)
        if i < len(labels) - 1:
            _arrow(ax, x + 0.16, 0.5, x + 0.2, 0.5)
    return _save(fig, "tensor.png")


def fig_class_bars(F):
    st = F["stats"]
    if not st:
        return None
    classes = D.PV_CLASSES
    full = [st["pv_full"]["per_class"][c] for c in classes]
    fig, ax = plt.subplots(figsize=(9, 8.5))
    y = np.arange(len(classes))
    ax.barh(y, full, color="#7aa6d6")
    for i, v in enumerate(full):
        ax.text(v + 30, i, str(v), va="center", fontsize=7)
    ax.set_yticks(y, [E.short_name(c) for c in classes], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("images in the full PlantVillage (color) dataset")
    ax.set_title("Images per class (full dataset, before our subsampling)")
    ax.set_xlim(0, max(full) * 1.12)
    fig.tight_layout()
    return _save(fig, "class_bars.png")


def fig_split_bars(F):
    st = F["stats"]
    if not st:
        return None
    classes = D.PV_CLASSES
    pc = st["our_split"]["per_class"]
    tr = np.array([pc[c]["train"] for c in classes])
    va = np.array([pc[c]["val"] for c in classes])
    te = np.array([pc[c]["test"] for c in classes])
    fig, ax = plt.subplots(figsize=(9, 8.5))
    y = np.arange(len(classes))
    ax.barh(y, tr, color="#7aa6d6", label="train")
    ax.barh(y, va, left=tr, color="#f0a868", label="val")
    ax.barh(y, te, left=tr + va, color="#8cc68c", label="test")
    ax.set_yticks(y, [E.short_name(c) for c in classes], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("images in OUR split")
    ax.legend(loc="lower right")
    ax.set_title("Our balanced subset per class")
    fig.tight_layout()
    return _save(fig, "split_bars.png")


def _local_example(split="train"):
    root = os.path.join("data", "plantvillage", split)
    if not os.path.isdir(root):
        return None
    ds = D.LeafDataset(root, mask_root=os.path.join("data", "plantvillage", "masks", split))
    return ds if len(ds) else None


def fig_backgrounds():
    ds = _local_example("train")
    if ds is None:
        return None
    picks = list(range(0, len(ds), max(1, len(ds) // 3)))[:3]  # three different classes
    root = os.path.join("data", "plantvillage", "train")
    fig, axes = plt.subplots(len(picks), 4, figsize=(8, 2.1 * len(picks)), squeeze=False)
    for row, idx in enumerate(picks):
        imgs = [D.LeafDataset(root, background=bg, mask_root=ds.mask_root, fixed_bg_seed=42)[idx][0]
                for bg in D.BACKGROUNDS]
        mask = ds.load_mask(idx)
        for col, (im, title) in enumerate(zip([imgs[0], mask, imgs[1], imgs[2]],
                                              ["original", "leaf mask", "leaf_only", "random_bg"])):
            if col == 1:
                axes[row, col].imshow(im, cmap="gray", vmin=0, vmax=255)
            else:
                axes[row, col].imshow(im)
            axes[row, col].set_title(title if row == 0 else "", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle("Real output of LeafDataset (data.py) for three training images", fontsize=10)
    fig.tight_layout()
    return _save(fig, "backgrounds.png")


def fig_augment():
    ds = _local_example("train")
    if ds is None:
        return None
    path = ds.samples[0][0]
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("RGB")
    torch.manual_seed(0)
    fig, axes = plt.subplots(2, 5, figsize=(9, 3.9))
    for row, kind in enumerate(["standard", "strong"]):
        tf = D.get_transforms(kind)
        for col in range(5):
            if col == 0:
                axes[row, col].imshow(img)
                axes[row, col].set_title("original", fontsize=8)
            else:
                axes[row, col].imshow(G.denormalize(tf(img)))
                axes[row, col].set_title(f"{kind} #{col}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("Real output of get_transforms('standard') and get_transforms('strong')", fontsize=10)
    fig.tight_layout()
    return _save(fig, "augment.png")


def fig_lab_vs_field():
    pv = os.path.join("data", "plantvillage", "train")
    doc = os.path.join("data", "plantdoc")
    if not (os.path.isdir(pv) and os.path.isdir(doc)):
        return None
    from PIL import Image as PILImage
    classes = [c for c in D.OVERLAP_CLASSES if os.path.isdir(os.path.join(pv, c)) and os.path.isdir(os.path.join(doc, c))]
    classes = classes[:: max(1, len(classes) // 4)][:4]
    if not classes:
        return None
    fig, axes = plt.subplots(2, len(classes), figsize=(2.3 * len(classes), 4.8), squeeze=False)
    for j, c in enumerate(classes):
        for i, root in enumerate([pv, doc]):
            f = sorted(os.listdir(os.path.join(root, c)))[0]
            axes[i, j].imshow(PILImage.open(os.path.join(root, c, f)).convert("RGB"))
            axes[i, j].set_title(("lab: " if i == 0 else "field: ") + E.short_name(c), fontsize=7.5)
            axes[i, j].axis("off")
    fig.suptitle("Top: PlantVillage (lab).  Bottom: PlantDoc (field). Same classes.", fontsize=10)
    fig.tight_layout()
    return _save(fig, "lab_vs_field.png")


# ===========================================================================
# 4. The documents
# ===========================================================================
# Every item: (who, text). who = "A", "B" or "shared". Shown in the guides.
def double_check_items(F):
    cfg, st = F["cfg"], F["stats"]
    shared_leaves = st["our_split"]["shared_leaf_groups"]["train_test"] if st else PENDING
    area_max = (f"{st['pv_local']['leaf_area']['max']:.2f}" if st and st["pv_local"]["leaf_area"] else PENDING)
    return [
        ("shared", "The files now in `results/` come from a 1-epoch CPU smoke test on 2 images per class. "
                   "They are NOT results. This builder detects that and prints [pending]. Run the Colab notebook, "
                   "copy `results/` back, then run `python scripts/build_viva_pdfs.py` again."),
        ("shared", "README.md says \"see 'Getting it onto Colab' below\", but that section does not exist in the README."),
        ("shared", f"Training on the command line (`python -m src.train`) uses batch size 32 and lr 1e-3 for BOTH models "
                   f"by default. The notebook uses batch {cfg['BATCH_SIZE']}, lr {cfg['CNN_LR']:g} (CNN) and "
                   f"{cfg['EFFNET_LR']:g} (EfficientNet). Always quote the notebook values; those produced the results."),
        ("A", "`fit` (train.py) always runs ALL epochs and only keeps the checkpoint with the best validation accuracy. "
              "The code comment says \"early-stopping style\"; it is not real early stopping (training never stops early)."),
        ("B", "`build_efficientnet(freeze_backbone=True)` exists, but `get_model` never passes it, so feature extraction "
              "was never run. We have no numbers for it; only describe it as an option."),
        ("shared", "The eval transform is `Resize((224, 224))`, which squashes non-square photos. PlantVillage photos are "
                   "square (256×256), but PlantDoc photos are not (e.g. 512×384), so they get distorted. This is part "
                   "of the measured domain gap."),
        ("shared", f"Leakage: our train and val share 0 leaf groups, but train and test share {shared_leaves} leaf groups. "
                   "These come from the authors' official split, which we use as is. Also, photos missing from the "
                   "leaf map count as their own group, so the check cannot be 100% complete."),
        ("A", f"Leaf masks: pixel = leaf if its brightest channel is > {F['mask_threshold']} in the segmented image, "
              f"then a 5×5 median filter. The largest leaf share among our local masks is {area_max} (1.00 = the mask "
              "covers the whole image), so some masks are poor. Dark lesions or shadows can be misclassified."),
        ("A", "In `leaf_only` the background is solid black, which is itself one constant background. `random_bg` "
              "backgrounds are synthetic colour blobs, not real field scenes. Say this when you present the experiment."),
        ("A", "Leaf-focus score: the CAM is low-resolution (SimpleCNN 14×14, EfficientNet 7×7) and is blown up to "
              "224×224, so heat spills over the leaf border. The score cannot tell lesion from healthy leaf, and it "
              "can only be computed on PlantVillage (PlantDoc has no masks)."),
        ("A", "Grad-CAM's target layer for SimpleCNN is `features[-1]`, the whole 4th conv_block, including its MaxPool "
              "(output 256×14×14). Call it 'the output of the last conv block'."),
        ("B", "The rejection threshold is chosen on the in-domain (lab) validation set. Softmax confidence is not "
              "calibrated, so field photos can be confidently wrong. The threshold is computed only for the two "
              "EfficientNet models. app.py falls back to 0.5 if `results/rejection/<name>.json` is missing."),
        ("B", "`RandomRotation(30)` fills the corners with black, and `RandomResizedCrop(scale 0.5-1)` can crop away "
              "the diseased spot, so some augmented images may no longer show the disease."),
        ("shared", "PlantDoc's spider-mites folder has only 2 images, and several PlantDoc classes are small, so "
                   "per-class field numbers are noisy. The PlantVillage test set has at most 60 images per class "
                   "(Potato healthy 32, Apple cedar rust 52)."),
        ("shared", "Each experiment is run once with seed 42. Differences of 1-2 points between models may be noise. "
                   "We report no confidence intervals or repeated runs."),
        ("shared", "How PlantVillage was collected comes from the dataset paper (Hughes & Salathé 2015; Mohanty et al. "
                   "2016), not from our code. Check the wording against the paper before quoting details."),
        ("B", f"The cosine LR schedule is stepped once per epoch with T_max = number of epochs and no warm-up, so the "
              f"last EfficientNet epoch uses lr {F['lr_eff'][-1]:.2e} (see the LR table)."),
        ("shared", "The \"~40 min on a T4\" estimate is not measured yet. Read `train_time_sec` in "
                   "`results/history_*.json` after the run."),
    ]


def add_double_check(doc, F, who):
    items = [t for w, t in double_check_items(F) if who == "all" or w in (who, "shared")]
    doc.h1("Things to double-check before the viva")
    doc.p("These are places where the code is easy to misdescribe, or where something is limited or "
          "undocumented. Better to know them than to be surprised.")
    doc.bullets(items)


def results_note(doc):
    if REAL:
        doc.p("Numbers below are read from `results/` (full Colab run).")
    else:
        doc.box("Results are not available yet",
                ["The numbers in `results/` are from a CPU smoke test (1 epoch, 2 images per class), so they "
                 "are shown as [pending] here. After the Colab run: copy `results/` into the project and run "
                 "`python scripts/build_viva_pdfs.py` again. All tables and figures fill in automatically."],
                color="#fdecec", border="#c62828")


def summary_table(doc, names):
    n = F_GLOBAL["n_overlap"]
    rows = [["Experiment", "PV test acc (38)", "PV macro-F1", f"PV acc ({n} shared)", f"PlantDoc acc ({n})",
             "PlantDoc macro-F1", "Domain gap"]]
    for name in names:
        p = f"eval/{name}_summary.json"
        rows.append([name] + [r(p, k) for k in ("pv_test_acc", "pv_test_macro_f1", "pv_overlap_acc",
                                                 "plantdoc_acc", "plantdoc_macro_f1", "domain_gap")])
    doc.table(rows, [0.2, 0.13, 0.12, 0.14, 0.14, 0.14, 0.13])


def background_results_table(doc):
    rows = [["Trained on", "test: original", "test: leaf only", "test: random bg", "PlantDoc"]]
    for name in ["cnn_standard", "cnn_leaf_only", "cnn_random_bg", "effnet_standard", "effnet_strong"]:
        rows.append([name] + [r(f"background/{name}.json", k) for k in
                              ("pv_test_original", "pv_test_leaf_only", "pv_test_random_bg", "plantdoc")])
    doc.table(rows, [0.24, 0.19, 0.19, 0.19, 0.19])


def leaf_focus_table(doc):
    rows = [["Model", "Leaf focus (mean)", "Leaf focus (median)", "Leaf area (mean)", "Focus when correct",
             "Focus when wrong"]]
    for name in ["cnn_standard", "cnn_strong", "cnn_leaf_only", "cnn_random_bg", "effnet_standard", "effnet_strong"]:
        p = f"leaf_focus/{name}.json"
        rows.append([name] + [r(p, k) for k in ("mean_leaf_focus", "median_leaf_focus", "mean_leaf_area",
                                                 "mean_leaf_focus_correct", "mean_leaf_focus_wrong")])
    doc.table(rows, [0.2, 0.16, 0.16, 0.16, 0.16, 0.16])


def rejection_table(doc):
    rows = [["Model", "Threshold", "PV test: answered", "PV test: acc on answered", "PV test: acc all",
             "PlantDoc: answered", "PlantDoc: acc on answered", "PlantDoc: acc all"]]
    for name in ["effnet_standard", "effnet_strong"]:
        p = f"rejection/{name}.json"
        rows.append([name, r(p, "threshold"), r(p, "pv_test.coverage", pct=True), r(p, "pv_test.accuracy_accepted"),
                     r(p, "pv_test.accuracy_all"), r(p, "plantdoc.coverage", pct=True),
                     r(p, "plantdoc.accuracy_accepted"), r(p, "plantdoc.accuracy_all")])
    doc.table(rows, [0.16, 0.1, 0.12, 0.13, 0.11, 0.12, 0.14, 0.12], font_size=8)


def hyper_table(doc, F, model):
    cfg = F["cfg"]
    is_cnn = model == "cnn"
    lr = cfg["CNN_LR"] if is_cnn else cfg["EFFNET_LR"]
    ep = cfg["CNN_EPOCHS"] if is_cnn else cfg["EFFNET_EPOCHS"]
    rows = [["Setting", "Value", "Where in the code", "Why"],
            ["Loss", "CrossEntropyLoss", "train.py: fit", "standard loss for picking one of many classes; "
             "includes the softmax, so the model outputs raw logits"],
            ["Optimizer", f"AdamW, weight decay {F['weight_decay']:g}", "train.py: fit",
             "Adam adapts the step size per weight; the 'W' version applies weight decay correctly (regularisation)"],
            ["Learning rate", f"{lr:g}", "notebook settings: " + ("CNN_LR" if is_cnn else "EFFNET_LR"),
             "scratch CNN can take bigger steps; a pretrained net needs smaller steps so it doesn't destroy "
             "ImageNet features" if not is_cnn else "a common, safe start value for Adam on a small CNN"],
            ["LR schedule", f"CosineAnnealingLR, T_max={ep}, stepped per epoch", "train.py: fit",
             "large steps first, small careful steps at the end"],
            ["Batch size", str(cfg["BATCH_SIZE"]), "notebook settings: BATCH_SIZE",
             "fits easily in T4 memory; 64 images give a stable gradient estimate"],
            ["Epochs", str(ep), "notebook settings: " + ("CNN_EPOCHS" if is_cnn else "EFFNET_EPOCHS"),
             "keeps all runs within ~40 min on a T4" + ("; a scratch CNN needs more epochs" if is_cnn
                                                         else "; a pretrained model converges fast")],
            ["Model selection", "best validation accuracy", "train.py: fit", "picks the epoch that generalises best "
             "without touching the test set"],
            ["Seed", str(cfg["SEED"]), "utils.py: set_seed (called in run_experiment)", "same split, init and "
             "shuffling every run"],
            ["Mixed precision", "on GPU only (autocast + GradScaler)", "train.py: train_one_epoch",
             "about 2× faster on a T4 with the same accuracy"],
            ["Input", "3×224×224, ImageNet mean/std", "data.py: get_transforms",
             "same input for both models, so the comparison is fair"]]
    doc.table(rows, [0.15, 0.22, 0.25, 0.38], font_size=8.2)


def lr_table(doc, F):
    rows = [["Epoch"] + [str(i + 1) for i in range(max(len(F["lr_cnn"]), len(F["lr_eff"])))]]
    rows.append(["SimpleCNN"] + [f"{v:.1e}" for v in F["lr_cnn"]])
    rows.append(["EfficientNet"] + [f"{v:.1e}" for v in F["lr_eff"]] + [""] * (len(F["lr_cnn"]) - len(F["lr_eff"])))
    n = len(rows[0])
    doc.table(rows, [0.14] + [0.86 / (n - 1)] * (n - 1), font_size=7.2)


# ---------------------------------------------------------------------------
# 4a. dataset_overview.pdf
# ---------------------------------------------------------------------------
def build_dataset_overview(F, figs):
    st, cfg = F["stats"], F["cfg"]
    doc = Doc("Dataset overview", "PlantVillage (training) and PlantDoc (field test). Every number on these "
              "pages was measured by `scripts/dataset_stats.py` on the real data, or read from our code.")
    if not st:
        doc.box("Statistics missing", ["Run `python scripts/dataset_stats.py` first, then rebuild."],
                color="#fdecec", border="#c62828")
        return doc.build("dataset_overview.pdf")
    full, our, pd_ = st["pv_full"], st["our_split"], st["plantdoc"]
    loc = st["pv_local"]

    doc.h1("1. What is PlantVillage?")
    doc.p(f"PlantVillage is an open collection of photos of single plant leaves, each labelled with the crop and "
          f"the disease (or 'healthy'). The full colour version has **{full['total_images']:,} images** in "
          f"**{full['num_classes']} classes** from **{full['num_crops']} crops**. {full['num_healthy_classes']} of "
          f"the classes are 'healthy', the other {full['num_classes'] - full['num_healthy_classes']} are diseases.")
    doc.p("**Who made it:** the PlantVillage project (Hughes & Salathé, 2015). It became well known through the "
          "2016 paper by Mohanty, Hughes and Salathé, 'Using deep learning for image-based plant disease detection'. "
          "We download the authors' own upload on Hugging Face (`mohanty/PlantVillage`): one 2.2 GB `data.zip` "
          f"containing three versions of every photo ({', '.join(full['versions_in_zip'])}), plus the official "
          "train/test split files and a 'leaf map' that says which photos show the same physical leaf.")
    doc.p("**How the images were collected** (from the dataset paper, not from our code): leaves were picked from "
          "the plant, laid on a plain sheet, and photographed one at a time. That is why every image shows one "
          f"centred leaf on a plain grey background. All photos are {', '.join(loc['splits']['train']['sizes']) if loc['splits']['train'] else '256×256'} "
          f"pixels and {', '.join(loc['splits']['train']['modes']) if loc['splits']['train'] else 'RGB'} "
          f"(checked on the {loc['splits']['train']['total'] if loc['splits']['train'] else 0} training images "
          "prepared on this laptop).")
    doc.h2("Why the plain background matters")
    doc.p("A plain, similar background makes the dataset easy: the model can find the leaf trivially, and accuracy "
          "on PlantVillage itself is usually very high. But a real farmer's photo has soil, other leaves, hands, "
          "shadows and uneven light. A model that only ever saw plain backgrounds may (a) never have learned to "
          "ignore clutter, and (b) secretly use the background as a clue. This is why our project includes the "
          "PlantDoc field test (domain shift) and the background-bias experiment.")
    doc.img(figs["lab_vs_field"], 0.95, "Real images from our prepared data folders.")

    doc.h1("2. Classes and crops")
    doc.p(f"The {full['num_crops']} crops: " + ", ".join(c.replace("_", " ") for c in full["crops"]) + ".")
    rows = [["#", "Crop", "Disease / healthy", "Full dataset", "Our train", "Our val", "Our test"]]
    for i, c in enumerate(D.PV_CLASSES):
        crop, dis = c.split("___")
        pc = our["per_class"][c]
        rows.append([str(i), crop.replace("_", " "), dis.replace("_", " "), str(full["per_class"][c]),
                     str(pc["train"]), str(pc["val"]), str(pc["test"])])
    rows.append(["", "TOTAL", "", f"{full['total_images']:,}", f"{our['train']:,}", f"{our['val']:,}", f"{our['test']:,}"])
    doc.table(rows, [0.05, 0.2, 0.35, 0.11, 0.1, 0.09, 0.1], font_size=7.8)
    doc.p("The '#' column is the label number the model uses: the position in `PV_CLASSES` (src/data.py).")
    doc.img(figs["class_bars"], 0.92, "Images per class in the full dataset (from the zip's file list).", max_h_frac=0.8)

    doc.h2("Class imbalance")
    lc, sc = full["largest_class"], full["smallest_class"]
    doc.p(f"The full dataset is very unbalanced: the largest class is **{E.short_name(lc[0])}** with "
          f"**{lc[1]:,}** images, the smallest is **{E.short_name(sc[0])}** with **{sc[1]:,}** images "
          f"(a ratio of about {lc[1] / sc[1]:.0f} : 1). If we trained on everything, the model would see Orange "
          "HLB 36 times more often than healthy potato, and accuracy would be dominated by the big classes.")
    doc.p(f"**Our fix:** we take at most {cfg['TRAIN_PER_CLASS']} train, {cfg['VAL_PER_CLASS']} val and "
          f"{cfg['TEST_PER_CLASS']} test images per class. Our subset is almost balanced: the largest class has "
          f"{our['largest_class'][1]} images in total, the smallest {our['smallest_class'][1]} "
          f"({E.short_name(our['smallest_class'][0])}, which simply has no more photos). Classes with fewer test "
          "images: " + ", ".join(f"{E.short_name(c)} ({n})" for c, n in our["classes_with_fewer_test"].items()) + ". "
          "We also report macro-F1, which counts every class equally.")
    doc.img(figs["split_bars"], 0.92, "Train / val / test images per class in our subset.", max_h_frac=0.8)

    doc.h1("3. How we split the data (and avoid leakage)")
    doc.p("**Leakage** means information from the test set sneaks into training. In PlantVillage, the same physical "
          "leaf was often photographed several times. If photo 1 of a leaf is in train and photo 2 in test, the "
          "model is tested on an image it has practically seen, and test accuracy is too optimistic.")
    doc.img(figs["split"], 1.0, "How plan_plantvillage_split (src/data.py) chooses the files.")
    doc.bullets([
        f"**Test** = random sample (seed {cfg['SEED']}) from the authors' official test list, which they made so "
        "that no leaf is on both sides.",
        "**Train / val** come from the official train list. We group photos by leaf (using the leaf map, see "
        "`_leaf_group`), shuffle the groups, and give whole groups to val until it has "
        f"{cfg['VAL_PER_CLASS']} images per class. The other groups go to train (random {cfg['TRAIN_PER_CLASS']} per class).",
        f"Only photos that have a 'segmented' twin are used (we need it for the leaf mask). In practice that is "
        f"all {our['color_photos_with_segmented_twin']:,} colour photos.",
    ])
    sg = our["shared_leaf_groups"]
    doc.table([["Check (leaf groups shared between ...)", "Result"],
               ["train and val", str(sg["train_val"])],
               ["val and test", str(sg["val_test"])],
               ["train and test", f"{sg['train_test']} (comes from the official split, which we use as it is)"],
               ["number of leaf groups: train / val / test",
                f"{our['leaf_groups']['train']:,} / {our['leaf_groups']['val']:,} / {our['leaf_groups']['test']:,}"]],
              [0.5, 0.5])
    doc.table([["Split", "Images", "Used for"],
               ["train", f"{our['train']:,}", "updating the weights"],
               ["val", f"{our['val']:,}", "choosing the best epoch, choosing the rejection threshold"],
               ["test (PlantVillage)", f"{our['test']:,}", "final in-domain numbers, used only once at the end"],
               ["PlantDoc (field)", f"{pd_['total_images']:,} in the repo", "final out-of-domain numbers, never trained on"]],
              [0.25, 0.2, 0.55])

    doc.h1("4. Images are tensors, not rows and columns")
    doc.p("A table (like a CSV) has rows = examples and columns = features. Images do not fit that picture well: "
          "the 'features' are pixels, and their **position** matters. A pixel only means something together with "
          "its neighbours. So we keep the 2-D layout and store each image as a 3-D array (a tensor).")
    doc.img(figs["tensor"], 1.0, "From file to model input (get_transforms('eval') in src/data.py).")
    t = st.get("tensor")
    if t:
        doc.p(f"**Real numbers from our code:** one image becomes a tensor of shape **{shape_str(t['single_image'])}** "
              f"(channels × height × width), type {t['dtype'].replace('torch.', '')}. After normalisation the "
              f"values in this example range from {t['min']:.2f} to {t['max']:.2f} (about 0 on average). A batch "
              f"stacks {cfg['BATCH_SIZE']} images: **{shape_str(t['batch_example'])}**.")
    doc.p("**Small example.** A tiny 2×2 RGB image is three 2×2 grids (one red, one green, one blue), "
          "so its shape is 3×2×2 = 12 numbers. Our 224×224 images have 3 × 224 × 224 = 150,528 numbers each. "
          "A conv layer slides small filters over these grids, so it can use the neighbourhood of every pixel. "
          "A row-and-column model would lose that.")
    doc.p("**Normalisation:** `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` subtracts the "
          "ImageNet average colour and divides by its spread, channel by channel. The pretrained model was trained "
          "on inputs like that; we use the same for the scratch CNN so both models get identical inputs.")

    doc.h1("5. Leaf masks and backgrounds")
    la = loc.get("leaf_area")
    doc.p("The zip's 'segmented' version shows the same leaf with the background painted black. `make_leaf_mask` "
          f"turns it into a mask: a pixel is 'leaf' if its brightest colour channel is above {F['mask_threshold']}, then "
          "a 5×5 median filter removes speckles. The mask is saved as a PNG next to each image "
          "(`data/plantvillage/masks/...`).")
    if la:
        doc.p(f"On the {la['num_masks']} masks prepared on this laptop, the leaf covers on average "
              f"**{la['mean']:.0%}** of the image (smallest {la['min']:.0%}, largest {la['max']:.0%}).")
    doc.img(figs["backgrounds"], 0.85, "Masks and the three background versions used in the background-bias "
            "experiment (real output of LeafDataset).")

    doc.h1("6. PlantDoc (field images)")
    doc.p(f"PlantDoc (Singh et al., 2020) contains photos from the internet and the field: cluttered backgrounds, "
          f"several leaves, hands, different light. The GitHub repository (pinned commit) has "
          f"**{pd_['folders']} class folders**, **{pd_['train_images']:,}** images in its train folder and "
          f"**{pd_['test_images']}** in its test folder, **{pd_['total_images']:,}** in total. We never train on it: "
          "we use both folders together as one field test set. `prepare_plantdoc` shrinks every photo to at most "
          "512 px on the longer side and saves it under the PlantVillage class name.")
    doc.p(f"**Overlap:** all {len(pd_['mapping'])} PlantDoc folders have a matching PlantVillage class "
          f"(`PLANTDOC_TO_PLANTVILLAGE` in src/data.py), so there are **{F['n_overlap']} shared classes**. "
          f"The other {len(F['not_in_plantdoc'])} PlantVillage classes never appear in PlantDoc.")
    rows = [["PlantDoc folder", "PlantVillage class", "train", "test"]]
    for k, v in sorted(pd_["mapping"].items(), key=lambda kv: kv[1]):
        c = pd_["per_folder"].get(k, {"train": 0, "test": 0})
        rows.append([k, E.short_name(v), str(c.get("train", 0)), str(c.get("test", 0))])
    doc.table(rows, [0.38, 0.42, 0.1, 0.1], font_size=8)
    doc.p("PlantVillage classes **not** in PlantDoc: " + ", ".join(E.short_name(c) for c in F["not_in_plantdoc"]) + ".")
    if pd_["unmapped_folders"]:
        doc.p("PlantDoc folders without a mapping (ignored): " + ", ".join(pd_["unmapped_folders"]) + ".")
    loc_pd = pd_.get("local")
    if loc_pd:
        doc.p(f"Image sizes after our resizing vary (most common in the {loc_pd['total']} images on this laptop: "
              + ", ".join(f"{k} ({v})" for k, v in list(loc_pd["sizes"].items())[:4]) + "). They are not square, "
              "unlike PlantVillage.")

    doc.box("Numbers to memorize", [
        f"PlantVillage: {full['total_images']:,} colour images, {full['num_classes']} classes, {full['num_crops']} crops "
        f"({full['num_healthy_classes']} healthy classes).",
        f"Imbalance in the full data: {E.short_name(lc[0])} {lc[1]:,} vs {E.short_name(sc[0])} {sc[1]} images.",
        f"Our subset: {our['train']:,} train / {our['val']:,} val / {our['test']:,} test "
        f"(max {cfg['TRAIN_PER_CLASS']}/{cfg['VAL_PER_CLASS']}/{cfg['TEST_PER_CLASS']} per class), seed {cfg['SEED']}.",
        "Train/val are split by physical leaf; test is the authors' official leaf-grouped test split.",
        "PlantVillage images: 256×256 RGB, one leaf, plain background.",
        f"Model input: 3×224×224 tensor; a batch is {cfg['BATCH_SIZE']}×3×224×224.",
        f"PlantDoc: {pd_['total_images']:,} field photos ({pd_['train_images']:,} + {pd_['test_images']}), "
        f"{pd_['folders']} folders, never used for training.",
        f"{F['n_overlap']} classes are shared by both datasets; {len(F['not_in_plantdoc'])} PlantVillage classes are not in PlantDoc.",
        f"Leaf mask = brightest channel > {F['mask_threshold']} in the segmented image; leaf covers about "
        f"{la['mean']:.0%} of an image on average." if la else "Leaf masks come from the segmented images.",
        "ImageNet normalisation: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).",
    ], numbered=True)
    add_double_check(doc, F, "dataset")
    return doc.build("dataset_overview.pdf")


# ---------------------------------------------------------------------------
# 4b. person_A_guide.pdf
# ---------------------------------------------------------------------------
QUESTIONS_A = [
    ("basic", "What does a convolution layer do?",
     "It slides small filters (3×3 in our SimpleCNN) over the image. At each position it multiplies the filter with "
     "the pixels under it and sums them, giving a feature map. Each filter learns to detect one pattern (an edge, a "
     "colour change, a spot). `nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)` in `conv_block`."),
    ("basic", "Why ReLU?",
     "ReLU(x) = max(0, x). Without a non-linearity, stacked layers collapse into one linear function. ReLU is cheap "
     "and does not saturate for positive values, so gradients flow well."),
    ("basic", "What does max pooling do, and where is it in your model?",
     "It keeps the largest value in each 2×2 window, halving height and width. It sits at the end of each conv_block, "
     "so the 224×224 input becomes 112, 56, 28 and then 14. It makes the network cheaper and a bit tolerant to small shifts."),
    ("basic", "Describe your baseline architecture in one breath.",
     "4 conv_blocks (32, 64, 128, 256 filters), each with two 3×3 conv + BatchNorm + ReLU and a 2×2 max-pool. Then "
     "global average pooling to a 256-vector, Dropout {cnn_dropout} and a Linear layer to 38 logits. About {cnn_params_m:.2f}M parameters."),
    ("basic", "What loss and optimizer did you use?",
     "CrossEntropyLoss and AdamW (weight decay {wd:g}), lr {cnn_lr:g} with a cosine schedule, batch {batch}, {cnn_epochs} epochs (notebook "
     "settings; fit in train.py)."),
    ("basic", "What is softmax, and where is it in your code?",
     "It turns the 38 logits into probabilities that sum to 1. It is not a layer in SimpleCNN: CrossEntropyLoss "
     "applies it internally during training, and `predict` / GradCAM / app.py apply `torch.softmax` when we need probabilities."),
    ("basic", "What is the difference between accuracy and F1?",
     "Accuracy = share of all images predicted correctly. F1 is computed per class as the harmonic mean of precision "
     "and recall; macro-F1 averages it over classes, so small classes count as much as big ones."),
    ("basic", "How do you read your confusion matrix?",
     "Rows = true class, columns = predicted class, each row normalised to sum to 1. So the diagonal is each class's "
     "recall. Bright off-diagonal cells show which classes are confused. The extra 'other' column counts predictions "
     "outside the shown classes."),
    ("intermediate", "Why global average pooling instead of flatten + a big dense layer?",
     "Flattening 256×14×14 = 50,176 values into a dense layer would need ~1.9M weights for 38 outputs, more than the "
     "whole network. GAP averages each feature map to one number (256 values), which keeps parameters low and "
     "reduces overfitting."),
    ("intermediate", "What does BatchNorm do?",
     "It normalises each channel over the batch to mean 0 and std 1, then applies a learned scale and shift. Training "
     "becomes faster and more stable. In eval mode it uses running averages instead of batch statistics, which is "
     "why we call model.eval()."),
    ("intermediate", "How did you prevent overfitting?",
     "Flip augmentation, BatchNorm, Dropout {cnn_dropout}, weight decay {wd:g}, global average pooling (few parameters), and "
     "keeping the checkpoint with the best validation accuracy. We watch the train vs val curves in history_*.png."),
    ("intermediate", "How do you know whether the model overfits?",
     "In the history plot: train accuracy keeps rising while val accuracy stalls or falls, and val loss goes up. "
     "A large gap between train and val accuracy is the warning sign."),
    ("intermediate", "Explain the three versions in your background-bias experiment.",
     "Same training leaves, different backgrounds: original (plain lab background), leaf_only (background set to "
     "black using the mask) and random_bg (leaf pasted on a new random coloured-blob background every time it is "
     "loaded). All are tested on the same test leaves with all three backgrounds, and on PlantDoc."),
    ("intermediate", "How is the leaf mask made?",
     "From PlantVillage's segmented photo of the same leaf (background painted black): pixel = leaf if its brightest "
     "channel > {mask_threshold}, then a 5×5 median filter (make_leaf_mask in data.py)."),
    ("intermediate", "Walk me through Grad-CAM in your code.",
     "A forward hook stores the last conv block's output A. We backpropagate the predicted class's logit; a tensor "
     "hook stores dScore/dA. We average the gradients over height and width to get one weight per channel, take the "
     "weighted sum of the maps, apply ReLU, upsample to 224×224 and scale to [0,1] (GradCAM.__call__)."),
    ("intermediate", "What is the leaf-focus score?",
     "sum(heatmap × leaf mask) / sum(heatmap): the share of Grad-CAM heat on the leaf. We average it over {leaf_focus_n} random "
     "PlantVillage test images and compare it with the leaf's share of the image area, which is what an evenly "
     "spread heatmap would score."),
    ("intermediate", "Why the last convolutional layer for Grad-CAM?",
     "It has the most high-level (class-specific) features, and it still keeps a spatial layout (14×14 for "
     "SimpleCNN). After global pooling the location information is gone."),
    ("tricky", "Why did you choose a small custom CNN instead of, say, VGG trained from scratch?",
     "The baseline only has to show what a network learns from our {train:,} training images alone. A huge network from scratch "
     "would overfit and take too long on a free T4. A small, readable CNN is a fair 'no prior knowledge' baseline."),
    ("tricky", "Your CNN might score very high on PlantVillage. Does that mean it understands plant disease?",
     "Not necessarily. Lab photos are easy and uniform, and the model could use shortcuts like the background. That "
     "is exactly why we check field images (PlantDoc), background swaps and Grad-CAM leaf focus."),
    ("tricky", "If accuracy drops a lot on random backgrounds for cnn_standard, what does it mean?",
     "The only thing that changed is the background, so the model must have been using the background as evidence. "
     "Training with random_bg should make it rely on the leaf, which we would see as a smaller drop and higher "
     "leaf focus. (Check the real numbers in the table.)"),
    ("tricky", "Is a high leaf-focus score proof that the model is right for the right reason?",
     "No. It only says the heat is on the leaf, not on the lesion. The heatmap is low-resolution (14×14 or 7×7) and "
     "blurred by upsampling, and the masks are imperfect. It is evidence, not proof."),
    ("tricky", "Why might cnn_leaf_only do badly on original test images?",
     "It only saw black backgrounds. A grey background is new to it, so the input distribution is different. A black "
     "background is also a constant clue, so it can be a shortcut too."),
    ("tricky", "What would you do if validation accuracy was much lower than training accuracy?",
     "Treat it as overfitting: more or stronger augmentation, more dropout or weight decay, fewer epochs (we already "
     "keep the best-val checkpoint), more data, or a smaller model. Also check for a bug, e.g. val images coming "
     "from a different distribution."),
    ("tricky", "Why do you normalise inputs with ImageNet statistics in a CNN trained from scratch?",
     "The model would also work with other statistics. We use the same input pipeline for both models so the only "
     "difference in the comparison is the model and its pretraining."),
    ("tricky", "Grad-CAM uses the logit, not the softmax probability. Why?",
     "The softmax probability of class c also depends on all other classes, so its gradient mixes in evidence "
     "against other classes. The raw class score (logit) gives cleaner per-class evidence, as in the original "
     "Grad-CAM paper. Our code calls `logits[0, class_idx].backward()`."),
]


def build_person_a(F, figs):
    cfg = F["cfg"]
    doc = Doc("Person A guide: baseline CNN, background bias, Grad-CAM",
              "Written to you, Person A. Every part refers to the real file and function. Read with the code open.")
    doc.h1("0. Your part in one paragraph")
    doc.p("You built and trained the **baseline**: a small CNN that starts from random weights and learns only from "
          "our PlantVillage subset. You evaluate it (accuracy, precision/recall/F1, confusion matrix). You run the "
          "**background-bias experiment**, which tests whether the model cheats by looking at the background. And "
          "you explain the models with **Grad-CAM** heatmaps plus our own **leaf-focus score**.")
    doc.table([["File: function", "What it does", "Why"],
               ["models.py: conv_block, SimpleCNN", "defines the baseline network", "a small, readable 'no prior "
                "knowledge' model to compare with transfer learning"],
               ["train.py: run_experiment, fit, train_one_epoch, evaluate_loss_acc",
                "trains a model, keeps the best checkpoint, saves the history", "one training loop for every experiment"],
               ["evaluate.py: predict, compute_metrics, plot_confusion_matrix, evaluate_on_folder, evaluate_experiment",
                "metrics and confusion matrices on PV test and PlantDoc", "accuracy alone hides per-class problems"],
               ["data.py: make_leaf_mask, random_background, LeafDataset (background=...)",
                "leaf masks and the three background versions", "the background-bias experiment"],
               ["evaluate.py: evaluate_background_bias, background_table, plot_background_bias",
                "tests every model on all backgrounds + PlantDoc", "measures background reliance"],
               ["gradcam.py: GradCAM, overlay, save_gradcam_examples", "heatmaps of where the model looks",
                "explainability"],
               ["gradcam.py: mask_for_cam, leaf_focus_score, evaluate_leaf_focus",
                "share of heatmap on the leaf", "turns 'looks at the leaf' into a number"]],
              [0.36, 0.34, 0.30], font_size=8.2)

    doc.h1("1. How a CNN works (the pieces you used)")
    doc.h3("Convolution and filters")
    doc.p("A filter is a small grid of weights, e.g. 3×3. It slides over the image; at each position it multiplies "
          "its 9 weights (times 3 colour channels) with the pixels below and adds them up. The result is a "
          "**feature map** that is large where the pattern matches. Example: the filter [[-1,0,1],[-1,0,1],[-1,0,1]] "
          "responds to vertical edges. Our network **learns** the filter values. `nn.Conv2d(3, 32, 3, padding=1)` "
          "learns 32 filters on the colour image; `padding=1` keeps the height and width the same.")
    doc.h3("ReLU")
    doc.p("ReLU(x) = max(0, x): negative values become 0. It adds the non-linearity that lets the network learn "
          "complex shapes. Without it, many layers would be no better than one.")
    doc.h3("Pooling")
    doc.p("`nn.MaxPool2d(2)` keeps the largest value of each 2×2 window, so height and width halve. Example: "
          "[[1,3],[2,8]] becomes 8. Fewer numbers mean faster layers, and later filters 'see' a bigger part of the leaf.")
    doc.h3("BatchNorm")
    doc.p("`nn.BatchNorm2d` normalises every channel over the batch, then rescales it with learned parameters. "
          "This makes training faster and more stable.")
    doc.h3("Fully connected layer and softmax")
    doc.p("After global average pooling, each of the 256 feature maps becomes one number. `nn.Linear(256, 38)` "
          "computes 38 weighted sums: one score (logit) per class. Softmax turns logits into probabilities: "
          "p_i = exp(z_i) / Σ exp(z_j). Example: logits (2, 1, 0) give probabilities (0.67, 0.24, 0.09). "
          "SimpleCNN outputs logits; the softmax happens inside CrossEntropyLoss during training, and via "
          "`torch.softmax` in `predict` (evaluate.py) and in the app.")

    doc.h1("2. Our baseline architecture, layer by layer")
    doc.img(figs["cnn"], 1.0, "SimpleCNN (src/models.py). Shapes measured with a real forward pass.")
    doc.p(f"Total parameters: **{F['cnn_params']:,}** (counted by `count_parameters`). The table comes from "
          "running a 224×224 image through the real model:")
    rows = [["Layer", "Type", "Output shape", "Parameters"]]
    rows += [[n, t, shape_str(s), f"{p:,}"] for n, t, s, p in F["cnn_layers"]]
    doc.table(rows, [0.28, 0.3, 0.24, 0.18], font_size=7.8)
    doc.p("**Check one number by hand:** the first conv has 32 filters of size 3×3×3 plus one bias each: "
          "32 × (27 + 1) = 896 parameters. Most parameters sit in the last block (128→256 and 256→256 filters).")
    doc.code(source(M.conv_block) + "\n\n" + source(M.SimpleCNN, "def forward", "return self.classifier"))
    doc.p(f"Source: {ref(M.conv_block)}; {ref(M.SimpleCNN)}.")

    doc.h1("3. Training setup and why")
    hyper_table(doc, F, "cnn")
    doc.p("The real learning rate in each epoch (the same scheduler, recomputed here):")
    lr_table(doc, F)
    doc.p("**How one training step works** (`train_one_epoch`): take a batch of 64 images, run the model, compute "
          "the loss, `backward()` computes the gradient of the loss for every weight, `optimizer.step()` moves each "
          "weight a little in the direction that lowers the loss. One epoch = one pass over all training images.")
    doc.code(source(T.train_one_epoch, "for images, labels in loader", "optimizer.step()"))
    doc.p("Training augmentation for cnn_standard is 'standard': " + "; ".join(f"`{t}`" for t in F["tf"]["standard"]) + ".")

    doc.h1("4. Overfitting and how we handled it")
    doc.p("**Overfitting** = the model memorises the training images (including noise and background details) "
          "instead of learning general rules. Signs: training accuracy keeps rising, validation accuracy stalls or "
          "drops, and validation loss rises. Our ~7.4k training images are few for a CNN from scratch, so this is a "
          "real risk.")
    doc.table([["Measure", "Where", "How it helps"],
               ["Data augmentation (flips; 'strong' in cnn_strong)", "data.py: get_transforms",
                "every epoch sees slightly different images"],
               [f"Dropout {F['cnn_dropout']}", "models.py: SimpleCNN.classifier",
                "randomly zeroes 30% of the 256 features during training, so the classifier cannot rely on one feature"],
               ["BatchNorm", "models.py: conv_block", "stabilises training; also a mild regulariser"],
               [f"Weight decay {F['weight_decay']:g}", "train.py: fit (AdamW)", "keeps weights small and simple"],
               ["Global average pooling", "models.py: SimpleCNN.pool", "few parameters in the classifier (256×38 + 38)"],
               ["Best checkpoint on validation accuracy", "train.py: fit",
                "we keep the epoch that generalises best (training still runs all epochs)"]],
              [0.33, 0.25, 0.42], font_size=8.2)
    doc.p("Look at `results/plots/history_cnn_standard.png`: left plot = loss, right plot = accuracy, train vs val.")
    doc.img(rfig("plots/history_cnn_standard.png"), 0.9, "Training curves of cnn_standard.")

    doc.h1("5. Evaluation: accuracy, precision, recall, F1, confusion matrix")
    doc.p("**Toy example (not our data).** 10 test images of class 'blight'; the model finds 8 of them, and it "
          "also calls 2 healthy leaves 'blight'.")
    doc.bullets(["Recall(blight) = found / really blight = 8 / 10 = 0.80",
                 "Precision(blight) = correct / predicted blight = 8 / (8 + 2) = 0.80",
                 "F1 = 2·P·R / (P + R) = 0.80. F1 is high only if both are high.",
                 "Accuracy = all correct predictions / all images (over every class).",
                 "Macro average = plain mean over the classes, so every class counts equally (good for small classes)."])
    doc.p("`compute_metrics` (evaluate.py) uses scikit-learn's `classification_report` and saves accuracy, macro "
          "precision/recall/F1 and every per-class value into `results/eval/<name>_<set>_metrics.json`.")
    doc.p("**Confusion matrix** (`plot_confusion_matrix`): rows = true class, columns = predicted class. We "
          "divide each row by its total, so each row sums to 1 and **the diagonal is the recall** of that class. "
          "A dark cell off the diagonal at (row X, column Y) means 'X is often mistaken for Y'. The extra "
          "**'other'** column counts predictions of a class outside the shown set; on PlantDoc that is one of the "
          f"{len(F['not_in_plantdoc'])} PlantVillage classes PlantDoc does not have.")
    doc.p(f"`evaluate_experiment` runs three tests per model: PlantVillage test (all 38 classes), PlantVillage test "
          f"(only the {F['n_overlap']} shared classes) and PlantDoc ({F['n_overlap']} classes). "
          f"Domain gap = PV accuracy on the {F['n_overlap']} classes − PlantDoc accuracy.")
    results_note(doc)
    summary_table(doc, ["cnn_standard", "cnn_strong", "effnet_standard"])
    doc.img(rfig("eval/cnn_standard_pv_test_confusion.png"), 0.95, "Confusion matrix, cnn_standard on PlantVillage test.")

    doc.h1("6. The background-bias experiment")
    doc.p("**Question:** does the CNN recognise the disease, or partly the plain lab background? Every PlantVillage "
          "photo has a similar background, so the background is a free clue. We cannot see that from accuracy "
          "alone, so we change only the background and watch what happens.")
    doc.table([["Training variant", "What the training images look like", "Code"],
               ["cnn_standard", "original photos", "run_experiment(..., background='original')"],
               ["cnn_leaf_only", "background set to black, only the leaf left", "background='leaf_only'"],
               ["cnn_random_bg", "leaf pasted on a NEW random colour-blob background each time it is loaded",
                "background='random_bg'"]], [0.22, 0.45, 0.33])
    doc.p("How it works in the code: `LeafDataset.__getitem__` loads the photo and its mask, builds the new "
          "background (`Image.new('RGB', ...)` = black, or `random_background`), and combines them with "
          "`Image.composite(image, new_bg, mask)`: leaf pixels from the photo, everything else from the new "
          "background. This happens **before** the transforms, so augmentation still applies. Validation images "
          "use the same background as training (fixed seed), because val picks the best checkpoint. "
          "`random_background` blows up a random 4×4 grid of colours to full size and adds pixel noise "
          "(std 20).")
    doc.code(source(D.LeafDataset.__getitem__, "if self.background", "image = Image.composite"))
    doc.p(f"`evaluate_background_bias` then tests each model on the same {F['stats']['our_split']['test'] if F['stats'] else ''} "
          "test leaves with all three backgrounds (random backgrounds fixed with seed 42, so every model sees "
          "identical images) and on PlantDoc. EfficientNet models are tested too, for comparison.")
    doc.img(figs["backgrounds"], 0.8, "The three versions (real output of our code).")
    doc.h3("What we expected")
    doc.bullets(["cnn_standard: good on 'original', worse on 'leaf only' and 'random bg' if it uses the background.",
                 "cnn_random_bg: similar accuracy on all three backgrounds (it learned that the background is "
                 "meaningless) and maybe a little better on PlantDoc.",
                 "cnn_leaf_only: good on 'leaf only', but grey or random backgrounds are new to it, so it may drop there."])
    doc.h3("What we found")
    results_note(doc)
    background_results_table(doc)
    doc.img(rfig("plots/background_bias.png"), 0.9, "Background-bias heatmap (rows = training, columns = test).")
    doc.p("**How to read it:** compare along a row. A big drop from 'original' to 'random bg' means "
          "background reliance. Compare down a column to see which training helps on that kind of test. "
          "[Write your 2-3 sentence conclusion here after the Colab run.]")

    doc.h1("7. Grad-CAM step by step, in our code")
    doc.img(figs["gradcam"], 1.0, "Grad-CAM as implemented in GradCAM.__call__ (src/gradcam.py).")
    doc.bullets([
        "**Target layer** (`get_gradcam_layer`, models.py): the last conv block, `features[-1]`. SimpleCNN gives "
        "256 maps of 14×14; EfficientNet-B0 gives 1280 maps of 7×7.",
        "**Forward hook** `_save_activation` stores the feature maps A, and registers a tensor hook so that "
        "`_save_gradient` receives dScore/dA during backward().",
        "**Class score:** `logits[0, class_idx].backward()`, by default for the predicted class (raw logit, not softmax).",
        "**Weights:** `self.gradients.mean(dim=(2, 3))` averages each gradient map, giving one weight α_k per channel.",
        "**Weighted sum + ReLU:** `F.relu((weights * self.activations).sum(dim=1))` keeps only regions that push the "
        "class score up.",
        "**Upsample + scale:** `F.interpolate(..., mode='bilinear')` to 224×224, then min-max scaling to [0, 1].",
        f"**Overlay** (`overlay`): jet colours (red = important) blended over the image with alpha {F['overlay_alpha']}.",
    ])
    doc.code(source(G.GradCAM.__call__, "logits = self.model(x)", "cam = cam / (cam.max()"))
    doc.p("`save_gradcam_examples` searches random images for 4 correct and 4 wrong predictions and plots image + "
          "overlay. On PlantVillage the title also shows the leaf-focus score.")
    doc.img(rfig("gradcam/cnn_standard_pv_test.png"), 0.6, "Grad-CAM examples, cnn_standard on PlantVillage test.",
            max_h_frac=0.8)

    doc.h1("8. The leaf-focus score")
    doc.p("Grad-CAM pictures are convincing but subjective. The leaf-focus score turns 'it looks at the leaf' into "
          "a number:")
    doc.code(source(G.leaf_focus_score))
    doc.bullets([
        "`mask_for_cam` loads the leaf mask and resizes it to the heatmap size (224×224, nearest neighbour), "
        "giving 1 = leaf and 0 = background.",
        "Score = Σ(heat × mask) / Σ(heat): the share of all heat that lies on the leaf. 1.0 = all heat on the "
        "leaf, 0.0 = all on the background.",
        f"`evaluate_leaf_focus` computes it for up to {cfg['LEAF_FOCUS_IMAGES']} random PlantVillage test images "
        "per model (seed 42), and also stores the mean **leaf area**, plus the averages for correct and wrong predictions.",
        "**Baseline to compare with:** a heatmap spread evenly over the image scores exactly the leaf's share of the "
        "area. So the score only means something if it is clearly **above** the mean leaf area.",
    ])
    doc.p("**Limits (say these yourself):** the heatmap is coarse (14×14 or 7×7 upsampled), so heat spills over "
          "the leaf edge. The masks come from a simple threshold and are not perfect. The score cannot tell a "
          "lesion from a healthy part of the leaf. And it only works on PlantVillage, because PlantDoc has no masks.")
    results_note(doc)
    leaf_focus_table(doc)

    add_double_check(doc, F, "A")
    doc.h1("25 likely viva questions for you (with model answers)")
    for level in ("basic", "intermediate", "tricky"):
        doc.h2(level.capitalize())
        for i, (lv, q, a) in enumerate(QUESTIONS_A, 1):
            if lv == level:
                doc.qa(i, q, a)
    return doc.build("person_A_guide.pdf")


# ---------------------------------------------------------------------------
# 4c. person_B_guide.pdf
# ---------------------------------------------------------------------------
QUESTIONS_B = [
    ("basic", "What is transfer learning?",
     "Reusing a network trained on one task (ImageNet: 1.28M photos, 1000 classes) as the starting point for "
     "another task (our 38 leaf classes). We keep its learned filters, replace the last layer, and train further on our data."),
    ("basic", "Which pretrained model did you use, and why?",
     "EfficientNet-B0 from torchvision with ImageNet weights (IMAGENET1K_V1). It is small (~{eff_params_m:.2f}M parameters with "
     "our head) and fast on a free T4, yet accurate, and it gives a good Grad-CAM layer (1280×7×7)."),
    ("basic", "What exactly did you change in the pretrained model?",
     "Only the last layer: classifier[1] = nn.Linear(1280, 38) instead of 1280→1000. The Dropout({eff_dropout}) before it "
     "stays. See build_efficientnet in models.py."),
    ("basic", "Did you freeze any layers?",
     "No. All layers are fine-tuned (freeze_backbone defaults to False and get_model never sets it), so all "
     "{eff_trainable:,} parameters are trainable."),
    ("basic", "What is data augmentation?",
     "Random changes to training images that keep the label: flips, crops, rotations, colour changes, blur. "
     "They are applied again every epoch, so the model sees more variety. Val/test images are never augmented."),
    ("basic", "What does 'strong' augmentation contain in your code?",
     "RandomResizedCrop(224, scale 0.5-1.0), horizontal and vertical flips, RandomRotation(30), ColorJitter "
     "(brightness, contrast and saturation 0.4, hue 0.05), and GaussianBlur (kernel 9, sigma 0.1-3) with p=0.3."),
    ("basic", "What is domain shift?",
     "The test data comes from a different distribution than the training data. Ours: lab photos (one leaf, plain "
     "background) vs field photos (clutter, several leaves, different light)."),
    ("basic", "What is the confidence used for rejection?",
     "The softmax probability of the predicted class (the largest of the 38 probabilities)."),
    ("intermediate", "Why does transfer learning work for leaves when ImageNet has hardly any diseased leaves?",
     "Early layers learn general things (edges, colours, textures) that are useful for any photo. Later layers "
     "combine them. Fine-tuning re-uses this and only adapts it to leaves, instead of learning it from {train:,} images."),
    ("intermediate", "Fine-tuning vs feature extraction: what's the difference?",
     "Feature extraction freezes the pretrained layers and trains only the new head ({eff_frozen:,} parameters). Fine-tuning "
     "also updates the pretrained layers, with a small learning rate. We fine-tune; feature extraction is an option "
     "in the code that we did not run."),
    ("intermediate", "Why is the EfficientNet learning rate ({eff_lr:g}) smaller than the CNN's ({cnn_lr:g})?",
     "The pretrained weights are already good. Big steps would destroy them ('catastrophic forgetting'). The scratch "
     "CNN starts from random weights and can take bigger steps."),
    ("intermediate", "Explain your learning-rate schedule.",
     "CosineAnnealingLR with T_max = number of epochs, stepped once per epoch: the LR starts at {eff_lr:g} and follows half "
     "a cosine wave down towards 0. Big steps first, fine adjustments at the end. No warm-up."),
    ("intermediate", "Why only {eff_epochs} epochs for EfficientNet?",
     "A pretrained model converges very fast on PlantVillage, and the total budget had to stay within ~40 minutes "
     "on a free T4. We keep the best validation checkpoint anyway."),
    ("intermediate", "How do you measure the lab-to-field gap fairly?",
     "Both accuracies are on the same {n_overlap} shared classes: PlantVillage test restricted to those classes vs PlantDoc. "
     "Domain gap = PV accuracy ({n_overlap}) − PlantDoc accuracy ({n_overlap}). The model still chooses among all 38 classes."),
    ("intermediate", "Why should colour jitter and blur help on field images?",
     "Field photos have different light, white balance and focus. If the training images vary in brightness, colour "
     "and sharpness, the model cannot rely on the exact lab colours and has to use more robust features."),
    ("intermediate", "How is the rejection threshold chosen?",
     "On the validation set only: the 5th percentile of the confidence of correct validation predictions "
     "(keep_correct=0.95). So about 95% of correct lab answers are still accepted. The test sets are never used to choose it."),
    ("intermediate", "What are coverage and 'accuracy on answered'?",
     "Coverage = the share of images whose confidence is above the threshold (the model answers). Accuracy on answered "
     "= accuracy on those images only. Raising the threshold lowers coverage and usually raises accuracy on answered."),
    ("tricky", "Why EfficientNet-B0 and not ResNet50?",
     "Both are valid. B0 has ~4M parameters vs ~25M for ResNet50, so it trains faster on a free T4 and runs fast in "
     "the Gradio app, with similar ImageNet accuracy. With more time we could compare both."),
    ("tricky", "If strong augmentation lowers PlantVillage accuracy a bit but raises PlantDoc accuracy, is that bad?",
     "No, that is the expected trade-off. The lab test is easy and over-optimistic. Field accuracy is what matters in "
     "practice, and a smaller domain gap means more robust features. (Check the real numbers in the table.)"),
    ("tricky", "Can augmentation alone close the domain gap?",
     "Probably not fully. It only imitates some differences (zoom, angle, light, blur). It cannot add cluttered "
     "backgrounds, several leaves or new disease appearances. Real field training data or domain adaptation would be "
     "the next steps."),
    ("tricky", "Why not choose the rejection threshold on PlantDoc, since that is where it matters?",
     "Then PlantDoc would no longer be an unseen test set, and the reported numbers would be optimistically biased. "
     "In a real project you would use a separate field validation set."),
    ("tricky", "The model is 95% confident on a field photo but wrong. How is that possible?",
     "Softmax confidence is not calibrated, especially for inputs unlike the training data. Networks are often "
     "over-confident out of domain. Temperature scaling or out-of-distribution detection could help; we did not do that."),
    ("tricky", "What would you do with more time to improve field accuracy?",
     "Add some labelled field images to training (or fine-tune on part of PlantDoc with a held-out test part), random "
     "background augmentation for EfficientNet, longer training, test-time augmentation, model calibration, and "
     "several seeds to report variance."),
    ("tricky", "Does RandomResizedCrop risk producing wrong labels?",
     "Yes, a little. With scale 0.5 the crop can miss the lesion, so a 'diseased' image may show only healthy tissue. "
     "Scale 0.5-1.0 keeps at least half of the image area to limit this."),
    ("tricky", "Is fine-tuning all layers always better than freezing?",
     "No. With very little data, freezing avoids overfitting and is faster. We have {train:,} training images and a domain quite "
     "different from ImageNet, so fine-tuning everything with a small LR is a reasonable choice, but we did not run "
     "the frozen variant to prove it."),
]


def build_person_b(F, figs):
    cfg = F["cfg"]
    doc = Doc("Person B guide: transfer learning, augmentation, rejection",
              "Written to you, Person B. Every part refers to the real file and function. Read with the code open.")
    doc.h1("0. Your part in one paragraph")
    doc.p("You built the **transfer-learning model**: EfficientNet-B0 pretrained on ImageNet and fine-tuned on our "
          "leaves. You evaluate it like the baseline, run the **augmentation experiments** that try to reduce the gap "
          "between lab photos (PlantVillage) and field photos (PlantDoc), and you add a **confidence threshold** so "
          "the app says 'not sure' instead of guessing.")
    doc.table([["File: function", "What it does", "Why"],
               ["models.py: build_efficientnet, get_model", "loads EfficientNet-B0 with ImageNet weights, new 38-class head",
                "transfer learning"],
               ["train.py: run_experiment, fit", "the same training loop as the baseline (AdamW, cosine LR, best val "
                "checkpoint)", "fair comparison"],
               ["data.py: get_transforms('standard' / 'strong')", "the augmentation pipelines",
                "the augmentation experiment"],
               ["evaluate.py: evaluate_experiment, results_table, plot_domain_shift",
                "PV test, PV shared classes, PlantDoc; domain gap", "measures the lab-to-field drop"],
               ["evaluate.py: rejection_analysis, coverage_and_accuracy", "chooses the threshold on val, reports "
                "coverage and accuracy", "honest 'not sure' answers"],
               ["app.py: find_threshold, build_app", "uses the threshold in the Gradio demo", "deployment behaviour"]],
              [0.36, 0.38, 0.26], font_size=8.2)

    doc.h1("1. What transfer learning is and why it works")
    doc.p("A CNN trained on ImageNet (1.28 million photos, 1000 classes) has learned a hierarchy of features. "
          "Early layers detect edges and colour blobs, middle layers textures and simple shapes, late layers object "
          "parts. Most of this is useful for any photo, including leaves with spots. **Transfer learning** re-uses "
          "these weights as the starting point instead of random numbers. Our ~7.4k training images are then only "
          "needed to adapt the features to leaf diseases, not to learn vision from zero. Typical result: higher "
          "accuracy, in fewer epochs, with less overfitting.")
    doc.img(figs["transfer"], 1.0, "What build_efficientnet (src/models.py) does.")

    doc.h1("2. EfficientNet-B0 at a high level")
    doc.p("EfficientNet-B0 (Tan & Le, 2019) is a CNN built from **MBConv** blocks: a cheap 1×1 conv expands the "
          "channels, a **depthwise** 3×3 or 5×5 conv filters each channel separately (much cheaper than a normal "
          "conv), a **squeeze-and-excitation** step re-weights channels by importance, and a 1×1 conv shrinks the "
          "channels again, with a skip connection. Activation is SiLU (x·sigmoid(x)). 'B0' is the smallest member of "
          "the family; bigger versions scale depth, width and resolution together ('compound scaling').")
    doc.p("Real output shapes and parameter counts of the torchvision model with our 38-class head "
          "(forward pass of one 224×224 image):")
    rows = [["Part", "Type", "Output shape", "Parameters"]]
    rows += [[n, t, shape_str(s), f"{p:,}"] for n, t, s, p in F["eff_stages"]]
    doc.table(rows, [0.22, 0.34, 0.24, 0.2], font_size=8)
    doc.p(f"Total parameters: **{F['eff_params']:,}**, of which trainable in our runs: **{F['eff_trainable']:,}** "
          f"(all of them). With `freeze_backbone=True` only **{F['eff_trainable_frozen']:,}** would be trainable "
          "(the new head).")
    doc.code(source(M.build_efficientnet, "weights =", "return model"))
    doc.p(f"Source: {ref(M.build_efficientnet)}.")

    doc.h1("3. What we froze or unfroze, and why")
    doc.table([["", "Feature extraction", "Fine-tuning (what we ran)"],
               ["Pretrained layers", "frozen (requires_grad=False)", "updated, with a small learning rate"],
               ["Trainable parameters", f"{F['eff_trainable_frozen']:,} (head only)", f"{F['eff_trainable']:,} (all)"],
               ["Pros", "very fast, little overfitting with tiny data", "features adapt to leaves and diseases; "
                "usually more accurate when the data differ from ImageNet"],
               ["In our code", "`build_efficientnet(freeze_backbone=True)`, never called by the notebook",
                "`get_model('effnet', 38)` -> freeze_backbone=False"]], [0.22, 0.36, 0.42])
    doc.p("Why we fine-tune everything: leaf disease spots look nothing like ImageNet's dogs and cars, we have "
          "several thousand images (enough to adapt the backbone a little), and the small learning rate protects "
          "the pretrained features. Be honest: we did not run the frozen variant, so we cannot show numbers that "
          "fine-tuning is better here.")

    doc.h1("4. Training setup and learning-rate schedule")
    hyper_table(doc, F, "effnet")
    doc.p("The real learning rate in each epoch (same scheduler, recomputed here):")
    lr_table(doc, F)
    doc.p("Cosine annealing: lr_e = lr_max · (1 + cos(π·e / T_max)) / 2 for epoch e = 0 … T_max−1. Big steps "
          "while the head is still random, then smaller and smaller steps to settle into a good minimum.")

    doc.h1("5. Evaluation of the transfer model")
    doc.p("The same functions as for the baseline (`evaluate_experiment`): accuracy, macro precision/recall/F1, a "
          "row-normalised confusion matrix (diagonal = recall), on three test sets: PlantVillage test (38 classes), "
          f"PlantVillage test restricted to the {F['n_overlap']} shared classes, and PlantDoc ({F['n_overlap']} classes).")
    results_note(doc)
    summary_table(doc, ["cnn_standard", "effnet_standard", "cnn_strong", "effnet_strong"])
    doc.p("[After the Colab run, write 2-3 sentences: how much better is effnet_standard than cnn_standard on "
          "PlantVillage, and on PlantDoc? Which classes are hardest (per_class in "
          "`results/eval/effnet_standard_pv_test_metrics.json`)?]")
    doc.img(rfig("plots/history_effnet_standard.png"), 0.9, "Training curves of effnet_standard.")
    doc.img(rfig("eval/effnet_standard_plantdoc_confusion.png"), 0.95, "Confusion matrix of effnet_standard on PlantDoc.")

    doc.h1("6. Domain shift: lab to field")
    doc.p("Domain shift = the test images come from a different distribution than the training images. Our model "
          "only saw PlantVillage: one centred leaf, plain background, controlled light. PlantDoc photos have soil, "
          "sky, hands, several leaves, shadows and blur. Even if the disease looks the same, the image statistics "
          "are different.")
    doc.img(figs["lab_vs_field"], 0.9, "Same classes, lab (top) vs field (bottom).")
    doc.bullets([f"We compare on the **same {F['n_overlap']} classes** so the comparison is fair.",
                 "The model still picks among all 38 classes, as a real app would. Picking a class PlantDoc "
                 "does not have simply counts as wrong (the 'other' column of the confusion matrix).",
                 "**Domain gap = PV accuracy (shared classes) − PlantDoc accuracy.**",
                 "PlantDoc photos are not square, and `Resize((224, 224))` squashes them; this distortion is part of the gap."])

    doc.h1("7. The augmentation experiments")
    doc.p("Idea: if the training images already vary the way field photos vary, the model learns features that do "
          "not depend on those details. We retrain **both** models with 'strong' augmentation, with everything else "
          "unchanged (same data, epochs, LR, seed), and compare before and after.")
    rows = [["Transform (exact code)", "What it does", "Why it should help on field photos"]]
    why = {
        "RandomResizedCrop": ("crops a random 50-100% area (aspect 3:4 to 4:3) and resizes it to 224",
                              "field photos are taken at different distances; the leaf is not always centred and full-size"),
        "RandomHorizontalFlip": ("mirrors left-right with p=0.5", "a leaf can point either way"),
        "RandomVerticalFlip": ("mirrors top-bottom with p=0.5", "photos are taken from any angle"),
        "RandomRotation": ("rotates by a random angle in [-30°, 30°]; corners become black",
                           "hand-held photos are tilted"),
        "ColorJitter": ("random brightness, contrast, saturation (±40%) and hue (±0.05)",
                        "sunlight, shade, different cameras and white balance"),
        "RandomApply": ("with p=0.3: Gaussian blur, kernel 9, sigma 0.1-3",
                        "out-of-focus or motion-blurred phone photos"),
        "ToTensor": ("PIL image -> 3×224×224 tensor in [0, 1]", "(needed, not augmentation)"),
        "Normalize": ("ImageNet mean/std", "(needed, not augmentation)"),
    }
    for t in F["tf"]["strong"]:
        key = t.split("(")[0]
        what, reason = why.get(key, ("", ""))
        rows.append([t, what, reason])
    doc.table(rows, [0.38, 0.31, 0.31], font_size=7.6)
    doc.p("'Standard' augmentation (the 'before' runs) is only: " + "; ".join(f"`{t}`" for t in F["tf"]["standard"]) + ".")
    doc.img(figs["augment"], 0.95, "The same training image through get_transforms('standard') and ('strong').")
    doc.h3("Before / after")
    results_note(doc)
    rows = [["Model", "Augmentation", "PV acc (38)", f"PV acc ({F['n_overlap']})", f"PlantDoc acc ({F['n_overlap']})",
             "Domain gap"]]
    for name, model, aug in [("cnn_standard", "SimpleCNN", "standard (before)"), ("cnn_strong", "SimpleCNN", "strong (after)"),
                             ("effnet_standard", "EfficientNet-B0", "standard (before)"),
                             ("effnet_strong", "EfficientNet-B0", "strong (after)")]:
        p = f"eval/{name}_summary.json"
        rows.append([model, aug, r(p, "pv_test_acc"), r(p, "pv_overlap_acc"), r(p, "plantdoc_acc"), r(p, "domain_gap")])
    doc.table(rows, [0.2, 0.2, 0.14, 0.14, 0.16, 0.16])
    doc.img(rfig("plots/domain_shift.png"), 0.95, "Lab vs field accuracy for every experiment.")
    doc.p("**How to read it:** the gap shrinking from 'standard' to 'strong' means augmentation helped robustness. "
          "A small loss on PlantVillage with a gain on PlantDoc is a good trade. [Write your conclusion here.]")

    doc.h1("8. Confidence and the rejection threshold")
    doc.img(figs["rejection"], 1.0, "How app.py decides between a prediction and 'NOT SURE'.")
    doc.p("Any classifier always outputs some class, even for a blurry photo or a non-leaf. Rejection lets the "
          "system say 'I am not sure' when the confidence is low. **Confidence** = the softmax probability of the "
          "predicted class.")
    doc.code(source(E.rejection_analysis, "y_true, y_pred, probs = predict_dataset(model, pv_dataset",
                    "threshold = float(np.quantile"))
    doc.bullets([
        f"**How the threshold is chosen:** on the validation set only. Take the confidences of the validation "
        f"images the model got right; the threshold is the value below which {1 - F['keep_correct']:.0%} of them "
        f"fall (`keep_correct={F['keep_correct']}`). So on lab images about {F['keep_correct']:.0%} of the correct answers survive.",
        "**Why the validation set:** choosing it on a test set would make the test numbers optimistic (leakage).",
        "**Coverage** = share of test images with confidence ≥ threshold (they get an answer). **Accuracy on "
        "answered** = accuracy on those images only (`coverage_and_accuracy`).",
        "**Trade-off:** a higher threshold means fewer answers (lower coverage) but more of them are right. The "
        "plot `results/rejection/<model>_coverage_accuracy.png` shows the whole curve for thresholds 0 to 0.99; "
        "the dot marks our threshold.",
        "Computed for effnet_standard and effnet_strong. The app reads `results/rejection/<checkpoint name>.json` "
        "(`find_threshold`); if the file is missing it uses 0.5, and `--threshold` overrides both.",
    ])
    results_note(doc)
    rejection_table(doc)
    doc.img(rfig("rejection/effnet_strong_coverage_accuracy.png"), 0.75, "Coverage vs accuracy, effnet_strong.")
    doc.p("**What to expect and say:** on PlantDoc the model should be less confident, so more images get "
          "rejected (lower coverage), and accuracy on the answered ones should be higher than accuracy on all of them. "
          "But softmax confidence is not calibrated: a field photo can be confidently wrong, so rejection reduces "
          "mistakes without removing them.")

    add_double_check(doc, F, "B")
    doc.h1("25 likely viva questions for you (with model answers)")
    for level in ("basic", "intermediate", "tricky"):
        doc.h2(level.capitalize())
        for i, (lv, q, a) in enumerate(QUESTIONS_B, 1):
            if lv == level:
                doc.qa(i, q, a)
    return doc.build("person_B_guide.pdf")


# ---------------------------------------------------------------------------
# 4d. shared_guide.pdf
# ---------------------------------------------------------------------------
CROSS_QUESTIONS = [
    ("Plant disease classification on PlantVillage has been done many times. What is new in your project?",
     "We do not claim a new model. Our contribution is an honest evaluation: lab vs field accuracy on the "
     "same shared classes, a background-bias test with leaf masks, a measurable Grad-CAM leaf-focus score, and a "
     "rejection threshold, all reproducible on free hardware."),
    ("Why is accuracy not enough?",
     "It hides per-class problems (a small class can be always wrong while total accuracy stays high), it says nothing "
     "about why the model decides (shortcuts), and lab accuracy says little about field performance. So we add "
     "macro-F1, confusion matrices, Grad-CAM/leaf focus and the PlantDoc test."),
    ("How did you prevent data leakage?",
     "Train/val are split by physical leaf using the leaf map; test is the authors' official leaf-grouped split; "
     "PlantDoc is never used for training; the rejection threshold and best epoch are chosen on validation only. "
     "Residual: {shared_tt} leaf groups are shared between train and test through the official split."),
    ("What is domain shift, and how big is it here?",
     "A difference between the training and deployment data distributions; here lab vs field photos. The size is "
     "the 'domain gap' column in our table: PV accuracy minus PlantDoc accuracy on the same shared classes. [quote numbers]"),
    ("Why did you use PlantDoc only for testing?",
     "To measure how a lab-trained model behaves on photos it has never seen, which is the realistic situation. "
     "Training on it would remove our only out-of-domain test."),
    ("Would you deploy this model for farmers?",
     "Not as it is. Field accuracy is lower than lab accuracy, confidence can be wrong, and it only knows 38 classes. "
     "It could be an assistant that says 'not sure' and points to an expert, after training on real field data and "
     "testing in the target region."),
    ("What are the ethical risks?",
     "A wrong diagnosis could lead to wrong or unnecessary pesticide use, or to missing a disease and losing a crop. "
     "Users might over-trust a confident app. The data come from a few crops and regions, so the model may be biased. "
     "Hence rejection, clear disclaimers and expert follow-up."),
    ("Why is the CNN trained from scratch at all if transfer learning is better?",
     "It is the baseline. Without it we could not show how much pretraining helps; it is also the model in the "
     "background-bias experiment, because a scratch model can only learn from our data."),
    ("How reproducible are your results?",
     "Seeds are fixed (set_seed, seed_worker, seeded generator), cudnn is deterministic, the PlantDoc commit is pinned, "
     "and the split comes from deterministic code. Small GPU differences are still possible, and each experiment ran "
     "once, so we report no variance."),
    ("Explain Grad-CAM in one sentence.",
     "The gradient of the class score tells how important each feature map of the last conv layer is; the weighted "
     "sum of the maps, after ReLU and upsampling, shows where the evidence for that class is."),
    ("Is Grad-CAM a proof of what the model does?",
     "No, it is an approximation at low resolution. That is why we add the leaf-focus number and the background-swap "
     "experiment, which test the same question in different ways."),
    ("Your test set is small. Is that a problem?",
     "At most {test_per_class} images per class ({test:,} in total; some classes fewer), so per-class numbers can vary by several "
     "points. Overall accuracy over {test:,} images is fairly stable. PlantDoc classes are also small."),
    ("Why these hyper-parameters? Did you tune them?",
     "No systematic tuning; they are standard values (Adam-type optimizer, 1e-3 scratch / 5e-4 fine-tuning, cosine "
     "schedule) chosen to fit the 40-minute free-GPU budget. Validation accuracy picks the epoch."),
    ("What does the 'other' column in the PlantDoc confusion matrix mean?",
     "Predictions of a PlantVillage class that PlantDoc does not have ({n_not} such classes). They are always wrong, and "
     "show that the model is confused by field images."),
    ("Why do you resize to 224×224?",
     "That is the resolution EfficientNet-B0 was trained on with ImageNet; we use the same input for the CNN for a "
     "fair comparison. It is also fast."),
    ("What would you do differently with more time and compute?",
     "Train on some labelled field images, try more models and seeds, tune hyper-parameters, calibrate confidence, "
     "use the full dataset, and test on photos from our own region."),
    ("What is the difference between validation and test sets in your project?",
     "Validation is used during development (best epoch, rejection threshold). The test sets (PlantVillage test, "
     "PlantDoc) are only used once at the end for the reported numbers."),
    ("How does the Gradio app work?",
     "app.py loads a checkpoint, resizes and normalises the uploaded photo, runs the model and Grad-CAM, shows the "
     "top-5 probabilities and the overlay, and says 'NOT SURE' when the confidence is below the saved threshold."),
    ("Why does the model sometimes predict a class from the wrong crop on field photos?",
     "Field photos contain clutter and other textures, and the model never learned to ignore them. It also never "
     "learned 'which crop is this' separately; it picks among all 38 classes at once."),
    ("Who did what?",
     "Person A: baseline CNN, its evaluation, the background-bias experiment, Grad-CAM and leaf focus. Person B: "
     "transfer learning, its evaluation, the augmentation experiments, the rejection threshold. Together: the "
     "comparison tables, the Gradio demo, the Colab runs, the README and the report."),
]

GLOSSARY = [
    ("Accuracy", "Share of all test images that were classified correctly."),
    ("Activation / feature map", "The output grid of a conv layer; large values where its pattern was found."),
    ("AdamW", "Optimizer that adapts the step size per weight and applies weight decay separately."),
    ("Augmentation", "Random label-preserving changes to training images (flip, crop, colour, blur) to add variety."),
    ("Backpropagation", "Computing the gradient of the loss for every weight, layer by layer from output to input."),
    ("Background bias / shortcut", "When a model uses an unrelated clue (like the background) instead of the real evidence."),
    ("Batch", "A group of images processed together; ours has 64 images, shape 64×3×224×224."),
    ("BatchNorm", "Normalises each channel over the batch, then rescales it; makes training stable."),
    ("Checkpoint", "A saved copy of the model weights (.pt file), here the epoch with the best validation accuracy."),
    ("Class imbalance", "Some classes have many more images than others."),
    ("Confidence", "The softmax probability of the predicted class."),
    ("Confusion matrix", "Table of true class (rows) vs predicted class (columns); ours is row-normalised."),
    ("Convolution", "Sliding a small learned filter over the image and summing the products at each position."),
    ("Cosine annealing", "Learning-rate schedule that decreases the LR along half a cosine curve."),
    ("Coverage", "Share of images the system answers instead of rejecting."),
    ("Cross-entropy loss", "Loss that is small when the model gives a high probability to the true class."),
    ("Data leakage", "Information from the test data getting into training, making results look too good."),
    ("Domain gap", "PlantVillage accuracy minus PlantDoc accuracy on the same shared classes."),
    ("Domain shift", "Test/deployment data differ in distribution from the training data."),
    ("Dropout", "Randomly zeroes some features during training so the model cannot rely on any single one."),
    ("EfficientNet-B0", "A small, efficient CNN made of MBConv blocks; our pretrained model."),
    ("Epoch", "One pass over all training images."),
    ("F1 score", "Harmonic mean of precision and recall for one class."),
    ("Feature extraction", "Transfer learning with the pretrained layers frozen; only the new head is trained."),
    ("Fine-tuning", "Transfer learning where the pretrained layers are also updated (what we did)."),
    ("Global average pooling", "Averages each feature map to one number."),
    ("Grad-CAM", "Heatmap of the image regions that increased the score of a class, from last-layer gradients."),
    ("Gradient", "How much the loss (or a score) changes when a weight or activation changes a little."),
    ("ImageNet", "1.28M labelled photos in 1000 classes, used to pretrain our EfficientNet."),
    ("Leaf-focus score", "Share of the Grad-CAM heat that lies inside the leaf mask."),
    ("Leaf mask", "Black-and-white image: white = leaf pixel, made from PlantVillage's segmented photo."),
    ("Learning rate", "Size of each weight update step."),
    ("Logit", "Raw class score before softmax."),
    ("Macro average", "Average of a per-class metric where every class counts equally."),
    ("Max pooling", "Keeps the largest value in each small window, shrinking the feature map."),
    ("Mixed precision", "Using 16-bit floats on the GPU for speed, with a GradScaler for safety."),
    ("Normalisation (input)", "Subtracting the mean and dividing by the std per colour channel."),
    ("Overfitting", "Model memorises training data and does worse on new data."),
    ("PlantDoc", "Dataset of real field photos of leaves; our out-of-domain test set."),
    ("PlantVillage", "Dataset of lab photos of single leaves on a plain background; our training data."),
    ("Precision", "Of the images predicted as class c, the share that really are c."),
    ("Recall", "Of the images that really are class c, the share the model found."),
    ("Rejection threshold", "Minimum confidence needed to give an answer; below it the app says 'not sure'."),
    ("ReLU", "Activation function max(0, x)."),
    ("Seed", "Fixed starting value for random number generators so runs repeat exactly."),
    ("Softmax", "Turns logits into probabilities that sum to 1."),
    ("Tensor", "A multi-dimensional array; an image is 3×224×224."),
    ("Transfer learning", "Starting from a model trained on another task instead of random weights."),
    ("Validation set", "Data used during development to pick the best epoch and the threshold, never for the final score."),
    ("Weight decay", "Penalty that keeps weights small, to reduce overfitting."),
]


def build_shared(F, figs):
    cfg, st = F["cfg"], F["stats"]
    n_tr = f"{st['our_split']['train']:,}" if st else "~7.4k"
    doc = Doc("Shared guide: the whole project", "For both of you: the story, the pipeline, how to explain each "
              "other's half, honest limitations, cross-questions and a glossary.")
    doc.h1("1. The 20-second pitch")
    doc.box("Say this", [
        "We detect 38 plant diseases from leaf photos. We compare a small CNN trained from scratch with a pretrained "
        "EfficientNet, and then we test what lab-trained models do on real field photos: we measure the accuracy drop, "
        "check with Grad-CAM and a background-swap experiment whether the model looks at the leaf, try stronger "
        "augmentation to close the gap, and let the app say 'not sure' when it is unsure."], color="#e8f3e8", border="#2e7d32")
    doc.h1("2. The project story in 2 minutes")
    doc.p(f"**Problem.** Plant diseases cost farmers a lot of harvest, and experts are not always available. A phone "
          f"app that recognises a disease from a leaf photo would help, but it has to work on real field photos, not "
          f"only on clean lab pictures.")
    doc.p(f"**Data.** We train on PlantVillage: lab photos of single leaves on a plain background, 38 classes, 14 "
          f"crops. We take a balanced subset ({n_tr} train / {st['our_split']['val'] if st else 1520:,} val / "
          f"{st['our_split']['test'] if st else 2244:,} test) and split it by physical leaf to avoid leakage. We "
          f"test field performance on PlantDoc ({st['plantdoc']['total_images'] if st else 2578:,} real photos, "
          f"{F['n_overlap']} shared classes), never training on it.")
    doc.p(f"**Methods.** (1) A small CNN from scratch vs EfficientNet-B0 fine-tuned from ImageNet. (2) Accuracy, "
          f"macro-F1 and confusion matrices. (3) Grad-CAM heatmaps and a leaf-focus score. (4) A background-bias "
          f"experiment: training on original, leaf-only and random-background images. (5) Strong augmentation "
          f"against the lab-to-field gap. (6) A confidence threshold for rejecting unsure predictions. Everything "
          f"runs in about 40 minutes on a free Colab T4.")
    doc.p("**Findings.** " + ("[Read from the tables in the person guides after the Colab run: transfer learning vs "
                              "scratch on PlantVillage; the size of the domain gap; whether strong augmentation reduced "
                              "it; whether background swaps hurt the baseline; leaf focus vs leaf area; coverage and "
                              "accuracy after rejection.]" if not REAL else
                              "[Summarise the numbers from the tables below in 3-4 sentences in your own words.]"))
    doc.p("**Limits.** Lab-to-field generalisation stays the weak point; augmentation imitates only part of the "
          "difference; test sets are small per class; one run per experiment; no field data in training.")
    results_note(doc)
    summary_table(doc, ["cnn_standard", "effnet_standard", "cnn_strong", "effnet_strong", "cnn_leaf_only", "cnn_random_bg"])

    doc.h1("3. The whole pipeline on one page")
    doc.img(figs["pipeline"], 0.98, "Blue = Person A, orange = Person B, grey = shared. Function names are the real ones.",
            max_h_frac=0.86)

    doc.h1("4. Explaining the other person's half (5 sentences each)")
    doc.h2("Person A explains Person B's half")
    doc.bullets([
        "Person B took EfficientNet-B0, pretrained on ImageNet, replaced its last layer with a 38-class layer, and "
        f"fine-tuned all layers with a smaller learning rate ({cfg['EFFNET_LR']:g}) for {cfg['EFFNET_EPOCHS']} epochs.",
        "Because the network already knows general visual features, it learns faster and generalises better than "
        "our scratch CNN.",
        f"B measured the lab-to-field drop on the {F['n_overlap']} classes shared with PlantDoc.",
        "B retrained both models with strong augmentation (random crops, rotation, colour jitter, blur) to imitate "
        "field conditions, and compared the gap before and after.",
        "B added a rejection threshold, chosen on the validation set, so the app says 'not sure' instead of guessing.",
    ])
    doc.h2("Person B explains Person A's half")
    doc.bullets([
        "Person A designed a small CNN from scratch: 4 conv blocks (32 to 256 filters), global average pooling, "
        f"dropout and a 38-class output, about {F['cnn_params'] / 1e6:.2f}M parameters, trained {cfg['CNN_EPOCHS']} epochs.",
        "A evaluated it with accuracy, macro-F1 and a row-normalised confusion matrix, whose diagonal is per-class recall.",
        "A tested background bias by training on original, leaf-only and random-background images, made with "
        "PlantVillage's leaf masks, and testing every model on all three backgrounds.",
        "A implemented Grad-CAM, which weights the last conv layer's feature maps by the gradients of the class "
        "score, to show where the model looks.",
        "A turned this into a number, the leaf-focus score: the share of the heatmap that lies on the leaf.",
    ])

    doc.h1("5. Shared parts")
    doc.bullets([
        "**Comparison tables:** `results_table` (evaluate.py) writes `results/results_table.md`; "
        "`background_table` writes `results/background_table.md`; the notebook writes `results/leaf_focus_table.md`.",
        "**Gradio demo:** `app.py` (local: `python app.py`; in Colab: last notebook cell with `--share`).",
        "**Colab runs:** the notebook `notebooks/train_plant_disease.ipynb` runs everything in order. Its sections: "
        + "; ".join(F["sections"]) + ".",
        "**README:** overview, how to run, results table (fill in after the run), cheat sheet.",
        "**These PDFs:** `python scripts/dataset_stats.py` then `python scripts/build_viva_pdfs.py`.",
    ])

    doc.h1("6. Honest limitations, and how to say them confidently")
    doc.table([["Limitation", "How to say it"],
               ["PlantVillage is a lab dataset with plain backgrounds", "\"We knew lab accuracy would be optimistic, "
                "so we measured the field gap directly instead of hiding it.\""],
               ["Small test sets (≤60 per class; PlantDoc classes small)", "\"Per-class numbers are noisy; we rely on "
                "overall accuracy and macro-F1 and do not over-interpret single classes.\""],
               ["Only one run per experiment", "\"Differences of 1-2 points may be noise; with more compute we would "
                "repeat with several seeds.\""],
               [f"Only {F['n_overlap']} of 38 classes exist in PlantDoc", "\"We compare on exactly the shared classes, "
                "so the comparison is fair.\""],
               ["No field images in training", "\"That was on purpose, to measure the gap; the obvious next step is to "
                "add some.\""],
               ["Grad-CAM and leaf focus are approximate", "\"We use them as evidence, not proof, and cross-check with "
                "the background experiment.\""],
               ["Confidence is not calibrated", "\"Rejection reduces wrong answers but cannot remove them; calibration "
                "would be future work.\""],
               ["We did not try: other architectures, hyper-parameter search, feature extraction, domain adaptation",
                "\"We focused on a clean comparison within a free-GPU budget.\""]], [0.42, 0.58])

    doc.h1("7. 20 cross-questions (either of you)")
    for i, (q, a) in enumerate(CROSS_QUESTIONS, 1):
        doc.qa(i, q, a)

    doc.h1("8. Glossary")
    doc.table([["Term", "Meaning in one sentence"]] + [[t, d] for t, d in GLOSSARY], [0.27, 0.73], font_size=8.4)

    add_double_check(doc, F, "all")
    return doc.build("shared_guide.pdf")


# ---------------------------------------------------------------------------
# 4e. mock_viva_question_bank.pdf
# ---------------------------------------------------------------------------
BANK = [
    ("A", "easy", "What does `nn.Conv2d(3, 32, kernel_size=3, padding=1)` do?",
     "Learns 32 filters of size 3×3 over the 3 colour channels; padding 1 keeps 224×224. Output 32×224×224."),
    ("A", "easy", "What are the output shapes after each conv_block?",
     "32×112×112, 64×56×56, 128×28×28, 256×14×14 (each MaxPool halves height and width)."),
    ("A", "easy", "How many parameters does SimpleCNN have?", "{cnn_params:,} (about {cnn_params_m:.2f}M)."),
    ("A", "medium", "Why Dropout {cnn_dropout} only before the last layer?",
     "The conv layers already have BatchNorm and few parameters; the classifier is where memorising is easiest. "
     "{cnn_dropout} is a moderate value."),
    ("A", "medium", "Your fit function: does it do early stopping?",
     "No. It runs all epochs and saves the checkpoint whenever validation accuracy improves, so we keep the best "
     "epoch, but training never stops early."),
    ("A", "medium", "What is in the 'other' column of the PlantDoc confusion matrix?",
     "Predictions of PlantVillage classes that PlantDoc does not have; they are always wrong."),
    ("A", "medium", "How is a random background made?",
     "A random 4×4 colour grid is enlarged to the image size (bicubic) and Gaussian noise (std 20) is added; "
     "a new one for every training image load (random_background in data.py)."),
    ("A", "medium", "Why does validation use the same background as training in the background runs?",
     "Validation chooses the best checkpoint; it must look like the training task. The test conditions then vary "
     "the background on purpose."),
    ("A", "hard", "Why are the random test backgrounds fixed with a seed?",
     "So every model is tested on exactly the same images; otherwise differences could come from luck in the backgrounds."),
    ("A", "hard", "What does a leaf-focus score equal to the mean leaf area tell you?",
     "The heat is spread as if the model did not care where the leaf is: no evidence of leaf focus."),
    ("A", "hard", "Why ReLU in Grad-CAM?",
     "Negative values are regions that lower the class score (evidence for other classes); we only want the positive evidence."),
    ("A", "hard", "Why can't you compute the leaf-focus score on PlantDoc?",
     "PlantDoc has no leaf masks; we would need to segment the leaves ourselves."),
    ("A", "hard", "cnn_standard drops on 'leaf only' test images. Is that background bias?",
     "Partly, but careful: a black background is also new (out of distribution) for it. The random-background "
     "test and the cnn_random_bg run make the argument stronger."),
    ("B", "easy", "Which weights did you load?", "torchvision `EfficientNet_B0_Weights.IMAGENET1K_V1`."),
    ("B", "easy", "Which layer did you replace?", "`model.classifier[1]`: Linear(1280, 1000) -> Linear(1280, 38)."),
    ("B", "easy", "How many parameters are trainable?", "All {eff_trainable:,}: no layer is frozen."),
    ("B", "medium", "Why fine-tune all layers?",
     "Leaf images differ a lot from ImageNet, we have several thousand images, and the small LR ({eff_lr:g}) protects the "
     "pretrained features. (We did not run the frozen variant to compare.)"),
    ("B", "medium", "What is the LR in the last epoch of EfficientNet training?",
     "About {eff_last_lr:.1e} (cosine schedule, T_max = {eff_epochs}, stepped per epoch; see the LR table)."),
    ("B", "medium", "List the strong augmentations with their parameters.",
     "RandomResizedCrop(224, scale 0.5-1), H/V flips, RandomRotation(30), ColorJitter(0.4, 0.4, 0.4, 0.05), "
     "GaussianBlur(9, sigma 0.1-3) with p=0.3."),
    ("B", "medium", "Why is augmentation not applied to validation and test images?",
     "We want to measure performance on real images as they are, and the measurement must be repeatable."),
    ("B", "medium", "How is the domain gap computed?",
     "Accuracy on PlantVillage test restricted to the {n_overlap} shared classes, minus accuracy on PlantDoc (same {n_overlap} classes)."),
    ("B", "hard", "What does keep_correct=0.95 mean?",
     "The threshold is the 5th percentile of the confidence of correct validation predictions: 95% of correct lab "
     "answers stay above it."),
    ("B", "hard", "Coverage on PlantDoc is much lower than on PlantVillage. Good or bad?",
     "Expected and useful: the model is less sure on unfamiliar images and refuses more. Bad only if the accuracy on "
     "the answered images does not improve."),
    ("B", "hard", "Why not use a fixed threshold like 0.9?",
     "0.9 is arbitrary. Our rule is tied to how confident the model is when it is right on validation data, and it "
     "is chosen without seeing the test sets."),
    ("B", "hard", "What is the risk of over-confident softmax outputs?",
     "Out-of-domain images can get high confidence and pass the threshold; calibration (e.g. temperature scaling) "
     "or OOD detection would be needed."),
    ("B", "hard", "Why use EfficientNet-B0's last conv block for Grad-CAM?",
     "It is the last layer with spatial information (1280×7×7) and the most class-specific features."),
    ("both", "easy", "What is the input tensor shape?", "3×224×224 per image, {batch}×3×224×224 per batch."),
    ("both", "easy", "How many classes, and how many are shared with PlantDoc?", "38 classes; {n_overlap} shared."),
    ("both", "easy", "What are the split sizes?", "{train:,} train, {val:,} val, {test:,} test (PlantVillage); PlantDoc {pd_total:,}, test only."),
    ("both", "medium", "How did you avoid leakage?",
     "Leaf-grouped train/val split, the official leaf-grouped test split, PlantDoc never used in training, and the "
     "epoch and threshold chosen on validation."),
    ("both", "medium", "Why macro-F1 in addition to accuracy?", "Every class counts equally, so weak small classes are visible."),
    ("both", "medium", "What does set_seed do?",
     "Seeds Python, NumPy and PyTorch (CPU and GPU) and makes cuDNN deterministic, so runs repeat."),
    ("both", "medium", "Why mixed precision?", "float16 on the T4 is about 2× faster; GradScaler prevents tiny gradients from becoming zero."),
    ("both", "medium", "Where are checkpoints saved on Colab, and why?",
     "In MyDrive/plant_disease/checkpoints, so they survive when the free runtime disconnects."),
    ("both", "hard", "Name two weaknesses of your evaluation.",
     "Small per-class test sets and a single run per experiment (no variance); also the squashed PlantDoc images and "
     "the {shared_tt} leaf groups shared between train and test."),
    ("both", "hard", "What is novel in your work?",
     "Not the model: the systematic check of lab-to-field robustness (domain gap, background swaps, leaf focus, rejection)."),
    ("both", "hard", "If you had labelled field data, what would you change first?",
     "Fine-tune on part of it, keep a separate field validation set for the threshold and the epoch, and test on a "
     "held-out field set."),
    ("both", "hard", "How does the app decide to say 'NOT SURE'?",
     "If the softmax confidence of the top class is below the threshold from results/rejection/<model>.json "
     "(default 0.5 if missing)."),
    ("both", "hard", "Why compare models with the same seed, data and epochs per model type?",
     "So that the only difference is the factor we study (pretraining, augmentation or background)."),
    ("both", "hard", "What would show that strong augmentation really helped?",
     "A higher PlantDoc accuracy and a smaller domain gap than the standard run of the same model; ideally repeated "
     "with several seeds."),
]


def build_bank(F):
    doc = Doc("Mock viva question bank", "40 questions tagged by person (A / B / both) and difficulty, with short "
              "model answers. Numbers refer to the current code and data.")
    doc.h1("How to use this bank")
    doc.bullets([
        "Cover the answers, read the question aloud, answer in 30-60 seconds, then compare.",
        "Practise your own questions first, then 'both', then the other person's.",
        "**Mock viva with Claude:** in the Claude chat, type \"run the mock viva for Person A\" (or Person B, or "
        "both). Claude acts as the examiner: one question at a time from this bank plus sensible follow-ups. It waits "
        "for your answer, says what was correct, what was missing and what the ideal answer is, then waits for you "
        "before the next question.",
    ])
    counts = {}
    for who, lvl, _, _ in BANK:
        counts[(who, lvl)] = counts.get((who, lvl), 0) + 1
    doc.table([["", "easy", "medium", "hard", "total"]] +
              [[w] + [str(counts.get((w, l), 0)) for l in ("easy", "medium", "hard")] +
               [str(sum(counts.get((w, l), 0) for l in ("easy", "medium", "hard")))] for w in ("A", "B", "both")],
              [0.2, 0.2, 0.2, 0.2, 0.2])
    for who, title in [("A", "Person A"), ("B", "Person B"), ("both", "Both")]:
        doc.h1(title)
        for i, (w, lvl, q, a) in enumerate(BANK, 1):
            if w == who:
                doc.qa(i, q, a, tag=f"{w} · {lvl}")
    return doc.build("mock_viva_question_bank.pdf")


# ===========================================================================
# 5. Main
# ===========================================================================
F_GLOBAL = {}


def main():
    global REAL
    os.makedirs(OUT_DIR, exist_ok=True)
    REAL = results_are_real()
    print("results from a full run:", REAL, "| figures in", FIG_DIR)
    F = collect_facts()
    F_GLOBAL.update(F)
    cfg, st = F["cfg"], F["stats"] or {}
    split = st.get("our_split", {})
    VALS.update(cnn_params=F["cnn_params"], cnn_params_m=F["cnn_params"] / 1e6, cnn_dropout=F["cnn_dropout"],
                eff_params=F["eff_params"], eff_params_m=F["eff_params"] / 1e6, eff_trainable=F["eff_trainable"],
                eff_frozen=F["eff_trainable_frozen"], eff_dropout=F["eff_dropout"], eff_last_lr=F["lr_eff"][-1],
                cnn_lr=cfg["CNN_LR"], eff_lr=cfg["EFFNET_LR"], cnn_epochs=cfg["CNN_EPOCHS"],
                eff_epochs=cfg["EFFNET_EPOCHS"], batch=cfg["BATCH_SIZE"], wd=F["weight_decay"],
                mask_threshold=F["mask_threshold"], leaf_focus_n=cfg["LEAF_FOCUS_IMAGES"],
                test_per_class=cfg["TEST_PER_CLASS"], n_overlap=F["n_overlap"], n_not=len(F["not_in_plantdoc"]),
                train=split.get("train", 0), val=split.get("val", 0), test=split.get("test", 0),
                pd_total=st.get("plantdoc", {}).get("total_images", 0),
                shared_tt=split.get("shared_leaf_groups", {}).get("train_test", "?"))
    figs = {"pipeline": fig_pipeline(F), "cnn": fig_cnn(F), "gradcam": fig_gradcam(),
            "transfer": fig_transfer(F), "rejection": fig_rejection(F), "split": fig_split(F),
            "tensor": fig_tensor(F), "class_bars": fig_class_bars(F), "split_bars": fig_split_bars(F),
            "backgrounds": fig_backgrounds(), "augment": fig_augment(), "lab_vs_field": fig_lab_vs_field()}
    build_dataset_overview(F, figs)
    build_person_a(F, figs)
    build_person_b(F, figs)
    build_shared(F, figs)
    build_bank(F)
    # The folder must contain PDFs only
    for f in os.listdir(OUT_DIR):
        if not f.endswith(".pdf"):
            os.remove(os.path.join(OUT_DIR, f))


if __name__ == "__main__":
    main()

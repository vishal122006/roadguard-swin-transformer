
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = APP_DIR / "configs" / "roadguard_swin_infer.py"
DEFAULT_CHECKPOINT = APP_DIR / "weights" / "best_coco_bbox_mAP_epoch_5.pth"
DEFAULT_CHECKPOINT_URL = os.getenv(
    "ROADGUARD_SWIN_CHECKPOINT_URL",
    "https://github.com/vishal122006/roadguard-swin-transformer/releases/download/swin-epoch5/best_coco_bbox_mAP_epoch_5.pth",
).strip()
DEFAULT_CACHE_DIR = Path(
    os.getenv("ROADGUARD_SWIN_CACHE_DIR", str(Path.home() / ".cache" / "roadguard_swin"))
).expanduser()
CLASS_NAMES = [
    "D00_Longitudinal_Crack",
    "D10_Transverse_Crack",
    "D20_Alligator_Crack",
    "D40_Pothole",
]
DISPLAY_NAMES = {
    "D00_Longitudinal_Crack": "Longitudinal crack",
    "D10_Transverse_Crack": "Transverse crack",
    "D20_Alligator_Crack": "Alligator crack",
    "D40_Pothole": "Pothole",
}
CLASS_COLORS = {
    0: (255, 113, 136),
    1: (102, 217, 232),
    2: (199, 243, 107),
    3: (255, 184, 107),
}

st.set_page_config(
    page_title="RoadGuard Swin | Road damage intelligence",
    page_icon="🛣️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--rg-bg:#0a0e0d;--rg-panel:#111815;--rg-panel2:#151e18;--rg-line:#29372e;--rg-text:#f1f5ef;--rg-muted:#9ba99f;--rg-lime:#c7f36b;--rg-teal:#66d9e8;--rg-orange:#ffb86b;--rg-red:#ff7188}
html,body,[class*="css"]{font-family:'Manrope',system-ui,sans-serif}
.stApp{background:radial-gradient(ellipse at 84% 0%,rgba(199,243,107,.07),transparent 28%),var(--rg-bg);color:var(--rg-text)}
[data-testid="stAppViewContainer"]{background:transparent}
[data-testid="stHeader"]{background:rgba(10,14,13,.7)}
.block-container{max-width:1500px;padding-top:1.7rem;padding-bottom:3rem}
section[data-testid="stSidebar"]{background:#0d1310;border-right:1px solid var(--rg-line)}
section[data-testid="stSidebar"] *{color:var(--rg-text)}
h1,h2,h3,h4{color:var(--rg-text)!important;letter-spacing:-.045em!important}
p,label,[data-testid="stMarkdownContainer"]{color:var(--rg-text)}
[data-testid="stMarkdownContainer"] a{color:var(--rg-lime)}
[data-testid="stFileUploaderDropzone"]{background:#101714;border:1px dashed #536b54;border-radius:16px}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--rg-lime);background:#151e17}
.stButton>button,.stDownloadButton>button{background:var(--rg-lime);color:#10170d;border:0;font-weight:800;border-radius:10px;min-height:2.7rem}
.stButton>button:hover,.stDownloadButton>button:hover{color:#10170d;filter:brightness(1.06);border:0}
.stSlider [role="slider"]{background:var(--rg-lime);border-color:var(--rg-lime)}
[data-testid="stMetric"]{background:linear-gradient(145deg,#151e18,#101512);border:1px solid var(--rg-line);border-radius:15px;padding:15px 17px}
[data-testid="stMetricLabel"]{color:var(--rg-muted)!important;font-size:.72rem!important;text-transform:uppercase;letter-spacing:.09em}
[data-testid="stMetricValue"]{color:var(--rg-text)!important;font-weight:800;letter-spacing:-.05em}
[data-testid="stDataFrame"]{border:1px solid var(--rg-line);border-radius:12px;overflow:hidden}
[data-testid="stExpander"]{border:1px solid var(--rg-line);background:#101613;border-radius:12px}
hr{border-color:var(--rg-line)!important}
.rg-hero{position:relative;overflow:hidden;border:1px solid #304033;border-radius:24px;padding:30px 32px 27px;background:linear-gradient(112deg,rgba(199,243,107,.085),transparent 47%),linear-gradient(145deg,#151e18,#101513 75%);margin-bottom:22px}
.rg-hero:after{content:'RG / SWIN';position:absolute;right:25px;bottom:4px;font:500 clamp(3.5rem,8vw,8rem) 'DM Mono',monospace;letter-spacing:-.12em;color:rgba(199,243,107,.05);pointer-events:none}
.rg-eyebrow,.rg-section{font:500 .68rem 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.17em;color:var(--rg-lime)}
.rg-hero h1{font-size:clamp(2.2rem,5vw,4.4rem);line-height:.98;margin:.7rem 0 .8rem;max-width:900px;font-weight:800}
.rg-hero p{color:#a8b5ab;max-width:790px;line-height:1.65;margin:0;font-size:.96rem}
.rg-chip{display:inline-flex;align-items:center;gap:8px;border:1px solid #465c3c;background:#182219;color:var(--rg-lime);border-radius:999px;padding:7px 11px;font:500 .66rem 'DM Mono',monospace;letter-spacing:.08em;text-transform:uppercase}
.rg-dot{width:7px;height:7px;border-radius:50%;background:var(--rg-lime);display:inline-block;box-shadow:0 0 12px #c7f36b80}
.rg-note{border:1px solid #54442d;border-radius:12px;padding:12px 15px;background:#1c1913;color:#f3d8a9;line-height:1.55;font-size:.82rem}
.rg-footer{border-top:1px solid var(--rg-line);margin-top:30px;padding-top:14px;color:#78857b;font:500 .64rem 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.1em}
@media(max-width:800px){.rg-hero{padding:23px 20px}.rg-hero:after{display:none}}
</style>
""", unsafe_allow_html=True)


def load_report() -> dict:
    path = APP_DIR / "results" / "final_summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_crack_eval_report() -> dict:
    path = APP_DIR / "results" / "crack_only_validation.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def percent(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "Not recorded"


def resolve_config_checkpoint() -> tuple[Path, Path]:
    config_value = os.getenv("ROADGUARD_SWIN_CONFIG", "").strip()
    checkpoint_value = os.getenv("ROADGUARD_SWIN_CHECKPOINT", "").strip()
    config = Path(config_value).expanduser() if config_value else DEFAULT_CONFIG
    checkpoint = Path(checkpoint_value).expanduser() if checkpoint_value else DEFAULT_CHECKPOINT
    return config.resolve(), checkpoint.resolve()


def ensure_checkpoint_available(checkpoint_path: Path) -> Path:
    """Use local weights when present; otherwise download the public release asset."""
    minimum_bytes = 10 * 1024 * 1024
    if checkpoint_path.is_file():
        if checkpoint_path.stat().st_size >= minimum_bytes:
            return checkpoint_path
        raise ValueError(f"Checkpoint file is too small to be a valid trained model: {checkpoint_path}")

    cache_path = DEFAULT_CACHE_DIR / "best_coco_bbox_mAP_epoch_5.pth"
    if cache_path.is_file() and cache_path.stat().st_size >= minimum_bytes:
        return cache_path

    url = os.getenv("ROADGUARD_SWIN_CHECKPOINT_URL", DEFAULT_CHECKPOINT_URL).strip()
    if not url:
        raise FileNotFoundError("No local Swin checkpoint and no checkpoint download URL configured.")

    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = cache_path.with_suffix(cache_path.suffix + ".part")
    progress = st.progress(0, text="Downloading the trained Swin checkpoint…")
    downloaded = 0
    expected_size = None
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "RoadGuard-Swin-Streamlit/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            expected_size = int(content_length) if content_length else None
            with partial_path.open("wb") as handle:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if expected_size:
                        progress.progress(
                            min(1.0, downloaded / expected_size),
                            text=f"Downloading trained checkpoint… {downloaded / 1024**2:.0f} MB",
                        )
        if expected_size is not None and downloaded != expected_size:
            raise IOError(f"Checkpoint download incomplete: {downloaded} of {expected_size} bytes.")
        if downloaded < minimum_bytes:
            raise IOError(
                "The downloaded file was too small to be model weights. "
                "The release asset may not have been uploaded yet."
            )
        os.replace(partial_path, cache_path)
        progress.progress(1.0, text="Checkpoint downloaded.")
        progress.empty()
        return cache_path
    except urllib.error.HTTPError as exc:
        partial_path.unlink(missing_ok=True)
        progress.empty()
        if exc.code == 404:
            raise FileNotFoundError(
                "The epoch-5 checkpoint release asset is not published yet. "
                "Run scripts/publish_checkpoint_release_colab.py from Google Colab "
                "to upload the saved checkpoint to the public release."
            ) from exc
        raise RuntimeError(f"Checkpoint download failed with HTTP {exc.code}.") from exc
    except Exception:
        partial_path.unlink(missing_ok=True)
        progress.empty()
        raise


@st.cache_resource(show_spinner=False)
def load_model(config_path: str, checkpoint_path: str):
    import torch
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(config_path, checkpoint_path, device=device)
    return model, device


def load_font(size: int = 27):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_comparison_canvas(image: Image.Image, boxes: list[dict] | None = None) -> Image.Image:
    canvas_w, canvas_h = 1280, 720
    image = image.convert("RGB")
    scale = min(canvas_w / image.width, canvas_h / image.height)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    offset_x = (canvas_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 22, 19))
    canvas.paste(resized, (offset_x, offset_y))

    if boxes is None:
        return canvas

    draw = ImageDraw.Draw(canvas)
    font = load_font()
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        sx1 = round(x1 * scale) + offset_x
        sy1 = round(y1 * scale) + offset_y
        sx2 = round(x2 * scale) + offset_x
        sy2 = round(y2 * scale) + offset_y
        color = CLASS_COLORS.get(item["label_id"], (199, 243, 107))
        draw.rectangle((sx1, sy1, sx2, sy2), outline=color, width=5)
        label = f'{item["class_name"]} · {item["score"]:.0%}'
        bounds = draw.textbbox((0, 0), label, font=font)
        tw = bounds[2] - bounds[0]
        th = bounds[3] - bounds[1]
        if tw > canvas_w - 16:
            label = f'{item["class_name"][:20]} · {item["score"]:.0%}'
            bounds = draw.textbbox((0, 0), label, font=font)
            tw = bounds[2] - bounds[0]
            th = bounds[3] - bounds[1]
        tx = min(max(4, sx1), canvas_w - tw - 12)
        ty = sy1 - th - 12 if sy1 - th - 12 >= 0 else min(sy1 + 3, canvas_h - th - 12)
        draw.rectangle((tx, ty, tx + tw + 10, ty + th + 10), fill=(12, 18, 14))
        draw.text((tx + 5, ty + 4), label, fill=color, font=font)
    return canvas


def infer(model, image: Image.Image, threshold: float) -> list[dict]:
    from mmdet.apis import inference_detector

    # MMDetection's ndarray inference path expects BGR image pixels.
    bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    result = inference_detector(model, bgr)
    instances = result.pred_instances
    bboxes = instances.bboxes.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy()
    labels = instances.labels.detach().cpu().numpy()

    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        confidence = float(score)
        class_id = int(label)
        if confidence < threshold or not 0 <= class_id < len(CLASS_NAMES):
            continue
        class_name = CLASS_NAMES[class_id]
        detections.append({
            "label_id": class_id,
            "class_name": DISPLAY_NAMES[class_name],
            "class_code": class_name,
            "score": confidence,
            "bbox": [float(x) for x in bbox.tolist()],
        })
    detections.sort(key=lambda x: x["score"], reverse=True)
    return detections


def csv_bytes(rows: list[dict]) -> bytes:
    output = io.StringIO()
    fields = ["class_code", "class_name", "score", "x1", "y1", "x2", "y2"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        x1, y1, x2, y2 = row["bbox"]
        writer.writerow({
            "class_code": row["class_code"],
            "class_name": row["class_name"],
            "score": f'{row["score"]:.6f}',
            "x1": round(x1, 2), "y1": round(y1, 2),
            "x2": round(x2, 2), "y2": round(y2, 2),
        })
    return output.getvalue().encode("utf-8")


def render_report(report: dict):
    st.markdown(
        '<div class="rg-section">02 / Current model accuracy & performance</div>',
        unsafe_allow_html=True,
    )
    test = report.get("test_set", {})
    coco = report.get("validation_coco_metrics_epoch_5", {})
    measured_map = coco.get("mAP@0.50:0.95")
    target_map = 0.70  # Project goal; never presented as a measured result.
    primary, explanation, goal = st.columns([1.05, 2.2, 0.9], gap="medium")
    with primary:
        primary.metric(
            "Detection accuracy proxy · Validation COCO mAP@0.50:0.95",
            percent(measured_map),
            help=(
                "For object detection, COCO mean Average Precision (mAP) is a "
                "standard overall metric. There is no single universally "
                "meaningful plain accuracy percentage for this task."
            ),
        )
    with explanation:
        st.markdown("**Current baseline: Swin-Tiny + FPN + Faster R-CNN, epoch 5**")
        st.write(
            "The latest measured validation COCO mAP@0.50:0.95 is "
            f"**{percent(measured_map)}**. This is the existing baseline checkpoint; "
            "the fine-tuned model has not yet been evaluated and no improvement is being claimed."
        )
        st.caption(
            "These are dataset-level results—not confidence or accuracy for "
            "the individual photo you upload."
        )
    with goal:
        st.metric(
            "Target mAP goal",
            "70%",
            help="Project target only. This is not an achieved or guaranteed result.",
        )
        st.caption("Project goal — not yet achieved")
        try:
            measured_fraction = float(measured_map)
            progress = max(0.0, min(measured_fraction / target_map, 1.0))
            st.progress(
                progress,
                text=f"{progress * 100:.1f}% of target value",
            )
        except (TypeError, ValueError):
            st.caption("Progress will appear when a measured validation mAP is recorded.")

    metrics = st.columns(4, gap="small")
    cards = [
        ("Validation mAP@0.50", percent(coco.get("mAP@0.50"))),
        ("Test precision*", percent(test.get("precision"))),
        ("Test recall*", percent(test.get("recall"))),
        ("Test F1*", percent(test.get("f1"))),
    ]
    for col, (label, value) in zip(metrics, cards):
        col.metric(label, value)
    st.caption(
        "*Previously recorded results for the epoch-5 baseline. The test split "
        "was inspected earlier during development, so these are not results from "
        "a completely untouched test set. Precision/recall/F1 use greedy "
        "one-to-one matching per class at IoU 0.50."
    )

    st.markdown("**Unified crack-only evaluation**")
    crack_report = load_crack_eval_report()
    if crack_report and crack_report.get("threshold_sweep"):
        threshold = crack_report.get("best_crack_f1_threshold_among_tested_thresholds")
        selected = min(
            crack_report["threshold_sweep"],
            key=lambda row: abs(float(row.get("threshold", 0)) - float(threshold)),
        )
        crack_metrics = selected.get("crack_object_level", {})
        image_metrics = selected.get("crack_image_level", {})
        ap = crack_report.get("ap_at_iou_threshold", {}).get("Crack")
        cols = st.columns(5, gap="small")
        cols[0].metric("Crack AP@0.50", percent(ap))
        cols[1].metric("Crack precision", percent(crack_metrics.get("precision")))
        cols[2].metric("Crack recall", percent(crack_metrics.get("recall")))
        cols[3].metric("Crack F1", percent(crack_metrics.get("f1")))
        cols[4].metric("Image-level crack accuracy", percent(image_metrics.get("accuracy")))
        st.caption(
            f"Validation split · confidence threshold {float(threshold):.2f} · "
            f"IoU {float(crack_report.get('iou_threshold', 0.5)):.2f}. "
            "AP is 101-point interpolated AP at IoU 0.50, not COCO mAP@[.50:.95]."
        )
        st.caption(
            "Object-level metrics evaluate whether crack boxes match annotated cracks. "
            "Image-level accuracy only checks whether an image contains any crack. "
            "The file is produced by scripts/evaluate_crack_only.py."
        )
    else:
        st.info(
            "A unified crack-only score has not been measured yet. Run "
            "scripts/evaluate_crack_only.py on the RDD2022 validation split, then "
            "save its JSON output as results/crack_only_validation.json to display "
            "the measured crack-specific scores here. The values above still describe "
            "the four-class baseline."
        )

    with st.expander("View per-class test performance and validation threshold sweep"):
        per_class = report.get("per_class_test", [])
        if per_class:
            st.dataframe([
                {
                    "Damage class": DISPLAY_NAMES.get(row["class"], row["class"]),
                    "Precision": percent(row.get("precision")),
                    "Recall": percent(row.get("recall")),
                    "F1": percent(row.get("f1")),
                }
                for row in per_class
            ], use_container_width=True, hide_index=True)
        sweep = report.get("validation_threshold_sweep", [])
        if sweep:
            st.markdown("**Validation threshold sweep**")
            st.dataframe([
                {
                    "Confidence threshold": row.get("threshold"),
                    "Precision": percent(row.get("precision")),
                    "Recall": percent(row.get("recall")),
                    "F1": percent(row.get("f1")),
                }
                for row in sweep
            ], use_container_width=True, hide_index=True)
    st.warning("Research prototype: false positives and missed damage remain. The recorded test results include a test-set inspection earlier in development and should not be described as an untouched holdout.")


st.markdown("""
<div class="rg-hero">
  <div class="rg-chip"><span class="rg-dot"></span> Swin Transformer · RDD2022</div>
  <h1>See the damage.<br>Measure the road.</h1>
  <p>The RoadGuard inspection workspace, adapted for the Swin Transformer Tiny + FPN + Faster R-CNN detector. Upload a road photograph, inspect real model-produced boxes, and export detections for research review.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.55, 1], gap="large")
with left:
    st.markdown('<div class="rg-section">01 / Inspection workspace</div>', unsafe_allow_html=True)
    st.markdown("### Road damage, made visible")
    st.write("Upload a full-resolution image and compare the original photograph with predicted damage boxes, class names, and confidence scores.")
with right:
    st.markdown('<div class="rg-section">System notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="rg-note"><b>Research prototype</b><br>A no-detection result does not mean that the road is damage-free. Verify model outputs before operational decisions.</div>', unsafe_allow_html=True)
    st.caption("This UI requires the actual trained Swin checkpoint. It does not fall back to YOLO or fabricate detections.")

st.divider()
render_report(load_report())
st.divider()

st.markdown('<div class="rg-section">03 / Detection controls</div>', unsafe_allow_html=True)
config_path, checkpoint_path = resolve_config_checkpoint()
st.caption(
    f"Configuration: {config_path.name} · Checkpoint: {checkpoint_path.name} · "
    + ("local weights available" if checkpoint_path.is_file() else "auto-downloads from the public release on first detection")
)
threshold = st.slider("Detection confidence threshold", 0.05, 0.95, 0.50, 0.05)
uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
submitted = False
image = None
image_hash = None
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.caption(f"Source image: {image.width} × {image.height} pixels. The original pixels are passed to MMDetection; its configured pipeline handles resizing.")
        if image.width < 640 or image.height < 480:
            st.warning("This image is small for fine-crack inspection. Prefer the original full-resolution camera photograph.")
        st.image(image, caption="Uploaded road photograph", use_container_width=True)
    except Exception as exc:
        st.error(f"Could not read this image: {type(exc).__name__}: {exc}")
        image = None

if uploaded is not None and image is not None:
    submitted = st.button("Run Swin road-damage detection", type="primary", use_container_width=True)

    if submitted:
        if not config_path.is_file():
            st.error(f"Model configuration is missing: {config_path}. Keep the repository config in place or set ROADGUARD_SWIN_CONFIG.")
        else:
            try:
                with st.spinner("Preparing trained Swin checkpoint…"):
                    checkpoint_path = ensure_checkpoint_available(checkpoint_path)
                with st.spinner("Loading Swin model and detecting road damage…"):
                    start = time.perf_counter()
                    model, device = load_model(str(config_path), str(checkpoint_path))
                    detections = infer(model, image, threshold)
                    elapsed = time.perf_counter() - start
                    annotated = make_comparison_canvas(image, detections)
                    original_canvas = make_comparison_canvas(image)
                    out_io = io.BytesIO()
                    annotated.save(out_io, format="JPEG", quality=94)
                    st.session_state["roadguard_swin_result"] = {
                        "image_hash": image_hash,
                        "image_name": uploaded.name,
                        "threshold": threshold,
                        "detections": detections,
                        "annotated": annotated,
                        "original_canvas": original_canvas,
                        "elapsed": elapsed,
                        "device": device,
                        "config": str(config_path),
                        "checkpoint": str(checkpoint_path),
                        "annotated_jpg": out_io.getvalue(),
                    }
            except Exception as exc:
                st.error(f"Swin inference failed: {type(exc).__name__}: {exc}")
                st.info("Check that PyTorch, MMCV, MMEngine, and MMDetection versions match the environment used for training. No fallback predictions were substituted.")

saved = st.session_state.get("roadguard_swin_result")
if saved and image_hash and saved.get("image_hash") == image_hash:
    if abs(saved["threshold"] - threshold) > 1e-8:
        st.info(f"Showing the previous result at confidence {saved['threshold']:.2f}. Click Run Swin road-damage detection to apply the new threshold.")
    st.markdown('<div class="rg-section">04 / Visual comparison</div>', unsafe_allow_html=True)
    col_in, col_out = st.columns(2, gap="medium")
    with col_in:
        st.markdown("**Original input**")
        st.image(saved["original_canvas"], caption="Aspect-ratio preserved · 1280 × 720 comparison canvas", use_container_width=True)
        st.download_button("Download original-size image", data=uploaded.getvalue(), file_name=saved["image_name"], mime=uploaded.type or "image/jpeg", key="swin-download-original")
    with col_out:
        st.markdown("**Swin detector output**")
        st.image(saved["annotated"], caption="Model-produced bounding boxes · aspect ratio preserved", use_container_width=True)
        st.download_button("Download annotated output", data=saved["annotated_jpg"], file_name=f"roadguard_swin_{Path(saved['image_name']).stem}.jpg", mime="image/jpeg", key="swin-download-annotated")
    st.caption(f"Model device: {saved['device']} · Inference and model-load time on first run: {saved['elapsed']:.2f} s · Threshold: {saved['threshold']:.2f}")
    st.markdown('<div class="rg-section">05 / Detection summary</div>', unsafe_allow_html=True)
    count_col, top_col, threshold_col = st.columns(3)
    count_col.metric("Detected objects", len(saved["detections"]))
    if saved["detections"]:
        top_col.metric("Highest confidence", f'{saved["detections"][0]["score"]:.1%}')
    else:
        top_col.metric("Highest confidence", "—")
    threshold_col.metric("Applied threshold", f'{saved["threshold"]:.2f}')
    if saved["detections"]:
        rows = []
        for item in saved["detections"]:
            x1, y1, x2, y2 = item["bbox"]
            rows.append({
                "Damage class": item["class_name"],
                "Class code": item["class_code"],
                "Confidence": f'{item["score"]:.1%}',
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button("Export detections as CSV", data=csv_bytes(saved["detections"]), file_name=f"roadguard_swin_{Path(saved['image_name']).stem}.csv", mime="text/csv")
    else:
        st.warning("No boxes were above the selected confidence threshold. This does not prove the road is damage-free.")

st.markdown('<div class="rg-footer">ROADGUARD SWIN · COMPUTER VISION RESEARCH PROTOTYPE · VERIFY BEFORE OPERATIONAL USE</div>', unsafe_allow_html=True)
st.caption("Swin Transformer Tiny + FPN + Faster R-CNN trained on RDD2022. See the experiment report for measured metrics and known limitations.")

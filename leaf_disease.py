import io
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="GreenGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
MODEL_ENV = "GREEN_GUARD_UNET_PATH"
DEFAULT_UNET_PATHS = [
    ROOT / "unet80.h5",
    Path.home() / "Downloads" / "unet80.h5",
]
CNN_MODEL_PATH = ROOT / "Potato_Disease_Detection_Model2.h5"
SEGMENTATION_SIZE = 256
CLASSIFICATION_SIZE = 512


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7f1;
            --panel: rgba(255, 255, 255, 0.82);
            --panel-strong: #ffffff;
            --ink: #163026;
            --muted: #5c7469;
            --line: rgba(22, 48, 38, 0.1);
            --green: #2f7d4d;
            --green-soft: #d8f0de;
            --amber: #cf8a25;
            --red: #bd4b42;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(116, 182, 131, 0.18), transparent 34%),
                radial-gradient(circle at top right, rgba(244, 203, 118, 0.20), transparent 25%),
                linear-gradient(180deg, #f8fbf5 0%, #eef5ec 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }

        .hero {
            background: linear-gradient(135deg, rgba(22, 48, 38, 0.96), rgba(47, 125, 77, 0.93));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 28px;
            padding: 2rem 2.2rem;
            color: white;
            box-shadow: 0 24px 70px rgba(25, 60, 35, 0.18);
            overflow: hidden;
            position: relative;
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 85% 18%, rgba(255,255,255,0.18), transparent 18%),
                radial-gradient(circle at 18% 82%, rgba(255,255,255,0.08), transparent 20%);
            pointer-events: none;
        }

        .hero h1 {
            font-size: 2.8rem;
            line-height: 1.05;
            margin-bottom: 0.8rem;
            letter-spacing: -0.03em;
        }

        .hero p {
            color: rgba(255,255,255,0.82);
            font-size: 1.02rem;
            max-width: 760px;
            margin-bottom: 1.1rem;
        }

        .tag-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .tag {
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0.48rem 0.8rem;
            border-radius: 999px;
            font-size: 0.86rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 16px 40px rgba(41, 64, 49, 0.06);
            backdrop-filter: blur(10px);
        }

        .metric-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            min-height: 132px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.83rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.55rem;
            line-height: 1.45;
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.8rem;
        }

        .insight-list {
            margin: 0;
            padding-left: 1rem;
            color: var(--muted);
            line-height: 1.7;
        }

        .status-good { color: var(--green); }
        .status-medium { color: var(--amber); }
        .status-high { color: var(--red); }

        div[data-testid="stSidebar"] {
            background: rgba(251, 253, 249, 0.96);
            border-right: 1px solid rgba(22, 48, 38, 0.08);
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_unet_path() -> Path:
    env_value = os.getenv(MODEL_ENV)
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists():
            return candidate

    for path in DEFAULT_UNET_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "U-Net model not found. Place `unet80.h5` in the project folder, "
        "or set GREEN_GUARD_UNET_PATH."
    )


@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[object, object, str]:
    unet_path = find_unet_path()
    unet_model = load_model(unet_path, compile=False)
    cnn_model = load_model(CNN_MODEL_PATH, compile=False)
    return unet_model, cnn_model, str(unet_path)


def preprocess_image(image: Image.Image, size: int) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return resized


def classify_patch(roi: np.ndarray, cnn_model: object) -> Tuple[str, float]:
    roi_input = cv2.resize(roi, (CLASSIFICATION_SIZE, CLASSIFICATION_SIZE))
    roi_input = np.expand_dims(roi_input.astype("float32") / 255.0, axis=0)
    raw = float(cnn_model.predict(roi_input, verbose=0).squeeze())
    disease_probability = min(max(raw, 0.0), 1.0)

    if disease_probability >= 0.65:
        label = "Late blight risk"
    elif disease_probability >= 0.4:
        label = "Early infection signal"
    else:
        label = "Mild stress pattern"

    return label, disease_probability


def analyze_leaf(
    image: Image.Image,
    unet_model: object,
    cnn_model: object,
    threshold: float,
    min_area_ratio: float,
    expand_ratio: float,
) -> Dict[str, object]:
    source = np.array(image.convert("RGB"))
    resized = cv2.resize(source, (SEGMENTATION_SIZE, SEGMENTATION_SIZE), interpolation=cv2.INTER_AREA)
    model_input = np.expand_dims(resized.astype("float32") / 255.0, axis=0)

    predicted_mask = unet_model.predict(model_input, verbose=0)[0, :, :, 0]
    binary_mask = (predicted_mask > threshold).astype(np.uint8)
    mask_uint8 = (binary_mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = resized.copy()

    regions: List[Dict[str, object]] = []
    total_box_area = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / float(SEGMENTATION_SIZE * SEGMENTATION_SIZE)
        if area_ratio < min_area_ratio:
            continue

        dx, dy = int(w * expand_ratio), int(h * expand_ratio)
        x1, y1 = max(x - dx, 0), max(y - dy, 0)
        x2 = min(x + w + dx, SEGMENTATION_SIZE)
        y2 = min(y + h + dy, SEGMENTATION_SIZE)

        roi = resized[y1:y2, x1:x2]
        label, disease_probability = classify_patch(roi, cnn_model)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (47, 125, 77), 2)
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (17, 35, 27),
            1,
            cv2.LINE_AA,
        )

        region_area = (x2 - x1) * (y2 - y1)
        total_box_area += region_area
        regions.append(
            {
                "roi": roi,
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "probability": disease_probability,
                "coverage_pct": 100 * region_area / float(SEGMENTATION_SIZE * SEGMENTATION_SIZE),
            }
        )

    leaf_pixels = int(binary_mask.sum())
    disease_ratio = total_box_area / float(SEGMENTATION_SIZE * SEGMENTATION_SIZE)

    if not regions:
        headline = "Healthy leaf signal"
        severity = "Minimal"
        severity_class = "status-good"
        confidence = round(float(1.0 - predicted_mask.mean()) * 100, 1)
        summary = (
            "No meaningful disease regions were isolated by the segmentation model. "
            "This sample currently looks safe for routine monitoring."
        )
    elif disease_ratio < 0.05:
        headline = "Early warning pattern"
        severity = "Low"
        severity_class = "status-good"
        confidence = round(max(r["probability"] for r in regions) * 100, 1)
        summary = (
            "Small localized regions were detected. This is a good moment for targeted inspection "
            "before symptoms spread."
        )
    elif disease_ratio < 0.15:
        headline = "Escalating infection risk"
        severity = "Medium"
        severity_class = "status-medium"
        confidence = round(max(r["probability"] for r in regions) * 100, 1)
        summary = (
            "Disease regions cover a meaningful portion of the leaf. Field follow-up and isolation "
            "checks are recommended."
        )
    else:
        headline = "Severe disease footprint"
        severity = "High"
        severity_class = "status-high"
        confidence = round(max(r["probability"] for r in regions) * 100, 1)
        summary = (
            "The infected footprint is large enough to threaten plant health. This sample should be "
            "treated as a high-priority alert."
        )

    heatmap = cv2.applyColorMap((predicted_mask * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(resized, 0.72, heatmap, 0.28, 0)

    return {
        "original": resized,
        "overlay": overlay,
        "heatmap": blended,
        "mask": mask_uint8,
        "regions": regions,
        "leaf_pixels": leaf_pixels,
        "headline": headline,
        "severity": severity,
        "severity_class": severity_class,
        "confidence": confidence,
        "summary": summary,
        "coverage_pct": round(disease_ratio * 100, 2),
        "region_count": len(regions),
        "avg_region_risk": round(
            (sum(r["probability"] for r in regions) / len(regions) * 100) if regions else 0.0,
            1,
        ),
    }


def build_report(data: Dict[str, object], source_name: str) -> bytes:
    lines = [
        "GreenGuard AI - Plant Health Report",
        f"Sample: {source_name}",
        f"Assessment: {data['headline']}",
        f"Severity: {data['severity']}",
        f"Detected regions: {data['region_count']}",
        f"Disease footprint: {data['coverage_pct']}%",
        f"Peak confidence: {data['confidence']}%",
        f"Average region risk: {data['avg_region_risk']}%",
        "",
        "Operational summary:",
        str(data["summary"]),
    ]

    if data["regions"]:
        lines.append("")
        lines.append("Region breakdown:")
        for idx, region in enumerate(data["regions"], start=1):
            lines.append(
                f"- Region {idx}: {region['label']} | "
                f"risk={round(region['probability'] * 100, 1)}% | "
                f"coverage={round(region['coverage_pct'], 2)}%"
            )

    return "\n".join(lines).encode("utf-8")


def render_hero(model_path: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <h1>GreenGuard AI</h1>
            <p>
                A sharper crop-health screening experience for demos, judging, and deployment.
                Upload a potato leaf image and get disease-region localization, severity scoring,
                and a cleaner analyst-style summary built for fast decision making.
            </p>
            <div class="tag-row">
                <span class="tag">UGNet / U-Net segmentation</span>
                <span class="tag">Patch-aware disease scoring</span>
                <span class="tag">Streamlit experience upgrade</span>
                <span class="tag">Active U-Net source: {Path(model_path).name}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_path: str) -> Tuple[float, float, float]:
    st.sidebar.markdown("## Control Panel")
    st.sidebar.markdown(
        """
        <div class="small-muted">
        Tune detection sensitivity for demos or field-style reviews. Lower thresholds catch more regions,
        while higher thresholds reduce false alarms.
        </div>
        """,
        unsafe_allow_html=True,
    )
    threshold = st.sidebar.slider("Segmentation threshold", 0.20, 0.90, 0.50, 0.05)
    min_area_ratio = st.sidebar.slider("Minimum region ratio", 0.0005, 0.0200, 0.0010, 0.0005, format="%.4f")
    expand_ratio = st.sidebar.slider("Bounding box expansion", 0.05, 0.40, 0.15, 0.01)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"U-Net path: `{model_path}`")
    st.sidebar.caption(f"CNN path: `{CNN_MODEL_PATH.name}`")
    return threshold, min_area_ratio, expand_ratio


def render_metrics(data: Dict[str, object]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        (col1, "Assessment", data["headline"], data["summary"]),
        (col2, "Severity", data["severity"], f"Disease footprint: {data['coverage_pct']}%"),
        (col3, "Peak confidence", f"{data['confidence']}%", f"Average risk: {data['avg_region_risk']}%"),
        (col4, "Detected regions", str(data["region_count"]), f"Leaf pixels in mask: {data['leaf_pixels']}"),
    ]
    for column, label, value, note in cards:
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    inject_styles()

    try:
        unet_model, cnn_model, model_path = load_models()
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.info(
            "Place `unet80.h5` in the project folder or at "
            "`C:/Users/admin/Downloads/unet80.h5`, or set `GREEN_GUARD_UNET_PATH`."
        )
        st.stop()

    render_hero(model_path)
    threshold, min_area_ratio, expand_ratio = render_sidebar(model_path)

    st.markdown("")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Run an Analysis</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a potato leaf image",
            type=["jpg", "jpeg", "png"],
            help="Best results come from bright, close-up leaf images with a visible lesion pattern.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Why this version scores better</div>
                <ul class="insight-list">
                    <li>Cleaner visual storytelling with decision-ready summaries.</li>
                    <li>Model loading now supports local-path reuse for your existing `unet80.h5`.</li>
                    <li>Binary classifier output is handled consistently instead of using a misleading two-class argmax.</li>
                    <li>Prepared for Hugging Face Spaces deployment with a dedicated root container setup.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        st.markdown("")
        showcase_a, showcase_b = st.columns(2, gap="large")
        with showcase_a:
            if (ROOT / "Disese-detection.png").exists():
                st.image(str(ROOT / "Disese-detection.png"), caption="Previous detection sample", use_container_width=True)
        with showcase_b:
            if (ROOT / "Masking.png").exists():
                st.image(str(ROOT / "Masking.png"), caption="Segmentation sample", use_container_width=True)
        return

    source_name = uploaded_file.name
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing leaf structure and disease footprint..."):
        analysis = analyze_leaf(
            image=image,
            unet_model=unet_model,
            cnn_model=cnn_model,
            threshold=threshold,
            min_area_ratio=min_area_ratio,
            expand_ratio=expand_ratio,
        )

    render_metrics(analysis)

    st.markdown("")
    result_left, result_right = st.columns([1.15, 0.85], gap="large")

    with result_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detection Studio</div>', unsafe_allow_html=True)
        tabs = st.tabs(["Original", "Overlay", "Heatmap", "Binary Mask"])
        with tabs[0]:
            st.image(analysis["original"], caption="Resized analysis frame", use_container_width=True)
        with tabs[1]:
            st.image(analysis["overlay"], caption="Localized disease regions", use_container_width=True)
        with tabs[2]:
            st.image(cv2.cvtColor(analysis["heatmap"], cv2.COLOR_BGR2RGB), caption="Segmentation confidence heatmap", use_container_width=True)
        with tabs[3]:
            st.image(analysis["mask"], caption="Predicted disease mask", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with result_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Decision Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="metric-label">Status</div>
            <div class="metric-value {analysis['severity_class']}">{analysis['headline']}</div>
            <div class="metric-note">{analysis['summary']}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        if analysis["regions"]:
            for idx, region in enumerate(analysis["regions"], start=1):
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-bottom: 0.8rem;">
                        <div class="metric-label">Region {idx}</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: var(--ink);">
                            {region['label']}
                        </div>
                        <div class="metric-note">
                            Risk: {round(region['probability'] * 100, 1)}%<br/>
                            Image footprint: {round(region['coverage_pct'], 2)}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No high-risk disease region was detected in this sample.")

        report_bytes = build_report(analysis, source_name)
        st.download_button(
            "Download analysis report",
            data=report_bytes,
            file_name=f"{Path(source_name).stem}_greenguard_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

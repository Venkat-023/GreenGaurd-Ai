import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ============================================================
#              CONFIGURATION AND MODEL LOADING
# ============================================================

st.set_page_config(page_title="Potato Leaf Disease Detector", layout="wide")

st.title("🍀 Potato Leaf Disease Detection & Classification")
st.write(
    "Upload a potato leaf image to detect and classify diseased regions "
    "using U-Net and CNN models."
)

@st.cache_resource
def load_models():
    # Load models directly from repo
    unet_model = load_model("unet80.h5", compile=False)
    cnn_model = load_model("Potato_Disease_Detection_Model2.h5", compile=False)
    return unet_model, cnn_model

unet_model, cnn_model = load_models()
class_names = ["Bacterial Blight", "Bacterial Late"]
IMG_SIZE = 256


# ============================================================
#              DETECTION + CLASSIFICATION FUNCTION
# ============================================================

def show_disease_detection(
    img,
    unet_model,
    cnn_model,
    class_names,
    threshold=0.5,
    min_area_ratio=0.001,
    expand_ratio=0.15,
    panel_column_width=250,
    spacing=20,
    IMG_SIZE=256,
):
    """
    Full integrated detection, classification, severity estimation, and visualization
    """
    # Step 1: Resize and preprocess
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    inp = np.expand_dims(img_resized / 255.0, 0)

    # Step 2: Predict mask with U-Net
    pred_mask = unet_model.predict(inp, verbose=0)[0, :, :, 0]
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255

    # Step 3: Extract contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    results = img_resized.copy()
    roi_info = []
    total_disease_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area_ratio = (w * h) / (IMG_SIZE * IMG_SIZE)
        if area_ratio < min_area_ratio:
            continue

        # Expand bounding box in all directions
        dx, dy = int(w * expand_ratio), int(h * expand_ratio)
        x1, y1 = max(x - dx, 0), max(y - dy, 0)
        x2, y2 = min(x + w + dx, IMG_SIZE), min(y + h + dy, IMG_SIZE)

        roi = img_resized[y1:y2, x1:x2]
        roi_info.append((roi, (x1, y1, x2 - x1, y2 - y1)))
        total_disease_area += (x2 - x1) * (y2 - y1)
        cv2.rectangle(results, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Step 4: If no regions -> Healthy Leaf
    if len(roi_info) == 0:
        healthy_img = img_resized.copy()
        cv2.putText(
            healthy_img,
            "Healthy Leaf 🌱",
            (40, IMG_SIZE // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 200, 0),
            3,
        )
        return healthy_img, []

    # Step 5: Classify ROIs using CNN
    final_predictions = []
    for roi, (x, y, w, h) in roi_info:
        roi_resized = cv2.resize(roi, (512, 512)) / 255.0
        pred = cnn_model.predict(np.expand_dims(roi_resized, axis=0), verbose=0)
        label = class_names[np.argmax(pred)]
        confidence = np.max(pred)
        final_predictions.append((roi, (x, y, w, h), label, confidence))

    # Step 6: Severity Estimation
    disease_fraction = total_disease_area / (IMG_SIZE * IMG_SIZE)
    n_regions = len(final_predictions)
    if disease_fraction < 0.05 and n_regions <= 2:
        severity = "Low Severity"
        color = (0, 255, 0)
    elif disease_fraction < 0.15:
        severity = "Medium Severity"
        color = (0, 255, 255)
    else:
        severity = "High Severity"
        color = (0, 0, 255)

    cv2.putText(
        results, f"Severity: {severity}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )

    # Step 7: Right Panel with Multi-columns
    panel_height = IMG_SIZE
    remaining_rois = final_predictions.copy()
    col_rois = []

    while remaining_rois:
        current_col = []
        y_offset = spacing
        for roi, _, label, _ in remaining_rois:
            roi_h, roi_w = roi.shape[:2]
            if y_offset + roi_h > panel_height:
                break
            current_col.append((roi, label, y_offset))
            y_offset += roi_h + spacing
        col_rois.append(current_col)
        remaining_rois = remaining_rois[len(current_col):]

    panel_width_total = len(col_rois) * panel_column_width
    panel = np.ones((panel_height, panel_width_total, 3), dtype=np.uint8) * 255
    panel_positions = []

    for col_idx, col in enumerate(col_rois):
        x_offset_col = col_idx * panel_column_width
        for roi, label, y_offset in col:
            roi_h, roi_w = roi.shape[:2]
            x_offset = x_offset_col + (panel_column_width - roi_w) // 2
            if y_offset + roi_h > panel_height:
                roi = roi[: panel_height - y_offset, :, :]
                roi_h = roi.shape[0]
            panel[y_offset:y_offset + roi_h, x_offset:x_offset + roi_w] = roi
            cv2.putText(
                panel,
                label,
                (x_offset, max(y_offset - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            panel_positions.append((x_offset + roi_w // 2, y_offset + roi_h // 2))

    # Step 8: Concatenate leaf and panel
    final_output = np.concatenate([results, panel], axis=1)

    # Step 9: Draw connecting lines (after concatenation)
    for (roi, (x, y, w, h), _, _), (px, py) in zip(final_predictions, panel_positions):
        leaf_center = (x + w // 2, y + h // 2)
        panel_center = (px + IMG_SIZE, py)
        cv2.line(final_output, leaf_center, panel_center, (0, 0, 255), 1)

    return final_output, final_predictions


# ============================================================
#              STREAMLIT UI LOGIC
# ============================================================

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    st.write("Analyzing leaf... please wait.")

    final_output, preds = show_disease_detection(
        img=img_array,
        unet_model=unet_model,
        cnn_model=cnn_model,
        class_names=class_names,
    )

    st.image(
        cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB),
        caption="Final Detection Result",
        use_column_width=True,
    )
    st.success(f"✅ Detection complete! {len(preds)} diseased region(s) detected.")

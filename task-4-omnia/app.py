import streamlit as st
import cv2
import pytesseract
import numpy as np
from pytesseract import Output


# ==================================================
# Tesseract Configuration
# ==================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="DecodeLabs OCR",
    page_icon="🔍",
    layout="wide"
)


# ==================================================
# Title
# ==================================================

st.title("🔍 DecodeLabs OCR Recognition")
st.write(
    "Image preprocessing and Optical Character Recognition "
    "with confidence-based validation."
)


# ==================================================
# Sidebar
# ==================================================

st.sidebar.header("⚙️ OCR Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=0,
    max_value=100,
    value=80
)

psm_mode = st.sidebar.selectbox(
    "Tesseract PSM Mode",
    [3, 6, 7, 11],
    index=1
)


# ==================================================
# Image Upload
# ==================================================

uploaded_file = st.file_uploader(
    "📁 Upload a document image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# Preprocessing
# ==================================================

def preprocess_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return gray, thresholded


# ==================================================
# OCR
# ==================================================

def perform_ocr(image, threshold):

    data = pytesseract.image_to_data(
        image,
        config=f"--psm {psm_mode}",
        output_type=Output.DICT
    )

    accepted = []
    rejected = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except ValueError:
            continue

        detection = {
            "text": text,
            "confidence": confidence,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i]
        }

        if confidence >= threshold:
            accepted.append(detection)
        else:
            rejected.append(detection)

    return accepted, rejected


# ==================================================
# Draw Bounding Boxes
# ==================================================

def draw_boxes(image, detections):

    output = image.copy()

    for detection in detections:

        x = detection["x"]
        y = detection["y"]
        w = detection["w"]
        h = detection["h"]

        text = detection["text"]
        confidence = detection["confidence"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = f"{text} ({confidence:.0f}%)"

        cv2.putText(
            output,
            label,
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return output


# ==================================================
# Main Application
# ==================================================

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Preprocess
    gray, processed = preprocess_image(image)

    # OCR
    accepted, rejected = perform_ocr(
        processed,
        confidence_threshold
    )

    # Annotated image
    annotated = draw_boxes(
        image,
        accepted
    )

    # Final text
    extracted_text = " ".join(
        item["text"]
        for item in accepted
    )

    # Average confidence
    if accepted:

        average_confidence = sum(
            item["confidence"]
            for item in accepted
        ) / len(accepted)

    else:

        average_confidence = 0


    # ==================================================
    # Metrics
    # ==================================================

    st.subheader("📊 Recognition Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Confidence",
        f"{average_confidence:.2f}%"
    )

    col2.metric(
        "Accepted Words",
        len(accepted)
    )

    col3.metric(
        "Rejected Words",
        len(rejected)
    )


    # ==================================================
    # Images
    # ==================================================

    st.subheader("🖼️ Visual Processing")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("Original Image")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

    with col2:

        st.write("Preprocessed Image")

        st.image(
            processed,
            use_container_width=True
        )

    with col3:

        st.write("OCR Detection")

        st.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )


    # ==================================================
    # Extracted Text
    # ==================================================

    st.subheader("📝 Extracted Text")

    if extracted_text:

        st.text_area(
            "Recognized Content",
            extracted_text,
            height=250
        )

    else:

        st.warning(
            "No text passed the confidence threshold."
        )


    # ==================================================
    # Confidence Details
    # ==================================================

    with st.expander("🔎 Detection Details"):

        for item in accepted:

            st.write(
                f"**{item['text']}** — "
                f"{item['confidence']:.2f}%"
            )


else:

    st.info(
        "👆 Upload a document image to start OCR."
    )

    st.markdown(
        """
        ### Pipeline

        **Image → Grayscale → Gaussian Blur → "
        "Adaptive Thresholding → Tesseract OCR → "
        "Confidence Filtering → Bounding Boxes**
        """
    )
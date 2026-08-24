import cv2
import pytesseract
import os
import json
from pytesseract import Output


# ==================================================
# Tesseract Configuration
# ==================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==================================================
# Project Configuration
# ==================================================

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

IMAGE_FILES = [
    "document_clean.jpg",
    "document_challenging.jpg"
]

CONFIDENCE_THRESHOLD = 80


# ==================================================
# Create Output Directory
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# Image Preprocessing
# ==================================================

def preprocess_image(image):
    """
    OCR preprocessing pipeline:
    1. Grayscale conversion
    2. Gaussian Blur
    3. Adaptive Thresholding
    """

    # 1. Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 2. Gaussian Blur
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # 3. Adaptive Thresholding
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresholded


# ==================================================
# OCR + Confidence Extraction
# ==================================================

def perform_ocr(processed_image):

    data = pytesseract.image_to_data(
        processed_image,
        config="--psm 6",
        output_type=Output.DICT
    )

    accepted_words = []
    rejected_words = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except ValueError:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        detection = {
            "text": text,
            "confidence": round(confidence, 2),
            "x": x,
            "y": y,
            "width": w,
            "height": h
        }

        # 80% confidence validation
        if confidence >= CONFIDENCE_THRESHOLD:
            accepted_words.append(detection)
        else:
            rejected_words.append(detection)

    return accepted_words, rejected_words


# ==================================================
# Draw Bounding Boxes
# ==================================================

def draw_detections(image, detections):

    output = image.copy()

    for detection in detections:

        x = detection["x"]
        y = detection["y"]
        w = detection["width"]
        h = detection["height"]

        text = detection["text"]
        confidence = detection["confidence"]

        # Bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Label
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
# Process Image
# ==================================================

def process_image(filename):

    print("\n" + "=" * 60)
    print(f"Processing: {filename}")
    print("=" * 60)

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    # Read image
    image = cv2.imread(input_path)

    if image is None:

        print(
            f"ERROR: Could not read image: {input_path}"
        )

        return

    # ------------------------------------------------
    # Preprocessing
    # ------------------------------------------------

    processed_image = preprocess_image(image)

    base_name = os.path.splitext(filename)[0]

    # Save preprocessed image
    processed_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_preprocessed.jpg"
    )

    cv2.imwrite(
        processed_path,
        processed_image
    )

    # ------------------------------------------------
    # OCR
    # ------------------------------------------------

    accepted, rejected = perform_ocr(
        processed_image
    )

    # ------------------------------------------------
    # Final Text
    # ------------------------------------------------

    final_text = " ".join(
        item["text"]
        for item in accepted
    )

    # ------------------------------------------------
    # Average Confidence
    # ------------------------------------------------

    if accepted:

        average_confidence = (
            sum(
                item["confidence"]
                for item in accepted
            )
            / len(accepted)
        )

    else:

        average_confidence = 0

    # ------------------------------------------------
    # Bounding Boxes
    # ------------------------------------------------

    annotated_image = draw_detections(
        image,
        accepted
    )

    annotated_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_annotated.jpg"
    )

    cv2.imwrite(
        annotated_path,
        annotated_image
    )

    # ------------------------------------------------
    # Save OCR Text
    # ------------------------------------------------

    text_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_ocr.txt"
    )

    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(final_text)

    # ------------------------------------------------
    # Save JSON Results
    # ------------------------------------------------

    results = {
        "image": filename,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "average_confidence": round(
            average_confidence,
            2
        ),
        "accepted_words": len(accepted),
        "rejected_words": len(rejected),
        "text": final_text,
        "detections": accepted
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ------------------------------------------------
    # Terminal Results
    # ------------------------------------------------

    print("\nExtracted Text:")
    print("-" * 50)
    print(final_text)
    print("-" * 50)

    print(
        f"Average Confidence: "
        f"{average_confidence:.2f}%"
    )

    print(
        f"Accepted Words: {len(accepted)}"
    )

    print(
        f"Rejected Words: {len(rejected)}"
    )

    print("\nGenerated Files:")

    print(
        f"Preprocessed: {processed_path}"
    )

    print(
        f"Annotated:    {annotated_path}"
    )

    print(
        f"Text:         {text_path}"
    )

    print(
        f"JSON:         {json_path}"
    )


# ==================================================
# Main Program
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("             DecodeLabs OCR Pipeline")
    print("=" * 60)

    for image_file in IMAGE_FILES:

        process_image(image_file)

    print("\n" + "=" * 60)
    print("          OCR PROCESSING COMPLETED")
    print("=" * 60)
🔍 DecodeLabs OCR Recognition Pipeline

📌 Project Overview

This project is part of DecodeLabs Artificial Intelligence – Project 4: Image or Text Recognition (Basic).

The goal is to build a practical Optical Character Recognition (OCR) pipeline that can take a document image, preprocess it, extract text using a pre-trained OCR engine, and validate the detected text using confidence scores.

The project demonstrates how computer vision and AI libraries can be combined to transform unstructured visual data into machine-readable text.

---

🎯 Objectives

- Read text from document images.
- Apply image preprocessing to improve OCR quality.
- Extract text using Tesseract OCR.
- Calculate confidence scores for detected words.
- Apply an 80% confidence threshold.
- Generate bounding boxes around accepted text.
- Save OCR results in TXT and JSON formats.
- Provide an interactive web demo for testing custom images.

---

🛠️ Technologies

- Python
- OpenCV
- Tesseract OCR
- pytesseract
- NumPy
- Streamlit

---

🔄 OCR Pipeline

Input Image
     ↓
Grayscale Conversion
     ↓
Gaussian Blur
     ↓
Adaptive Thresholding
     ↓
Tesseract OCR
     ↓
Confidence Score
     ↓
80% Confidence Filtering
     ↓
Bounding Boxes
     ↓
Extracted Text

---

🧹 Image Preprocessing

The system applies three main preprocessing steps:

1. Grayscale Conversion

Converts the RGB image into a single-channel grayscale image to simplify the visual information.

2. Gaussian Blur

Reduces small noise and image imperfections before OCR processing.

3. Adaptive Thresholding

Converts the grayscale image into a high-contrast binary image to improve text visibility under uneven lighting and background conditions.

---

🤖 OCR Engine

The project uses Tesseract OCR through the "pytesseract" Python library.

Tesseract is used to detect:

- Text content
- Word positions
- Confidence scores
- Bounding box coordinates

---

📊 Confidence Filtering

Each detected word receives a confidence score.

The project uses an 80% minimum confidence threshold:

if confidence >= 80:
    accept_detection()
else:
    reject_detection()

This helps reduce low-confidence OCR results and false detections.

---

📦 Project Structure

task-4-omnia/
│
├── data/
│   ├── input/
│   │   ├── document_clean.jpg
│   │   └── document_challenging.jpg
│   │
│   └── output/
│
├── src/
│   └── ocr.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

---

🚀 Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_LINK
cd task-4-omnia

Create and activate a virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install Python dependencies:

pip install -r requirements.txt

Tesseract OCR must also be installed on the system.

---

▶️ Run the OCR Pipeline

Run the main OCR script:

python src/ocr.py

The system processes the sample images from:

data/input/

and generates results inside:

data/output/

---

🌐 Interactive Demo

The project includes a Streamlit web application.

Run:

streamlit run app.py

The demo allows users to:

- Upload custom document images.
- Select the OCR confidence threshold.
- Select the Tesseract PSM mode.
- View the original image.
- View the preprocessed image.
- View OCR bounding boxes.
- View extracted text.
- View confidence statistics.

The demo supports:

JPG
JPEG
PNG

So users are not limited to the two sample images included in the repository.

---

⚙️ Tesseract PSM Modes

The demo provides several Page Segmentation Modes:

PSM| Usage
3| Automatic page segmentation
6| Single uniform block of text
7| Single text line
11| Sparse text

For the provided document images, PSM 6 is used as the default configuration.

---

📈 Sample Results

Clean Document

Average Confidence: 92.64%
Accepted Words: 109
Rejected Words: 59

Challenging Document

Average Confidence: 93.19%
Accepted Words: 79
Rejected Words: 39

The results demonstrate that the OCR pipeline achieved confidence levels above the required 80% validation threshold on the provided sample images.

---

🎥 Demo Video

Watch the project demonstration here:

Demo Link: https://youtu.be/mYx4EB6KVG4

The demonstration shows:

1. Uploading a document image.
2. OCR preprocessing.
3. Text extraction.
4. Confidence filtering.
5. Changing the confidence threshold.
6. Changing the PSM mode.
7. Testing a challenging document.

---

✅ Project Requirements

Requirement| Status
Pre-trained OCR Library| ✅
Image Recognition| ✅
Grayscale Preprocessing| ✅
Gaussian Blur| ✅
Adaptive Thresholding| ✅
OCR Text Extraction| ✅
Confidence Validation| ✅
80% Threshold| ✅
Bounding Boxes| ✅
Visual Output| ✅
Interactive Demo| ✅

---

🔮 Future Improvements

- Support for multiple languages.
- Automatic deskewing.
- Advanced noise removal.
- Better document layout analysis.
- PDF document processing.
- OCR result export to CSV.
- Cloud deployment.

---

👩‍💻 Author

Omnia Ayman

Artificial Intelligence Engineering Student

---

📜 Project

DecodeLabs – Artificial Intelligence Industrial Training Kit

Project 4: Image or Text Recognition (Basic)

---
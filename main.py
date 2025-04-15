import cv2
import pytesseract
from ultralytics import YOLO
from paddleocr import PaddleOCR
import easyocr
import csv
import numpy as np
import os
import time
from datetime import datetime

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# High-Resolution Capture
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080

# Initialize OCR engines
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
easy_ocr = easyocr.Reader(['en'])

# Load YOLO Model
model = YOLO(os.path.join('models', 'best.pt'))

# Create directories
image_dir, csv_file_path = 'images', 'number_plates.csv'
os.makedirs(image_dir, exist_ok=True)

def preprocess_image(img):
    """Enhance image for better OCR accuracy."""
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return gray

def upscale_image(img, scale=2):
    """Upscales image using INTER_CUBIC interpolation."""
    height, width = img.shape[:2]
    return cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

def perform_ocr(img):
    """Performs OCR using PaddleOCR, EasyOCR, and Tesseract as fallback."""
    img_processed = preprocess_image(img)
    
    # PaddleOCR
    paddle_results = paddle_ocr.ocr(img, cls=True)
    if paddle_results:
        return ''.join([res[1][0] for res in paddle_results[0]])
    
    # EasyOCR fallback
    easy_results = easy_ocr.readtext(img)
    if easy_results:
        return easy_results[0][1]
    
    # Tesseract fallback
    upscaled_img = upscale_image(img_processed)
    text = pytesseract.image_to_string(upscaled_img, config='--psm 6')
    return text.strip() if text else "N/A"

def detect_plate_color(img):
    """Detect the dominant color of the number plate."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for different plate colors
    colors = {
        "White": [(0, 0, 200), (180, 30, 255)],
        "Yellow": [(20, 100, 100), (30, 255, 255)],
        "Green": [(35, 100, 100), (85, 255, 255)],
        "Blue": [(90, 50, 50), (130, 255, 255)],
        "Red": [(0, 100, 100), (10, 255, 255)],
        "Black": [(0, 0, 0), (180, 255, 50)]
    }

    max_pixels = 0
    dominant_color = "Unknown"

    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
        pixels = cv2.countNonZero(mask)
        if pixels > max_pixels:
            max_pixels = pixels
            dominant_color = color

    return dominant_color

def categorize_plate(plate_color):
    """Categorize the number plate based on Indian regulations."""
    categories = {
        "White": "Private Vehicle",
        "Yellow": "Commercial Vehicle",
        "Green": "Electric Vehicle (Private/Commercial)",
        "Blue": "Diplomatic Vehicle",
        "Red": "Government Vehicle",
        "Black": "Rental/Self-Drive Vehicle"
    }
    return categories.get(plate_color, "Unknown")

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Open CSV file
csv_file = open(csv_file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Filename', 'Number Plate Text', 'Plate Color', 'Category', 'Date Time'])

start_time = time.time()
best_plate_img, best_conf = None, 0.5  # Confidence threshold

while time.time() - start_time < 10:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    results = model.predict(frame)
    
    for box in results[0].boxes:
        if box.conf[0] > best_conf:
            best_conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            best_plate_img = frame[y1:y2, x1:x2]
    
    cv2.imshow('Number Plate Detection', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Process and save the best detected plate
if best_plate_img is not None and best_plate_img.size > 0:
    filename = os.path.join(image_dir, 'plate_best.png')
    cv2.imwrite(filename, best_plate_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # Perform OCR
    text = perform_ocr(best_plate_img)
    
    # Detect Plate Color
    plate_color = detect_plate_color(best_plate_img)
    
    # Categorize Based on Plate Color
    category = categorize_plate(plate_color)

    # Get timestamp
    date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV
    csv_writer.writerow([filename, text, plate_color, category, date_time])
    csv_file.flush()
    
    print(f"Stored in CSV - Number Plate: {text}, Color: {plate_color}, Category: {category}, Timestamp: {date_time}")
else:
    print("Error: No valid plate image detected.")

csv_file.close()
cap.release()
cv2.destroyAllWindows()

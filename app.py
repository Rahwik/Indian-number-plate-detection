import base64
import cv2
import pytesseract
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR
import easyocr
import csv
import numpy as np
import os
import time
from datetime import datetime
import streamlit as st

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
API_URL = "https://f626-152-59-48-189.ngrok-free.app/api/detect-vehicle"

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
easy_ocr = easyocr.Reader(['en'])
model = YOLO(os.path.join('models', 'best.pt'))


image_dir = 'images'
csv_file_path = 'number_plates.csv'
os.makedirs(image_dir, exist_ok=True)

if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Filename', 'Number Plate Text', 'Plate Color', 'Category', 'Date Time'])

def image_to_base64(img):
    _, img_encoded = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64

def preprocess_image(img):
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 2)
    return gray

def upscale_image(img, scale=2):
    height, width = img.shape[:2]
    return cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

def perform_ocr(img):
    img_processed = preprocess_image(img)
    paddle_results = paddle_ocr.ocr(img, cls=True)
    if paddle_results:
        return ''.join([res[1][0] for res in paddle_results[0]])
    easy_results = easyocr.readtext(img)
    if easy_results:
        return easy_results[0][1]
    upscaled_img = upscale_image(img_processed)
    text = pytesseract.image_to_string(upscaled_img, config='--psm 6')
    return text.strip() if text else "N/A"

def detect_plate_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 185])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_combined = cv2.bitwise_or(mask_white, mask_black)
    mask_inverted = cv2.bitwise_not(mask_combined)
    kernel = np.ones((3,3), np.uint8)
    mask_inverted = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel, iterations=2)
    hsv_colored = cv2.bitwise_and(hsv, hsv, mask=mask_inverted)
    colors = {
        "Black": [(0, 0, 200), (180, 30, 255)],
        "Yellow": [(20, 100, 100), (30, 255, 255)],
        "Green": [(35, 50, 50), (85, 255, 255)],
        "Blue": [(90, 50, 50), (130, 255, 255)],
        "Red": [(0, 100, 100), (10, 255, 255)],
        "White": [(0, 0, 0), (180, 255, 50)]
    }
    max_pixels = 0
    dominant_color = "Unknown"
    for color, (lower, upper) in colors.items():
        mask_color = cv2.inRange(hsv_colored, np.array(lower), np.array(upper))
        pixels = cv2.countNonZero(mask_color)
        if pixels > max_pixels:
            max_pixels = pixels
            dominant_color = color
    return dominant_color

def categorize_plate(plate_color):
    categories = {
        "White": "Private Vehicle",
        "Yellow": "Commercial Vehicle",
        "Green": "Electric Vehicle (Private/Commercial)",
        "Blue": "Diplomatic Vehicle",
        "Red": "Government Vehicle",
        "Black": "Rental/Self-Drive Vehicle"
    }
    return categories.get(plate_color, "Unknown")

def send_to_api(img_base64, text, plate_color, category, date_time):
    data = {
        "vehicle_image": img_base64,
        "vehicle_number": text,
        "color": plate_color,
        "category": category,
        "timestamp": date_time
    }
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            st.success(f"Sent to API successfully: {response.json()}")
        else:
            st.error(f"API Error: {response.status_code}, Response: {response.text}")
    except Exception as e:
        st.error(f"Failed to send data to API: {str(e)}")

def detect_plate():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        st.error("Could not open camera.")
        return None

    best_plate_img, best_conf = None, 0.5
    start_time = time.time()

    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        for box in results[0].boxes:
            if box.conf[0] > best_conf:
                best_conf = box.conf[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_plate_img = frame[y1:y2, x1:x2]
        preview_img = results[0].plot()
        st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detection Preview")
    cap.release()

    if best_plate_img is not None and best_plate_img.size > 0:
        filename = os.path.join(image_dir, f'plate_{int(time.time())}.png')
        cv2.imwrite(filename, best_plate_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        text = perform_ocr(best_plate_img)
        plate_color = detect_plate_color(best_plate_img)
        category = categorize_plate(plate_color)
        date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        img_base64 = image_to_base64(best_plate_img)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([filename, text, plate_color, category, date_time])
        st.success(f"Record saved: Number Plate: {text}, Color: {plate_color}, Category: {category}")
        st.image(cv2.cvtColor(best_plate_img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detected Plate")
        send_to_api(img_base64, text, plate_color, category, date_time)
        return {
            "filename": filename,
            "number_plate_text": text,
            "plate_color": plate_color,
            "category": category,
            "date_time": date_time
        }
    else:
        st.error("Error: No valid plate image detected.")
        return None

st.title("Vehicle Number Plate Detection")

if st.button("Capture Plate"):
    with st.spinner("Capturing and processing image..."):
        record = detect_plate()
    if record:
        st.write(record)

st.markdown("Click **Capture Plate** to restart the process and store a new record.")

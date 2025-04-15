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
from flask import Flask, jsonify

app = Flask(__name__)
API_URL = "https://44b58635-e9bd-4d2e-83f5-f574643a3f4d.mock.pstmn.io/detect-plate"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
easy_ocr = easyocr.Reader(['en'])
model = YOLO(os.path.join('models', 'best.pt'))
image_dir, csv_file_path = 'images', 'number_plates.csv'
os.makedirs(image_dir, exist_ok=True)

def preprocess_image(img):
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return gray

def upscale_image(img, scale=2):
    height, width = img.shape[:2]
    return cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

def perform_ocr(img):
    img_processed = preprocess_image(img)
    paddle_results = paddle_ocr.ocr(img, cls=True)
    if paddle_results:
        return ''.join([res[1][0] for res in paddle_results[0]])
    easy_results = easy_ocr.readtext(img)
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
        "White": [(0, 0, 200), (180, 30, 255)],
        "Yellow": [(20, 100, 100), (30, 255, 255)],
        "Green": [(35, 50, 50), (85, 255, 255)],
        "Blue": [(90, 50, 50), (130, 255, 255)],
        "Red": [(0, 100, 100), (10, 255, 255)],
        "Black": [(0, 0, 0), (180, 255, 50)]
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

def send_to_api(filename, text, plate_color, category, date_time):
    data = {
        "filename": filename,
        "number_plate_text": text,
        "plate_color": plate_color,
        "category": category,
        "date_time": date_time
    }
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            print("✅ Sent to API successfully:", response.json())
        else:
            print("❌ API Error:", response.status_code, "Response:", response.text)
    except Exception as e:
        print("❌ Failed to send data to API:", str(e))

@app.route('/', methods=['GET'])
def detect_plate():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        return jsonify({"error": "Could not open camera."})
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Filename', 'Number Plate Text', 'Plate Color', 'Category', 'Date Time'])
    start_time = time.time()
    best_plate_img, best_conf = None, 0.5
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
        cv2.imshow('Number Plate Detection', results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if best_plate_img is not None and best_plate_img.size > 0:
        filename = os.path.join(image_dir, 'plate_best.png')
        cv2.imwrite(filename, best_plate_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        text = perform_ocr(best_plate_img)
        plate_color = detect_plate_color(best_plate_img)
        category = categorize_plate(plate_color)
        date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_writer.writerow([filename, text, plate_color, category, date_time])
        csv_file.flush()
        print("Stored in CSV - Number Plate:", text, "Color:", plate_color, "Category:", category, "Timestamp:", date_time)
        send_to_api(filename, text, plate_color, category, date_time)
    else:
        print("Error: No valid plate image detected.")
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({
        "filename": filename if best_plate_img is not None else "N/A",
        "number_plate_text": text if best_plate_img is not None else "N/A",
        "plate_color": plate_color if best_plate_img is not None else "N/A",
        "category": category if best_plate_img is not None else "N/A",
        "date_time": date_time if best_plate_img is not None else "N/A"
    })

if __name__ == '__main__':
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True)


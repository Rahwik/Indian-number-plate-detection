# PROJECT: Number Plate Detection and Classification System

![Car Detection](https://raw.githubusercontent.com/Rahwik/Indian-number-plate-detection/main/car.gif)

**Overview**  
This system uses a YOLOv8 model to detect vehicle number plates from a live camera feed. Once detected, it extracts the plate text using an OCR fallback pipeline, identifies the plate color, classifies the vehicle type, stores the best result over a 10-second window, logs the output into a CSV, and sends the data to a defined API endpoint.

---

**Features**  
- Real-time webcam-based detection  
- OCR pipeline: PaddleOCR → EasyOCR → Tesseract (fallback mechanism)  
- Plate color detection using HSV color space  
- Vehicle category classification based on plate color  
- Captures and stores the best-detected plate every 10 seconds  
- Logs results into a CSV file  
- Sends data via POST request to an API  

---

**Vehicle Categories Based on Plate Color**  
- White: Private Vehicle  
- Yellow: Commercial Vehicle  
- Green: Electric Vehicle  
- Red: Government Vehicle  
- Blue: Diplomatic Vehicle  
- Black: Rental/Self-drive  

---

**Directory Structure**
```
project_root/
├── app.py                  # Main Flask application
├── models/
│   └── best.pt             # Trained YOLOv8 weights
├── images/
│   └── plate_best.png      # Most recent valid detection
├── number_plates.csv       # Output log
├── requirements.txt        # Python dependencies
└── README.txt              # Project description
```

---

**Setup Instructions**

1. Install Tesseract  
   - Windows: https://github.com/tesseract-ocr/tesseract  
   - Make sure it’s added to your system path  

2. Install required Python packages  
   ```
   pip install -r requirements.txt
   ```

3. Place your YOLOv8 weights (`best.pt`) in the `models/` directory  

4. Run the Flask app  
   ```
   python app.py
   ```

5. Open your browser to: `http://127.0.0.1:5000/`

---

**Sample Output (CSV Entry)**  
```
Filename: images/plate_best.png  
Number Plate Text: CG04AB1234  
Plate Color: White  
Category: Private Vehicle  
Date Time: 2025-04-16 12:25:43  
```

---

**API Integration**  
- **Endpoint**:  
  `https://44b58635-e9bd-4d2e-83f5-f574643a3f4d.mock.pstmn.io/detect-plate`  
- **POST Format**:  
```json
{
  "filename": "images/plate_best.png",
  "number_plate_text": "CG04AB1234",
  "plate_color": "White",
  "category": "Private Vehicle",
  "date_time": "2025-04-16 12:25:43"
}
```

---

**OCR Fallback Flow**
1. Try with PaddleOCR  
2. If no output, fallback to EasyOCR  
3. If still empty, use Tesseract (after upscaling)  
4. If all fail, assign `"N/A"`  

---

**Troubleshooting**
- If the webcam isn’t detected: check camera access permissions or try `cv2.VideoCapture(1)`  
- If PaddleOCR throws errors: ensure you're in CPU mode or reinstall dependencies  
- No output from model: confirm `best.pt` exists and matches training format  
- OCR returns gibberish: try adjusting image brightness/contrast or re-train model with better annotations  

---

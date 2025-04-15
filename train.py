from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data=r'D:\numberplate\dataset\data.yaml',
    epochs=50,        
    batch=16,                
    imgsz=1280,                  
    name='license_plate_detector' 
)

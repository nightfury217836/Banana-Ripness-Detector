from ultralytics import YOLO
import torch

# Force CPU usage
device = "cpu"

# Load small YOLO11 model (nano = best for CPU)
model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=20,
    patience=10,             
    imgsz=416,          
    batch=4,            
    device=device,      
    workers=2,         
    cache=False,        
    name="banana_ripeness_cpu"
)
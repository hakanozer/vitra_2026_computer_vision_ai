import os
import sys
import cv2
import numpy as np

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not found.")
        return

    model_path = "data/models/registry/v20260713_001/model.pt"
    print(f"Loading PyTorch model from {model_path}...")
    model = YOLO(model_path)
    
    images = [
        "5fa80da4-9e29-4a93-a71a-440b10808773",
        "9b005703-ab74-433a-b896-ee589fe7f8ab",
        "1243b7c1-5991-489f-8185-c68e9aa411a1",
        "bd37a79d-9131-4056-9c43-0c0ebac2ed14"
    ]
    
    for img_id in images:
        img_path = f"data/labeling_queue/{img_id}.png"
        print(f"\n--- PyTorch Inference on {img_path} ---")
        results = model(img_path, conf=0.01)
        for r in results:
            print(f"Boxes found: {len(r.boxes)}")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"  Class: {cls_name} (ID: {cls_id}), Conf: {conf:.4f}, BBox: {xyxy}")

if __name__ == "__main__":
    main()

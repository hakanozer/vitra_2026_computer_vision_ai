import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.onnx_inference import OnnxInference
from src.utils.config_loader import config

def main():
    model_path = "data/models/registry/v20260713_001/model.onnx"
    print(f"Loading model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    class_names = ["defect_scratch", "defect_crack", "defect_dent", "ok"]
    inference = OnnxInference(session, class_names)
    
    # Temporarily set conf_threshold to 0.01 to see all raw predictions
    inference._conf_threshold = 0.01
    inference._candidate_threshold = 0.01
    
    images = [
        "5fa80da4-9e29-4a93-a71a-440b10808773",
        "9b005703-ab74-433a-b896-ee589fe7f8ab",
        "1243b7c1-5991-489f-8185-c68e9aa411a1",
        "bd37a79d-9131-4056-9c43-0c0ebac2ed14"
    ]
    
    for img_id in images:
        img_path = f"data/labeling_queue/{img_id}.png"
        print(f"\n--- Testing image {img_path} ---")
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read image {img_path}")
            continue
        
        result = inference.predict(frame, camera_id="camera-0")
        print(f"Inference time: {result.inference_time_ms:.2f} ms")
        print(f"Detections ({len(result.detections)}):")
        for d in result.detections:
            print(f"  Class: {d.class_name} (ID: {d.class_id}), Conf: {d.confidence:.4f}, BBox XYXY: {d.bbox_xyxy}")

if __name__ == "__main__":
    main()

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

# Add project root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.onnx_inference import OnnxInference, INPUT_SIZE

def main():
    model_path = "data/models/registry/v20260713_001/model.onnx"
    print(f"Loading model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    
    outputs = session.get_outputs()
    for i, out in enumerate(outputs):
        print(f"Output {i} name: {out.name}, shape: {out.shape}, type: {out.type}")
        
    img_path = "data/labeling_queue/5fa80da4-9e29-4a93-a71a-440b10808773.png"
    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to read image")
        return
        
    # Preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    
    raw_outputs = session.run(None, {input_name: input_tensor})
    print(f"\nNumber of outputs returned: {len(raw_outputs)}")
    for i, val in enumerate(raw_outputs):
        print(f"Output {i} shape: {val.shape}")
        print(f"Output {i} min: {np.min(val)}, max: {np.max(val)}")
        print(f"Output {i} mean: {np.mean(val)}")
        # Let's print some values of the raw output
        if val.ndim == 3:
            predictions = val[0]
            print(f"  predictions shape: {predictions.shape}")
            # If shape is (8, 8400), print predictions[4:, :] max
            if predictions.shape[0] >= 4:
                box_part = predictions[:4, :]
                score_part = predictions[4:, :]
                print(f"  Box part shape: {box_part.shape}, min={np.min(box_part)}, max={np.max(box_part)}")
                print(f"  Score part shape: {score_part.shape}, min={np.min(score_part)}, max={np.max(score_part)}")
                # Show top 5 highest scores and their indices
                max_scores = np.max(score_part, axis=0)
                top_indices = np.argsort(max_scores)[-5:]
                print(f"  Top 5 max scores across all anchors: {max_scores[top_indices]}")
                print(f"  Top 5 anchor predictions:\n{predictions[:, top_indices]}")

if __name__ == "__main__":
    main()

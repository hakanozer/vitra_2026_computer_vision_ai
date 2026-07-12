#!/usr/bin/env python3
"""
scripts/export.py
best.pt → best.onnx dışa aktarma CLI wrapper.
Çalıştırma: python scripts/export.py data/training_runs/train_latest/weights/best.pt
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.export_onnx import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/evaluate.py
Yeni modeli mevcut production modeliyle karşılaştıran değerlendirme scripti.
Çalıştırma: python scripts/evaluate.py --onnx data/models/registry/v20240115_001/model.onnx --dataset latest --version v20240115_001
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.model_evaluator import compare_and_decide


def main():
    parser = argparse.ArgumentParser(description="Model karşılaştırma ve karar")
    parser.add_argument("--onnx", required=True, help="Değerlendirilecek ONNX model yolu")
    parser.add_argument("--dataset", default="latest", help="Validation dataset versiyonu")
    parser.add_argument("--version", default="pending", help="Model versiyon adı")
    args = parser.parse_args()

    result = compare_and_decide(
        onnx_path=Path(args.onnx),
        dataset_version=args.dataset,
        new_version=args.version,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["decision"] == "promote" else 1)


if __name__ == "__main__":
    main()

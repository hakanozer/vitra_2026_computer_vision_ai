"""
src/training/export_onnx.py
best.pt → best.onnx dışa aktarma scripti.
Host ortamında çalışır (ultralytics gerektirir).
"""
import argparse
import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def export(pt_path: str | Path, output_path: str | Path = None) -> Path:
    """
    YOLOv8 .pt modelini ONNX formatına dışa aktarır.

    Parameters
    ----------
    pt_path     : best.pt dosyasının yolu
    output_path : Hedef .onnx dosyasının yolu (verilmezse pt ile aynı dizine yazılır)

    Returns
    -------
    Path : Üretilen .onnx dosyasının yolu
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics bulunamadı. requirements-training.txt kurunuz.")

    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"PT model not found: {pt_path}")

    logger.info("Exporting %s → ONNX …", pt_path.name)

    model = YOLO(str(pt_path))

    # ONNX export
    # opset=12 → geniş uyumluluk; simplify=True → gereksiz node'ları temizler
    export_result = model.export(
        format="onnx",
        opset=12,
        simplify=True,
        dynamic=False,   # Sabit batch=1 — inference için optimize
        half=False,      # FP32 — CPU/CoreML için güvenli
    )

    # ultralytics, export yolunu döner
    onnx_path = Path(str(export_result))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if onnx_path.resolve() != output_path.resolve():
            import shutil
            shutil.copy2(onnx_path, output_path)
            onnx_path = output_path

    logger.info("ONNX export complete: %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="best.pt → best.onnx dışa aktarma")
    parser.add_argument("pt_path", help="Kaynak .pt model dosyası")
    parser.add_argument("--output", default=None, help="Hedef .onnx yolu")
    args = parser.parse_args()

    onnx_path = export(args.pt_path, args.output)
    print(json.dumps({"onnx_path": str(onnx_path)}, indent=2))


if __name__ == "__main__":
    main()

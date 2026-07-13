#!/usr/bin/env python3
"""
scripts/bbox_picker.py
Bir görüntü üzerinde fareyle kutu çizip bbox_xywh koordinatlarını üretir.
Birden fazla kutu çizebilirsiniz (her biri için sınıf seçilir), sonunda
doğrudan /api/labeling/label/{id} endpoint'ine POST gönderilir.

Kullanım:
    python scripts/bbox_picker.py --id <sample_id>
    python scripts/bbox_picker.py --image data/labeling_queue/<uuid>.png

Kontroller:
    Sol tık + sürükle : kutu çiz
    Kutu çizince       : terminalde sınıf numarası sorar (0,1,2,3)
    'u'                : son kutuyu geri al (undo)
    's'                : bitir, etiketi otomatik gönder
    'q' / ESC          : iptal
"""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2

CLASS_NAMES = ["defect_scratch", "defect_crack", "defect_dent", "ok"]

boxes = []          # [(class_id, x, y, w, h), ...]
drawing = False
start_point = (0, 0)
current_point = (0, 0)
img = None
img_display = None


def redraw():
    global img_display
    img_display = img.copy()
    for cls_id, x, y, w, h in boxes:
        color = (0, 255, 0) if CLASS_NAMES[cls_id] == "ok" else (0, 0, 255)
        cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_display, CLASS_NAMES[cls_id], (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if drawing:
        cv2.rectangle(img_display, start_point, current_point, (255, 255, 0), 1)


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_point = (x, y)
        redraw()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = start_point
        x2, y2 = current_point
        bx, by = min(x1, x2), min(y1, y2)
        bw, bh = abs(x2 - x1), abs(y2 - y1)
        if bw > 2 and bh > 2:
            print("\nSınıf seçin:")
            for i, name in enumerate(CLASS_NAMES):
                print(f"  {i} = {name}")
            while True:
                choice = input("class_id: ").strip()
                if choice.isdigit() and int(choice) < len(CLASS_NAMES):
                    boxes.append((int(choice), bx, by, bw, bh))
                    break
                print("Geçersiz, tekrar deneyin.")
        redraw()


def main():
    global img

    parser = argparse.ArgumentParser(description="Fareyle bbox çizme aracı")
    parser.add_argument("--id", help="labeling_queue içindeki sample_id")
    parser.add_argument("--image", help="Doğrudan görüntü dosyası yolu")
    parser.add_argument(
        "--queue-dir", default="data/labeling_queue",
        help="labeling_queue klasörü (varsayılan: data/labeling_queue)",
    )
    args = parser.parse_args()

    if args.id:
        img_path = Path(args.queue_dir) / f"{args.id}.png"
        sample_id = args.id
    elif args.image:
        img_path = Path(args.image)
        sample_id = img_path.stem
    else:
        parser.error("--id veya --image belirtmelisiniz")

    if not img_path.exists():
        print(f"Görüntü bulunamadı: {img_path}")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Görüntü okunamadı: {img_path}")
        return

    redraw()
    cv2.namedWindow("bbox_picker")
    cv2.setMouseCallback("bbox_picker", mouse_callback)

    print("Sol tık + sürükle: kutu çiz | 'u': geri al | 's': bitir | 'q': iptal")

    while True:
        cv2.imshow("bbox_picker", img_display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and boxes:
            boxes.pop()
            redraw()
        elif key == ord("s"):
            break
        elif key == ord("q") or key == 27:
            boxes.clear()
            break

    cv2.destroyAllWindows()

    if not boxes:
        print("Hiç kutu çizilmedi.")
        return

    annotations = [
        {"class_id": cls_id, "bbox_xywh": [x, y, w, h], "confidence": 1.0}
        for cls_id, x, y, w, h in boxes
    ]
    payload = {
        "annotations": annotations,
        "class_names": CLASS_NAMES,
        "operator_id": "manual",
    }

    url = f"http://localhost:8000/api/labeling/label/{sample_id}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=10) as response:
            response.read()
        print(f"Etiket gönderildi: {sample_id}")
    except HTTPError as error:
        print(f"Etiket gönderilemedi: {error.code} {error.reason}")
    except URLError as error:
        print(f"Etiket gönderilemedi: {error.reason}")


if __name__ == "__main__":
    main()
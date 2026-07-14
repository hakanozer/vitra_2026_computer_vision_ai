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
import numpy as np

CLASS_NAMES = ["defect_scratch", "defect_crack", "defect_dent", "ok"]
PANEL_WIDTH = 280
WINDOW_NAME = "bbox_picker"
CLASS_BUTTON_TOP = 70
CLASS_BUTTON_HEIGHT = 44
CLASS_BUTTON_GAP = 12
ACTION_BUTTON_HEIGHT = 44
ACTION_BUTTON_GAP = 12

boxes = []          # [(class_id, x, y, w, h), ...]
drawing = False
start_point = (0, 0)
current_point = (0, 0)
img = None
img_display = None
selected_class_id = 0
action_regions = {}
save_requested = False
cancel_requested = False
status_message = "Sınıf seçip kutu çizin."


def _class_button_rect(index: int, image_width: int) -> tuple[int, int, int, int]:
    x1 = image_width + 18
    y1 = CLASS_BUTTON_TOP + index * (CLASS_BUTTON_HEIGHT + CLASS_BUTTON_GAP)
    x2 = image_width + PANEL_WIDTH - 18
    y2 = y1 + CLASS_BUTTON_HEIGHT
    return x1, y1, x2, y2


def _action_button_rect(name: str, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    buttons_top = image_height - (ACTION_BUTTON_HEIGHT * 3 + ACTION_BUTTON_GAP * 2 + 24)
    names = ["save", "undo", "cancel"]
    index = names.index(name)
    x1 = image_width + 18
    y1 = buttons_top + index * (ACTION_BUTTON_HEIGHT + ACTION_BUTTON_GAP)
    x2 = image_width + PANEL_WIDTH - 18
    y2 = y1 + ACTION_BUTTON_HEIGHT
    return x1, y1, x2, y2


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _draw_button(canvas, rect, text, *, active=False, primary=False, danger=False):
    x1, y1, x2, y2 = rect
    if danger:
        fill = (64, 64, 180)
        border = (90, 90, 220)
    elif primary:
        fill = (180, 120, 30)
        border = (220, 160, 70)
    elif active:
        fill = (50, 110, 40)
        border = (90, 170, 80)
    else:
        fill = (55, 55, 55)
        border = (90, 90, 90)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
    cv2.putText(
        canvas,
        text,
        (x1 + 10, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def redraw():
    global img_display, action_regions
    image_h, image_w = img.shape[:2]
    canvas = np.full((image_h, image_w + PANEL_WIDTH, 3), 24, dtype=np.uint8)
    canvas[:, :image_w] = img.copy()

    for cls_id, x, y, w, h in boxes:
        color = (0, 255, 0) if CLASS_NAMES[cls_id] == "ok" else (0, 0, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(canvas, CLASS_NAMES[cls_id], (x, max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if drawing:
        cv2.rectangle(canvas, start_point, current_point, (255, 255, 0), 1)

    panel_x = image_w
    cv2.rectangle(canvas, (panel_x, 0), (panel_x + PANEL_WIDTH, image_h), (31, 41, 55), -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, image_h), (70, 80, 95), 1)

    cv2.putText(
        canvas,
        "Controls",
        (panel_x + 18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Class secin ve kutu cizin",
        (panel_x + 18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 190, 205),
        1,
        cv2.LINE_AA,
    )

    action_regions = {}
    for index, name in enumerate(CLASS_NAMES):
        rect = _class_button_rect(index, image_w)
        action_regions[f"class:{index}"] = rect
        _draw_button(canvas, rect, f"{index} - {name}", active=index == selected_class_id)

    for name, label, primary, danger in [
        ("save", "Save", True, False),
        ("undo", "Undo", False, False),
        ("cancel", "Cancel", False, True),
    ]:
        rect = _action_button_rect(name, image_w, image_h)
        action_regions[name] = rect
        _draw_button(canvas, rect, label, primary=primary, danger=danger)

    cv2.putText(
        canvas,
        f"Selected: {CLASS_NAMES[selected_class_id]}",
        (panel_x + 18, image_h - 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Boxes: {len(boxes)}",
        (panel_x + 18, image_h - 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status_message[:34],
        (panel_x + 18, image_h - 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 190, 205),
        1,
        cv2.LINE_AA,
    )

    img_display = canvas


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_point, selected_class_id, save_requested, cancel_requested, status_message
    image_h, image_w = img.shape[:2]

    if x >= image_w:
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, rect in action_regions.items():
                if _point_in_rect(x, y, rect):
                    if key.startswith("class:"):
                        selected_class_id = int(key.split(":", 1)[1])
                        status_message = f"Secilen sinif: {CLASS_NAMES[selected_class_id]}"
                    elif key == "save":
                        save_requested = True
                    elif key == "undo" and boxes:
                        boxes.pop()
                        status_message = "Son kutu geri alindi."
                    elif key == "cancel":
                        boxes.clear()
                        cancel_requested = True
                    redraw()
                    return

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
            boxes.append((selected_class_id, bx, by, bw, bh))
            status_message = f"Kutu eklendi: {CLASS_NAMES[selected_class_id]}"
        redraw()


def main():
    global img, save_requested, cancel_requested, status_message

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
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        cv2.imshow(WINDOW_NAME, img_display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and boxes:
            boxes.pop()
            status_message = "Son kutu geri alindi."
            redraw()
        elif key == ord("s"):
            save_requested = True
        elif key == ord("q") or key == 27:
            cancel_requested = True

        if save_requested:
            save_requested = False
            break
        if cancel_requested:
            cancel_requested = False
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
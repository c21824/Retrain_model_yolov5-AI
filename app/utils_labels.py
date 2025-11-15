import os, json, logging
from typing import List, Dict, Tuple, Optional
from PIL import Image

logger = logging.getLogger("utils_labels")
logging.basicConfig(level=logging.INFO)

# kiểm tra nomalize
def _is_normalized(vals: List[float]) -> bool:
    try:
        return all(0.0 <= float(v) <= 1.0 for v in vals)
    except Exception:
        return False

# nhận về dạng xyxy và chuyển sang nomalize
def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    x1f = max(0.0, float(x1)); y1f = max(0.0, float(y1))
    x2f = max(0.0, float(x2)); y2f = max(0.0, float(y2))
    x1f = min(x1f, img_w - 1); x2f = min(x2f, img_w - 1)
    y1f = min(y1f, img_h - 1); y2f = min(y2f, img_h - 1)

    if x2f <= x1f or y2f <= y1f:
        return None
    cx = (x1f + x2f) / 2.0
    cy = (y1f + y2f) / 2.0
    w = x2f - x1f
    h = y2f - y1f
    return (cx / img_w, cy / img_h, w / img_w, h / img_h)

# trả lại nhãn của yolo
def cxcywh_to_yolo(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int, normalized: bool=False):
    if normalized:
        return (float(cx), float(cy), float(w), float(h))
    return (float(cx) / img_w, float(cy) / img_h, float(w) / img_w, float(h) / img_h)

# tạo chuỗi theo format của yolo
def format_line(cid: int, xc: float, yc: float, w: float, h: float, prec: int = 6) -> str:
    return f"{int(cid)} {xc:.{prec}f} {yc:.{prec}f} {w:.{prec}f} {h:.{prec}f}"

#tạo file .txt trong bên label với tên như ảnh của mẫu
def write_yolo_label_single_class(labels_dir: str, image_basename: str, boxes: List[Dict],
                                  img_size: Tuple[int, int], precision: int = 6):
    os.makedirs(labels_dir, exist_ok=True)
    base_noext = os.path.splitext(image_basename)[0]
    out_path = os.path.join(labels_dir, base_noext + ".txt")
    img_w, img_h = int(img_size[0]), int(img_size[1])

    lines = []
    for b in boxes or []:
        yolo = None

        if all(k in b for k in ("xTop", "yTop", "xBot", "yBot")):
            x1 = float(b["xTop"]); y1 = float(b["yTop"]); x2 = float(b["xBot"]); y2 = float(b["yBot"])
            if _is_normalized([x1, y1, x2, y2]):
                yolo = xyxy_to_yolo(x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h, img_w, img_h)
            else:
                yolo = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)

        elif all(k in b for k in ("x1", "y1", "x2", "y2")):
            x1 = float(b["x1"]); y1 = float(b["y1"]); x2 = float(b["x2"]); y2 = float(b["y2"])
            if _is_normalized([x1, y1, x2, y2]):
                yolo = xyxy_to_yolo(x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h, img_w, img_h)
            else:
                yolo = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)

        elif all(k in b for k in ("cx", "cy", "w", "h")):
            cx = float(b["cx"]); cy = float(b["cy"]); w = float(b["w"]); h = float(b["h"])
            if _is_normalized([cx, cy, w, h]):
                yolo = cxcywh_to_yolo(cx, cy, w, h, img_w, img_h, normalized=True)
            else:
                yolo = cxcywh_to_yolo(cx, cy, w, h, img_w, img_h, normalized=False)

        elif "bbox" in b:
            arr = b["bbox"]
            try:
                arr = list(arr)
                if len(arr) == 4:
                    if _is_normalized(arr):
                        yolo = xyxy_to_yolo(arr[0] * img_w, arr[1] * img_h, arr[2] * img_w, arr[3] * img_h, img_w, img_h)
                    else:
                        yolo = xyxy_to_yolo(arr[0], arr[1], arr[2], arr[3], img_w, img_h)
            except Exception:
                yolo = None

        if yolo is None:
            logger.debug("skip unknown box format: %s", b)
            continue

        xc, yc, ww, hh = yolo
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= ww <= 1.0 and 0.0 <= hh <= 1.0):
            logger.debug("box out of range skip: %s", (xc, yc, ww, hh))
            continue

        lines.append(format_line(0, xc, yc, ww, hh, precision))

    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write("\n".join(lines))
    return out_path

"""
preview_sample_and_labels.py (updated)

Mục đích:
 - Cho xem trước (preview) một sample và các label sẽ được chèn vào DB.
 - Với numeric class 0 -> chuyển thành "cccd" (trường label trong DB).
 - KHÔNG ghi DB. Chỉ in ra dict / JSON của sample và list labels.

Cách dùng:
 - Sửa IMAGES_DIR / LABELS_DIR theo bạn.
 - Gọi hàm preview_by_basename("0001") hoặc chạy script và truyền basename/label file.
"""
import os
import json
import datetime
from typing import Optional, Tuple, Dict
from PIL import Image

# cấu hình
IMAGES_DIR = r"F:\Deep_learning\PTHTTM\data\detect1\train\images"
LABELS_DIR = r"F:\Deep_learning\PTHTTM\data\detect1\train\labels"
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# MAPPING: sửa/extend nếu cần. key là int class index, value là tên sẽ lưu vào tblLabel.label
CLASS_MAPPING = {
    0: "cccd"
    # ví dụ: 1: "person", 2: "id_card", ...
}

def find_image_path_for_basename(basename: str) -> Optional[str]:
    for ext in IMG_EXTS:
        p = os.path.join(IMAGES_DIR, basename + ext)
        if os.path.exists(p):
            return p
    return None

def parse_label_line(line: str) -> Optional[Tuple[str,float,float,float,float]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = parts[0]
    try:
        xc = float(parts[1]); yc = float(parts[2]); w = float(parts[3]); h = float(parts[4])
    except ValueError:
        return None
    return cls, xc, yc, w, h

def map_class_to_label_string(cls_raw: str) -> str:
    """
    Map class token to the label string to store in DB.
    - If cls_raw is numeric (e.g. "0" or "0.0") and exists in CLASS_MAPPING -> mapped value.
    - If numeric but not in mapping -> use int(cls_raw) as string (e.g. "1")
    - If non-numeric (e.g. "cccd" already) -> use as-is.
    """
    try:
        cls_int = int(float(cls_raw))
        # if mapping exists, use it; otherwise use string of int (keeps deterministic)
        return CLASS_MAPPING.get(cls_int, str(cls_int))
    except Exception:
        # non-numeric label (like "cccd" or "person"), return as-is
        return cls_raw

def yolo_to_bbox(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float,float,float,float,bool]:
    """
    Trả về (xTop, yTop, xBot, yBot, is_normalized)
    """
    is_norm = max(xc, yc, w, h) <= 1.0
    if is_norm:
        xc_px = xc * img_w
        yc_px = yc * img_h
        w_px = w * img_w
        h_px = h * img_h
    else:
        xc_px = xc
        yc_px = yc
        w_px = w
        h_px = h

    xTop = xc_px - w_px / 2.0
    yTop = yc_px - h_px / 2.0
    xBot = xc_px + w_px / 2.0
    yBot = yc_px + h_px / 2.0

    # clamp vào trong ảnh
    xTop = max(0.0, min(xTop, img_w))
    yTop = max(0.0, min(yTop, img_h))
    xBot = max(0.0, min(xBot, img_w))
    yBot = max(0.0, min(yBot, img_h))

    return xTop, yTop, xBot, yBot, is_norm

def preview_by_basename(basename: str, sample_id_placeholder: Optional[int]=None) -> Dict:
    """
    basename: ví dụ '0001' tương ứng 0001.txt và 0001.jpg...
    sample_id_placeholder: nếu muốn hiển thị id giả (chưa có trong DB)
    Trả về dict chứa 'sample' và 'labels'.
    """
    label_fp = os.path.join(LABELS_DIR, basename + ".txt")
    if not os.path.exists(label_fp):
        raise FileNotFoundError(f"Label file không tồn tại: {label_fp}")

    img_path = find_image_path_for_basename(basename)
    if img_path is None:
        raise FileNotFoundError(f"Ảnh tương ứng không tìm thấy trong IMAGES_DIR cho basename {basename}")

    # đọc kích thước ảnh
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    sample = {
        "nameImg": os.path.basename(img_path),
        "path": img_path,
        "createDate": datetime.date.today().isoformat(),
        "type": "detect1",
        "id_preview": sample_id_placeholder
    }

    labels = []
    with open(label_fp, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for i, ln in enumerate(lines, start=1):
        parsed = parse_label_line(ln)
        if parsed is None:
            labels.append({
                "line_no": i,
                "raw": ln,
                "error": "cannot parse (need: class xc yc w h)"
            })
            continue
        cls_raw, xc, yc, w, h = parsed

        # map class -> label string that will be stored in DB
        label_to_store = map_class_to_label_string(cls_raw)

        xTop, yTop, xBot, yBot, is_norm = yolo_to_bbox(xc, yc, w, h, img_w, img_h)
        lbl = {
            "line_no": i,
            "raw": ln,
            "class_raw": cls_raw,
            "label_to_store": label_to_store,   # <-- đây là giá trị sẽ lưu vào tblLabel.label
            "normalized_input": bool(is_norm),
            "xc": xc, "yc": yc, "w": w, "h": h,
            "xTop": round(xTop, 3),
            "yTop": round(yTop, 3),
            "xBot": round(xBot, 3),
            "yBot": round(yBot, 3),
            # tblSampleid sẽ được gắn khi insert vào DB; preview hiển thị placeholder
            "tblSampleid_preview": sample_id_placeholder
        }
        labels.append(lbl)

    result = {
        "sample": sample,
        "image_size": {"width": img_w, "height": img_h},
        "labels": labels,
        "label_count": len(labels)
    }
    # in đẹp ra console
    print("=== PREVIEW SAMPLE ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

# Ví dụ chạy khi chạy file trực tiếp
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preview_sample_and_labels.py <basename> [sample_id_preview]")
        print("Ví dụ: python preview_sample_and_labels.py 0001 123")
        sys.exit(1)
    basename = sys.argv[1]
    sid = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    try:
        preview_by_basename(basename, sample_id_placeholder=sid)
    except Exception as e:
        print("Lỗi:", e)

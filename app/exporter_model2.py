import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

import yaml

from db import SessionLocal

from models import TblDatasetDetail, TblSample, TblLabel

from utils_labels import xyxy_to_yolo, cxcywh_to_yolo, format_line

logger = logging.getLogger("exporter_model2")
logging.basicConfig(level=logging.INFO)

DATASETS_ROOT = Path(r"F:\Deep_learning\PTHTTM\data\datasets").resolve()


def get_dataset_details_from_db(dataset_id: int) -> List[Dict[str, Any]]:
    out = []
    db = SessionLocal()
    try:
        details = db.query(TblDatasetDetail).filter(TblDatasetDetail.tblDatasetid == dataset_id).all()
        if not details:
            return []

        seen = set()
        sample_ids = []
        for d in details:
            sid = d.tblSampleid
            if sid not in seen:
                seen.add(sid)
                sample_ids.append(sid)
        samples = db.query(TblSample).filter(TblSample.id.in_(sample_ids)).all()
        sample_map = {s.id: s for s in samples}
        labels = db.query(TblLabel).filter(TblLabel.tblSampleid.in_(sample_ids)).all()
        labels_by_sample = {}
        for lab in labels:
            sid = lab.tblSampleid
            labels_by_sample.setdefault(sid, []).append(lab)

        for sid in sample_ids:
            s = sample_map.get(sid)
            if s is None:
                logger.warning("sample id %s referenced in tblDatasetDetail but not found in TblSample", sid)
                continue
            lab_list = []
            for lab in labels_by_sample.get(sid, []):
                lab_list.append({
                    "xTop": lab.xTop,
                    "yTop": lab.yTop,
                    "xBot": lab.xBot,
                    "yBot": lab.yBot,
                    "label": lab.label
                })
            out.append({
                "sample_id": s.id,
                "nameImg": s.nameImg,
                "image_path": s.path,
                "labels": lab_list
            })
    finally:
        db.close()
    return out


# create labels (multi-class) for a sample using a name->index mapping
def create_label_file_for_sample_multi(sample: Dict[str, Any], images_dir: str, labels_dir: str,
                                       class_to_index: Dict[str, int]) -> Optional[str]:
    src_img = sample.get("image_path")
    if not src_img:
        logger.warning("no image_path for sample %s", sample.get("sample_id"))
        return None

    basename = os.path.basename(src_img)
    sid = str(sample.get("sample_id"))
    prefix = f"sample_{sid}_"
    if basename.startswith(prefix):
        dst_basename = basename
    else:
        dst_basename = f"{prefix}{basename}"

    dst_img_path = os.path.join(images_dir, dst_basename)
    if not os.path.exists(dst_img_path):
        if os.path.exists(src_img) and os.path.abspath(os.path.dirname(src_img)) != os.path.abspath(images_dir):
            try:
                shutil.copy2(src_img, dst_img_path)
            except Exception as e:
                logger.exception("copy failed %s -> %s : %s", src_img, dst_img_path, e)
                return None
        else:
            candidate = os.path.join(images_dir, basename)
            if os.path.exists(candidate):
                try:
                    os.rename(candidate, dst_img_path)
                except Exception as e:
                    logger.exception("rename existing candidate failed %s -> %s: %s", candidate, dst_img_path, e)
                    try:
                        shutil.copy2(candidate, dst_img_path)
                    except Exception as e2:
                        logger.exception("fallback copy failed %s -> %s : %s", candidate, dst_img_path, e2)
                        return None
            else:
                logger.warning("source image not found for sample %s: %s", sid, src_img)
                return None

    try:
        with Image.open(dst_img_path) as im:
            w, h = im.size
    except Exception as e:
        logger.exception("open image failed %s: %s", dst_img_path, e)
        return None

    raw_labels = sample.get("labels") or []
    lines = []
    for lb in raw_labels:
        if isinstance(lb, dict):
            xTop = lb.get("xTop"); yTop = lb.get("yTop"); xBot = lb.get("xBot"); yBot = lb.get("yBot")
            label_name = lb.get("label")
        else:
            xTop = getattr(lb, "xTop", None); yTop = getattr(lb, "yTop", None)
            xBot = getattr(lb, "xBot", None); yBot = getattr(lb, "yBot", None)
            label_name = getattr(lb, "label", None)

        try:
            yolo = None
            if None not in (xTop, yTop, xBot, yBot):
                # handle normalized or absolute xyxy
                try:
                    x1 = float(xTop); y1 = float(yTop); x2 = float(xBot); y2 = float(yBot)
                    if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
                        yolo = xyxy_to_yolo(x1 * w, y1 * h, x2 * w, y2 * h, w, h)
                    else:
                        yolo = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                except Exception:
                    yolo = None
            if yolo is None:
                logger.debug("skip unknown box format: %s", lb)
                continue

            xc, yc, ww, hh = yolo
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= ww <= 1.0 and 0.0 <= hh <= 1.0):
                logger.debug("box out of range skip: %s", (xc, yc, ww, hh))
                continue

            cid = 0
            if label_name is not None:
                cid = class_to_index.get(str(label_name), class_to_index.get(label_name, 0))
            lines.append(format_line(int(cid), xc, yc, ww, hh))
        except Exception:
            continue

    os.makedirs(labels_dir, exist_ok=True)
    base_noext = os.path.splitext(dst_basename)[0]
    out_path = os.path.join(labels_dir, base_noext + ".txt")
    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write("\n".join(lines))
    return out_path


def create_dataset_for_detect2(dataset_id: int, output_base: Optional[str] = None,
                               val_images_dir: Optional[str] = None,
                               val_ratio: float = 0.1,
                               classes: Optional[List[str]] = None) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]

    if output_base:
        ds_path = Path(output_base).resolve()
    else:
        ds_path = DATASETS_ROOT / f"dataset{dataset_id}"

    train_images_dir = ds_path / "train" / "images"
    train_labels_dir = ds_path / "train" / "labels"
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    if val_images_dir is None:
        default_val = project_root / "data" / "detect2" / "valid" / "images"
        if default_val.exists() and default_val.is_dir():
            val_images_dir = str(default_val.resolve())
            logger.info("Using default detect2 valid images at %s", val_images_dir)
        else:
            val_images_dir = None
            logger.info("No default detect2 valid found; will fallback to splitting if necessary")

    samples = get_dataset_details_from_db(dataset_id)
    if not samples:
        raise ValueError("No samples found for dataset_id=" + str(dataset_id))

    # build class name mapping: prefer classes param, else try to read data/detect2/data.yaml
    names_list = None
    if classes:
        names_list = classes
    else:
        yaml_path = project_root / "data" / "detect2" / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as fy:
                    y = yaml.safe_load(fy)
                    if isinstance(y, dict) and "names" in y:
                        names_list = list(y.get("names") or [])
            except Exception:
                names_list = None

    if not names_list:
        # fallback to single class
        names_list = ["object"]

    class_to_index = {str(n): i for i, n in enumerate(names_list)}

    details = {}
    for s in samples:
        sid = str(s["sample_id"]) if isinstance(s.get("sample_id"), (int, str)) else str(s.get("sample_id"))
        src = s.get("image_path")
        if not src or not os.path.exists(src):
            logger.warning("sample %s source image not found: %s", sid, src)
            continue

        basename = os.path.basename(src)
        prefix = f"sample_{sid}_"
        if basename.startswith(prefix):
            dst_basename = basename
        else:
            dst_basename = f"{prefix}{basename}"

        dst_img_path = train_images_dir / dst_basename

        if not dst_img_path.exists():
            try:
                shutil.copy2(src, dst_img_path)
            except Exception as e:
                logger.exception("copy failed %s -> %s : %s", src, dst_img_path, e)
                continue

        sample_copy = {
            "sample_id": s["sample_id"],
            "nameImg": s.get("nameImg"),
            "image_path": str(dst_img_path),
            "labels": s.get("labels", [])
        }

        txt_path = create_label_file_for_sample_multi(sample_copy, str(train_images_dir), str(train_labels_dir), class_to_index)
        details[sid] = {
            "sample_id": sid,
            "image_basename": dst_basename,
            "src_path": src,
            "label_path": txt_path,
            "labels_count": 0 if not txt_path or not os.path.exists(txt_path) else sum(1 for _ in open(txt_path, "r", encoding="utf-8"))
        }

    train_abs = train_images_dir.resolve().as_posix()
    val_abs = None
    if val_images_dir and os.path.isdir(val_images_dir):
        val_abs = Path(val_images_dir).resolve().as_posix()
    else:
        train_imgs = sorted([f for f in os.listdir(str(train_images_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        total = len(train_imgs)
        if total == 0:
            raise ValueError("No train images created for dataset " + str(dataset_id))
        if val_images_dir is None:
            if total < 5:
                val_dir = ds_path / "valid" / "images"
                os.makedirs(val_dir, exist_ok=True)
                val_abs = val_dir.resolve().as_posix()
                logger.info("Too few images, created empty valid folder %s", val_dir)
            else:
                val_count = max(1, int(total * val_ratio))
                valid_dir = ds_path / "valid" / "images"
                valid_lbl_dir = ds_path / "valid" / "labels"
                os.makedirs(valid_dir, exist_ok=True)
                os.makedirs(valid_lbl_dir, exist_ok=True)
                moved = 0
                for i, fname in enumerate(train_imgs[:val_count]):
                    src_img = train_images_dir / fname
                    dst_img = valid_dir / fname
                    shutil.move(str(src_img), str(dst_img))
                    lbl_src = train_labels_dir / (os.path.splitext(fname)[0] + ".txt")
                    if lbl_src.exists():
                        shutil.move(str(lbl_src), str(valid_lbl_dir / lbl_src.name))
                    moved += 1
                val_abs = valid_dir.resolve().as_posix()
                logger.info("Fallback created valid set by moving %d images to %s", moved, valid_dir)

    yaml_path = ds_path / "data_detect2.yaml"
    with open(yaml_path, "w", encoding="utf-8") as fy:
        fy.write(f"train: {train_abs}\n")
        if val_abs:
            fy.write(f"val: {val_abs}\n")
        else:
            fy.write("val: \n")
        fy.write("\n")
        fy.write(f"nc: {len(names_list)}\n")
        # write names as a python-like list for detect scripts
        fy.write("names: ")
        fy.write(repr(names_list) + "\n")

    result = {
        "dataset_id": dataset_id,
        "dataset_path": str(ds_path),
        "details": details,
        "yaml": str(yaml_path.resolve())
    }
    logger.info("Created dataset %s train_images=%d", ds_path, len([f for f in os.listdir(str(train_images_dir)) if f.lower().endswith(('.jpg','.jpeg','.png'))]))
    return result

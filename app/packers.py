import os
import json
import logging
from typing import Dict, Any, List, Optional
from PIL import Image

logger = logging.getLogger("packers")
logging.basicConfig(level=logging.INFO)


def write_yaml_datafile(yaml_path: str, train: str, val: str, nc: int, names: List[str]):

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: '{train}'\n")
        f.write(f"val: '{val}'\n")
        f.write(f"nc: {nc}\n")
        f.write("names: [")
        f.write(", ".join([f'\"{n}\"' for n in names]))
        f.write("]\n")


def generate_train_val_txts(images_dir: str,
                            out_dataset_dir: str,
                            val_ratio: float = 0.1,
                            val_images_dir: Optional[str] = None):

    images_dir = os.path.abspath(images_dir)
    out_dataset_dir = os.path.abspath(out_dataset_dir)
    os.makedirs(out_dataset_dir, exist_ok=True)

    def list_images(d):
        if not os.path.isdir(d):
            return []
        return sorted([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if val_images_dir:
        val_images_dir = os.path.abspath(val_images_dir)
        if os.path.isdir(val_images_dir):
            train_imgs = [os.path.join(images_dir, f) for f in list_images(images_dir)]
            val_imgs = [os.path.join(val_images_dir, f) for f in list_images(val_images_dir)]
        else:
            logger.warning("val_images_dir provided but not found: %s. Falling back to split.", val_images_dir)
            imgs = list_images(images_dir)
            total = len(imgs)
            val_count = max(1, int(total * val_ratio)) if total > 0 else 0
            val_list = imgs[:val_count]; train_list = imgs[val_count:]
            train_imgs = [os.path.join(images_dir, f) for f in train_list]
            val_imgs = [os.path.join(images_dir, f) for f in val_list]
    else:
        imgs = list_images(images_dir)
        total = len(imgs)
        val_count = max(1, int(total * val_ratio)) if total > 0 else 0
        val_list = imgs[:val_count]; train_list = imgs[val_count:]
        train_imgs = [os.path.join(images_dir, f) for f in train_list]
        val_imgs = [os.path.join(images_dir, f) for f in val_list]

    train_file = os.path.join(out_dataset_dir, "train.txt")
    val_file = os.path.join(out_dataset_dir, "val.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        for p in train_imgs:
            f.write(os.path.abspath(p) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        for p in val_imgs:
            f.write(os.path.abspath(p) + "\n")
    return os.path.abspath(train_file), os.path.abspath(val_file)


def rewrite_yolo_to_single_class(labels_dir: str):

    if not os.path.isdir(labels_dir):
        logger.warning("labels dir not found for rewrite: %s", labels_dir)
        return
    for fname in os.listdir(labels_dir):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(labels_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fr:
                lines = [ln.strip() for ln in fr.readlines() if ln.strip()]
            new = []
            for ln in lines:
                parts = ln.split()
                if len(parts) >= 5:
                    nx, ny, nw, nh = parts[1:5]
                    new.append(f"0 {nx} {ny} {nw} {nh}")
                elif len(parts) == 4:
                    nx, ny, nw, nh = parts[0:4]
                    new.append(f"0 {nx} {ny} {nw} {nh}")
            with open(path, "w", encoding="utf-8") as fw:
                fw.write("\n".join(new))
        except Exception as e:
            logger.warning("rewrite fail %s: %s", path, e)


def prepare_for_model_after_export(export_result: Dict[str, Any],
                                   model_id: int,
                                   val_images_dir: Optional[str] = None,
                                   force_unique_names: bool = True) -> Dict[str, Any]:

    ds_path = export_result["dataset_path"]
    images_dir = os.path.join(ds_path, "images")
    labels_dir = os.path.join(ds_path, "labels")
    if not os.path.isdir(images_dir):
        raise RuntimeError("images missing: " + images_dir)

    if force_unique_names:
        for sid_str, info in export_result.get("details", {}).items():
            old = info.get("image_basename")
            if not old:
                continue
            name, ext = os.path.splitext(old)
            newname = f"sample_{sid_str}_{name}{ext}"
            old_path = os.path.join(images_dir, old)
            new_path = os.path.join(images_dir, newname)
            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                old_json = os.path.join(labels_dir, os.path.splitext(old)[0] + ".json")
                new_json = os.path.join(labels_dir, os.path.splitext(newname)[0] + ".json")
                if os.path.exists(old_json):
                    os.rename(old_json, new_json)
                old_txt = os.path.join(labels_dir, os.path.splitext(old)[0] + ".txt")
                new_txt = os.path.join(labels_dir, os.path.splitext(newname)[0] + ".txt")
                if os.path.exists(old_txt):
                    os.rename(old_txt, new_txt)
                info["image_basename"] = newname

    if model_id == 1:
        rewrite_yolo_to_single_class(labels_dir)
        train_path, val_path = generate_train_val_txts(images_dir, ds_path, 0.1, val_images_dir=val_images_dir)
        yaml_path = os.path.join(ds_path, "data_detect1.yaml")
        write_yaml_datafile(yaml_path, train_path, val_path, nc=1, names=["cccd"])
        return {"model_id": model_id, "yaml": os.path.abspath(yaml_path), "train_txt": train_path, "val_txt": val_path}

    elif model_id == 2:
        classes = export_result.get("classes", [])
        if not classes:
            ct = os.path.join(ds_path, "classes.txt")
            if os.path.exists(ct):
                with open(ct, "r", encoding="utf-8") as f:
                    classes = [ln.strip() for ln in f if ln.strip()]
        nc = len(classes)
        train_path, val_path = generate_train_val_txts(images_dir, ds_path, 0.1, val_images_dir=val_images_dir)
        yaml_path = os.path.join(ds_path, "data_detect2.yaml")
        write_yaml_datafile(yaml_path, train_path, val_path, nc=nc, names=classes or [])
        return {"model_id": model_id, "yaml": os.path.abspath(yaml_path), "train_txt": train_path, "val_txt": val_path, "classes": classes}

    elif model_id == 3:
        crops_dir = os.path.join(ds_path, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        manifest = []
        for sid_str, info in export_result.get("details", {}).items():
            basename = info.get("image_basename")
            if not basename:
                continue
            image_path = os.path.join(images_dir, basename)
            if not os.path.exists(image_path):
                logger.warning("image not found for cropping: %s", image_path)
                continue
            label_json_path = os.path.join(labels_dir, os.path.splitext(basename)[0] + ".json")
            if not os.path.exists(label_json_path):
                logger.warning("label json not found: %s", label_json_path)
                continue
            try:
                with open(label_json_path, "r", encoding="utf-8") as jf:
                    j = json.load(jf)
                labels = j.get("labels", [])
                if not labels:
                    continue
                with Image.open(image_path) as im:
                    w, h = im.size
                    for idx, lab in enumerate(labels):
                        x1 = lab.get("xTop"); y1 = lab.get("yTop"); x2 = lab.get("xBot"); y2 = lab.get("yBot")
                        if None in (x1, y1, x2, y2):
                            continue
                        try:
                            is_normalized = all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in (x1, y1, x2, y2))
                        except Exception:
                            is_normalized = False
                        try:
                            if is_normalized:
                                ix1 = int(round(float(x1) * w))
                                iy1 = int(round(float(y1) * h))
                                ix2 = int(round(float(x2) * w))
                                iy2 = int(round(float(y2) * h))
                            else:
                                ix1 = int(round(float(x1)))
                                iy1 = int(round(float(y1)))
                                ix2 = int(round(float(x2)))
                                iy2 = int(round(float(y2)))
                        except Exception:
                            continue

                        ix1 = max(0, min(w - 1, ix1))
                        iy1 = max(0, min(h - 1, iy1))
                        ix2 = max(0, min(w - 1, ix2))
                        iy2 = max(0, min(h - 1, iy2))
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue

                        box = (ix1, iy1, ix2, iy2)
                        crop = im.crop(box)
                        crop_name = f"sample_{sid_str}_{idx}.png"
                        crop_path = os.path.join(crops_dir, crop_name)
                        crop.save(crop_path)

                        trans = lab.get("text") or lab.get("transcription") or lab.get("value") or lab.get("label") or ""
                        if trans is None:
                            trans = ""
                        trans = str(trans).strip().replace("\n", " ").replace("\r", " ")
                        if trans == "":
                            continue

                        txt_path = os.path.splitext(crop_path)[0] + ".txt"
                        with open(txt_path, "w", encoding="utf-8") as tf:
                            tf.write(trans)

                        manifest.append({"crop": crop_name, "transcription": trans, "sample_id": sid_str})
            except Exception as e:
                logger.warning("crop fail %s: %s", label_json_path, e)

        manifest_path = os.path.join(ds_path, "ocr_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        return {"model_id": model_id, "crops_dir": os.path.abspath(crops_dir), "manifest": os.path.abspath(manifest_path), "crop_count": len(manifest)}

    else:
        return {"error": "unknown model_id", "model_id": model_id}

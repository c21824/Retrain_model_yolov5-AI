import asyncio
import sys
import os
import uuid
import shlex
import subprocess
import logging
import re
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from fastapi import FastAPI, Request, HTTPException

from exporter_model1 import create_dataset_for_detect1
from exporter_model2 import create_dataset_for_detect2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrain_api")

app = FastAPI(title="Retrain API")

# tạo dict để theo dõi job
JOB_STATUS: Dict[str, str] = {}
JOB_MSG: Dict[str, str] = {}
JOB_LOG_PATH: Dict[str, str] = {}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PY = PROJECT_ROOT / "app" / "yolov5" / "train.py"
DEFAULT_VAL_PY = DEFAULT_TRAIN_PY.parent / "val.py"
if not DEFAULT_TRAIN_PY.exists():
    logger.warning("train.py not found at %s - adjust DEFAULT_TRAIN_PY", DEFAULT_TRAIN_PY)
if not DEFAULT_VAL_PY.exists():
    logger.info("val.py not found at %s - val step may fail if val.py missing", DEFAULT_VAL_PY)

FALLBACK_WEIGHTS = "yolov5s.pt"


def start_background_training(cmd_list: list, job_id: str, log_file: Path, cwd: Optional[Path] = None):
    JOB_STATUS[job_id] = "running"
    JOB_MSG[job_id] = "started"

    def _runner():
        try:
            logger.info("[%s] running: %s", job_id, " ".join(shlex.quote(p) for p in cmd_list))
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write("COMMAND: " + " ".join(shlex.quote(p) for p in cmd_list) + "\n\n")
                lf.flush()
                #Chạy subprocess, đọc stdout streaming log_file
                proc = subprocess.Popen(cmd_list, stdout=lf, stderr=subprocess.STDOUT, text=True,
                                        cwd=str(cwd) if cwd else None)
                rc = proc.wait()#dừng chặn luồng cho đến khi thực hiện xong
                if rc == 0: #luồng thực hiện thành công
                    JOB_STATUS[job_id] = "finished"
                    JOB_MSG[job_id] = f"finished (rc=0)"
                else:
                    JOB_STATUS[job_id] = "failed"
                    JOB_MSG[job_id] = f"exit_code={rc}"
                logger.info("[%s] finished with rc=%s", job_id, rc)
        except Exception as e:
            logger.exception("[%s] training failed", job_id)
            JOB_STATUS[job_id] = "failed"
            JOB_MSG[job_id] = str(e)
    import threading
    t = threading.Thread(target=_runner, name=f"train-{job_id}", daemon=True)
    t.start()


def _run_subprocess_collect(cmd_list: List[str], log_file: Optional[Path] = None,
                            stream_to_console: bool = False, cwd: Optional[Path] = None) -> Tuple[int, List[str]]:
    logger.info("Running subprocess: %s (cwd=%s)", " ".join(shlex.quote(p) for p in cmd_list), str(cwd) if cwd else "<cwd=None>")
    proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            cwd=str(cwd) if cwd else None)
    lines: List[str] = []
    lf = None
    try:
        if log_file:
            lf = open(log_file, "a", encoding="utf-8")
            lf.write("COMMAND: " + " ".join(shlex.quote(p) for p in cmd_list) + "\n\n")
            lf.flush()
        for ln in proc.stdout:
            ln = ln.rstrip("\n")
            lines.append(ln)
            if lf:
                lf.write(ln + "\n")
            if stream_to_console:
                print(ln)
    except Exception as e:
        logger.exception("Error while reading subprocess output: %s", e)
    finally:
        try:
            if lf:
                lf.close()
        except Exception:
            pass
    rc = proc.wait()
    return rc, lines

# tìm folder run gần nhất
def _find_latest_run_dir(project_arg: str) -> Optional[Path]:
    p = Path(project_arg)
    if p.exists():
        childs = [c for c in p.iterdir() if c.is_dir()]
        if childs:
            sorted_runs = sorted(childs, key=lambda x: x.stat().st_mtime, reverse=True)
            return sorted_runs[0].resolve()
    cand = Path.cwd() / "runs" / "train"
    if cand.exists():
        childs = [c for c in cand.iterdir() if c.is_dir()]
        if childs:
            return sorted(childs, key=lambda x: x.stat().st_mtime, reverse=True)[0].resolve()
    return None

# dùng để tìm lỗi
def _contains_detectionmodel_error(lines: List[str]) -> bool:
    text = "\n".join(lines[-200:])
    if "Can't get attribute 'DetectionModel'" in text or "AttributeError: Can't get attribute 'DetectionModel'" in text:
        return True
    return False

# đọc thông tin từ metric và lấy kết quả
def _read_metrics_from_results_csv_average(run_dir: Path) -> Dict[str, Optional[float]]:

    if not run_dir or not run_dir.exists():
        return {"precision": None, "recall": None}

    candidates: List[Path] = []
    for name in ("results.csv", "metrics.csv"):
        p = run_dir / name
        if p.exists():
            candidates.append(p)
    for p in run_dir.glob("*.csv"):
        if p not in candidates:
            candidates.append(p)
    rfolder = run_dir / "results"
    if rfolder.exists():
        for p in rfolder.glob("*.csv"):
            if p not in candidates:
                candidates.append(p)

    if not candidates:
        return {"precision": None, "recall": None}

    for csv_path in candidates:
        try:
            print(csv_path)
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as cf:
                reader = csv.DictReader(cf)
                if not reader.fieldnames:
                    continue
                fieldnames = [fn.strip() for fn in reader.fieldnames]
                norm_to_orig = {fn.strip(): fn for fn in reader.fieldnames}

                prec_col = None
                rec_col = None
                for orig in reader.fieldnames:
                    n = orig.strip().lower()
                    if "precision" in n and prec_col is None:
                        prec_col = orig
                    if "recall" in n and rec_col is None:
                        rec_col = orig
                if prec_col is None:
                    for orig in reader.fieldnames:
                        n = orig.strip().lower()
                        if n == "p" or n == "precision" or n == "prec":
                            prec_col = orig
                            break
                if rec_col is None:
                    for orig in reader.fieldnames:
                        n = orig.strip().lower()
                        if n == "r" or n == "recall" or n == "rec":
                            rec_col = orig
                            break

                if prec_col is None and rec_col is None:
                    continue

                prec_vals: List[float] = []
                rec_vals: List[float] = []
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as cf2:
                rdr = csv.DictReader(cf2)
                for row in rdr:
                    if prec_col:
                        rawp = row.get(prec_col, "") or row.get(prec_col.strip(), "") or ""
                        rawp = str(rawp).strip()
                        if rawp != "":
                            try:
                                pv = float(rawp)
                                if pv > 1.0:
                                    pv = pv / 100.0
                                prec_vals.append(pv)
                            except Exception:
                                try:
                                    pv = float(rawp.replace(",", ""))
                                    if pv > 1.0:
                                        pv = pv / 100.0
                                    prec_vals.append(pv)
                                except Exception:
                                    pass
                    if rec_col:
                        rawr = row.get(rec_col, "") or row.get(rec_col.strip(), "") or ""
                        rawr = str(rawr).strip()
                        if rawr != "":
                            try:
                                rv = float(rawr)
                                if rv > 1.0:
                                    rv = rv / 100.0
                                rec_vals.append(rv)
                            except Exception:
                                try:
                                    rv = float(rawr.replace(",", ""))
                                    if rv > 1.0:
                                        rv = rv / 100.0
                                    rec_vals.append(rv)
                                except Exception:
                                    pass

                avg_p = None
                avg_r = None
                if prec_vals:
                    avg_p = sum(prec_vals) / len(prec_vals)
                if rec_vals:
                    avg_r = sum(rec_vals) / len(rec_vals)

                if avg_p is not None:
                    avg_p = round(avg_p*100, 2)
                if avg_r is not None:
                    avg_r = round(avg_r*100, 2)

                return {"precision": avg_p, "recall": avg_r}
        except Exception as e:
            logger.exception("Failed to read CSV %s: %s", csv_path, e)
            continue

    return {"precision": None, "recall": None}


def finalize_metrics(precision: Optional[float], recall: Optional[float]) -> Dict[str, Optional[float]]:
    res: Dict[str, Optional[float]] = {"precision": None, "recall": None, "f1": None, "accuracy": None}
    if precision is not None:
        res["precision"] = float(round(precision, 2))
    if recall is not None:
        res["recall"] = float(round(recall, 2))

    if precision is not None and recall is not None:
        p = float(precision)
        r = float(recall)
        if (p + r) > 0:
            f1 = 2.0 * p * r / (p + r)
            res["f1"] = float(round(f1, 2))
        else:
            res["f1"] = 0.0
        res["accuracy"] = float(round((p + r) / 2.0, 2))
    return res

# xử lý chính
@app.post("/api/retrain_detect")
async def retrain_detect(req: Request):

    data = await req.json()
    mode_id = int(data.get("model_id"))

    try:
        dataset_id = int(data.get("dataset_id"))
    except Exception:
        raise HTTPException(status_code=400, detail="dataset_id required and must be int")

    mode = data.get("mode", "sync").lower()
    epochs = 10
    batch = 16
    img = 640
    weights = data.get("weights")
    cfg = "models/yolov5s.yaml"
    name = f"finetune_{dataset_id}"
    train_py_path = DEFAULT_TRAIN_PY
    val_py_path = DEFAULT_VAL_PY

    output_base = data.get("output_base")
    #tạo yaml file và dataset
    if mode_id == 1:
        try:
            create_res = await asyncio.to_thread(create_dataset_for_detect1, dataset_id, output_base, data.get("val_images_dir"))
        except Exception as e:
            logger.exception("create_dataset failed")
            raise HTTPException(status_code=500, detail="create_dataset failed: " + str(e))
    else:
        try:
            create_res = await asyncio.to_thread(create_dataset_for_detect2, dataset_id, output_base, data.get("val_images_dir"))
        except Exception as e:
            logger.exception("create_dataset failed")
            raise HTTPException(status_code=500, detail="create_dataset failed: " + str(e))

    yaml_path = Path(create_res["yaml"])
    ds_path = Path(create_res["dataset_path"])

    project_arg = str((ds_path / "runs").resolve())
    name_arg = name + "_" + uuid.uuid4().hex[:8]
    python_cmd = sys.executable or "python"
    train_cwd = train_py_path.parent if train_py_path.exists() else None

    cmd_list = [
        python_cmd,
        str(train_py_path),
        "--img", str(img),
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--data", str(yaml_path.resolve()),
        "--cfg", cfg,
        "--weights", str(weights),
        "--project", project_arg,
        "--name", name_arg,
        "--exist-ok"
    ]
# kiểm tra file train.py tồn lại, viết log file và trả lại json kết thúc huấn luyện
    if mode == "sync":
        #kiểm tra file train.py
        if not train_py_path.exists():
            raise HTTPException(status_code=500, detail=f"train.py not found at {train_py_path}")

        logs_dir = ds_path / "train_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        train_log = logs_dir / f"train_{name_arg}_{uuid.uuid4().hex[:6]}.log"

        def _run_train_and_handle_retry():
            rc, out_lines = _run_subprocess_collect(cmd_list, log_file=train_log, stream_to_console=False, cwd=train_cwd)
            #hoàn thành xử lý
            if rc == 0:
                run_dir = _find_latest_run_dir(project_arg)
                return run_dir, out_lines, weights, False
            #nễu hỏng chạy lại với fallback weight
            if _contains_detectionmodel_error(out_lines):
                logger.warning("Detected DetectionModel unpickle error; retrying with fallback weights: %s", FALLBACK_WEIGHTS)
                with open(train_log, "a", encoding="utf-8") as lf:
                    lf.write("\n\n--- DetectionModel error detected; retrying with fallback weights: %s ---\n\n" % FALLBACK_WEIGHTS)
                retry = list(cmd_list)
                try:
                    idx = retry.index("--weights")
                    retry[idx + 1] = str(FALLBACK_WEIGHTS)
                except ValueError:
                    retry += ["--weights", str(FALLBACK_WEIGHTS)]
                rc2, out_lines2 = _run_subprocess_collect(retry, log_file=train_log, stream_to_console=False, cwd=train_cwd)
                if rc2 == 0:
                    run_dir = _find_latest_run_dir(project_arg)
                    return run_dir, out_lines + ["--- RETRY OUTPUT ---"] + out_lines2, str(FALLBACK_WEIGHTS), True
                else:
                    raise RuntimeError(f"train.py failed twice (rcs {rc} and {rc2}). See log {train_log}")
            else:
                raise RuntimeError(f"train.py failed (rc={rc}). See log {train_log}")

        try:
            run_dir, train_out_lines, used_weights, used_fallback = await asyncio.to_thread(_run_train_and_handle_retry)
        except Exception as ex:
            logger.exception("sync train subprocess failed")
            raise HTTPException(status_code=500, detail="sync train failed: " + str(ex))
        #lấy thông tin metrics
        precision = None
        recall = None
        if run_dir:
            csv_metrics = _read_metrics_from_results_csv_average(run_dir)
            print(csv_metrics)
            precision = csv_metrics.get("precision")
            recall = csv_metrics.get("recall")

        final = finalize_metrics(precision, recall)

        return {
            "status": "ok",
            "mode": "sync",
            "dataset_id": dataset_id,
            "dataset_path": str(ds_path),
            "yaml": str(yaml_path.resolve()),
            "run_dir": str(run_dir) if run_dir else None,
            "weights": str(run_dir) + "\weights\\best.pt",
            "metrics": final,
            "note": ("retried_with_fallback" if used_fallback else None)
        }
    
    elif mode == "background":
        if not train_py_path.exists():
            raise HTTPException(status_code=500, detail=f"train.py not found at {train_py_path}")

        logs_dir = ds_path / "train_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        job_id = "job-" + uuid.uuid4().hex[:12]
        log_file = logs_dir / f"{name_arg}_{job_id}.log"

        JOB_STATUS[job_id] = "queued"
        JOB_MSG[job_id] = "queued"
        JOB_LOG_PATH[job_id] = str(log_file.resolve())
        start_background_training(cmd_list, job_id, log_file, cwd=train_cwd)

        return {"status": "ok", "mode": "background", "job_id": job_id, "yaml": str(yaml_path.resolve()), "log": JOB_LOG_PATH[job_id]}
    else:
        raise HTTPException(status_code=400, detail="Unknown mode, allowed: sync | background")


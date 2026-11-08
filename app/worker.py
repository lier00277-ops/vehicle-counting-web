"""后台 worker：从 Redis 队列取任务，跑两套方案，写回结果。"""
from __future__ import annotations

import base64
import json
import os
import time

import cv2
import numpy as np

from app import queue
from app.config import settings
from app.db import Base, SessionLocal, engine
from app.models import Prediction
from app.services import detector, vlm

Base.metadata.create_all(bind=engine)


def _save_record(rec: Prediction) -> Prediction:
    db = SessionLocal()
    try:
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
    finally:
        db.close()


def _save_image(job_id: str, image_base64: str) -> str:
    """把标注图落到 data/results/{job_id}.jpg，返回路径。"""
    results_dir = settings.upload_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    img_file = results_dir / f"{job_id}.jpg"
    img_file.write_bytes(base64.b64decode(image_base64))
    return str(img_file)


def process(job: dict) -> None:
    job_id = job["job_id"]
    payload = job["payload"]
    path = payload["path"]

    # 崩溃恢复会把已完成的任务重新入队，这里幂等跳过
    cached = queue.get_result(job_id)
    if cached and cached.get("status") == "done":
        queue.ack(job)
        return

    queue.set_result(job_id, {"status": "running"})

    try:
        data = open(path, "rb").read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("图片解码失败，文件可能不是有效的图片")

        yolo_res = detector.analyze(img)

        vlm_error = None
        try:
            vlm_res = vlm.analyze(data)
        except RuntimeError as e:
            vlm_res = {"raw": "", "latency_ms": 0}
            vlm_error = str(e)
        parsed = vlm_res.get("parsed")
        if vlm_error is None:
            if parsed is None:
                vlm_error = "大模型返回内容无法解析为 JSON"
            else:
                vlm_error = vlm.validate(parsed)
                if vlm_error is None:
                    parsed = vlm.normalize(parsed)

        status = "done" if vlm_error is None else "partial"
        image_path = _save_image(job_id, yolo_res["image_base64"]) if yolo_res.get("image_base64") else ""

        rec = _save_record(Prediction(
            job_id=job_id,
            filename=payload.get("filename", ""),
            status=status,
            error=vlm_error or "",
            yolo_total=yolo_res["total"],
            yolo_counts=json.dumps(yolo_res["counts"], ensure_ascii=False),
            vlm_total=int(parsed["total"]) if parsed else None,
            vlm_counts=json.dumps(parsed, ensure_ascii=False) if parsed else "{}",
            vlm_raw=vlm_res.get("raw", ""),
            image_path=image_path,
            yolo_ms=yolo_res["latency_ms"],
            vlm_ms=vlm_res["latency_ms"],
        ))

        queue.set_result(job_id, {
            "status": "done",
            "id": rec.id,
            "record_status": status,
            "filename": payload.get("filename", ""),
            "traditional": {
                "counts": yolo_res["counts"],
                "total": yolo_res["total"],
                "image_base64": yolo_res["image_base64"],
                "latency_ms": yolo_res["latency_ms"],
            },
            "vlm": {
                "counts": parsed or {},
                "raw": vlm_res.get("raw", ""),
                "latency_ms": vlm_res["latency_ms"],
                "error": vlm_error,
            },
        })
    except Exception as e:  # noqa: BLE001
        _save_record(Prediction(
            job_id=job_id,
            filename=payload.get("filename", ""),
            status="failed",
            error=str(e),
        ))
        queue.set_result(job_id, {"status": "error", "error": str(e)})
    finally:
        queue.ack(job)
        try:
            os.remove(path)
        except OSError:
            pass


def main() -> None:
    try:
        n = queue.recover()
        if n:
            print(f"recovered {n} unfinished job(s) from previous run")
    except Exception as e:  # noqa: BLE001
        print("recover failed:", e)
    print("worker started, waiting for jobs...")
    while True:
        try:
            job = queue.pop()
        except Exception as e:  # noqa: BLE001
            print("redis error, retry:", e)
            time.sleep(2)
            continue
        if job:
            process(job)


if __name__ == "__main__":
    main()

"""后台 worker：从 Redis 队列取任务，跑两套方案，写回结果。"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from app import queue
from app.db import Base, SessionLocal, engine
from app.models import Prediction
from app.services import detector, vlm

Base.metadata.create_all(bind=engine)


def process(job: dict) -> None:
    job_id = job["job_id"]
    payload = job["payload"]
    path = payload["path"]
    queue.set_result(job_id, {"status": "running"})

    try:
        data = open(path, "rb").read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        yolo_res = detector.analyze(img)
        try:
            vlm_res = vlm.analyze(data)
        except RuntimeError as e:
            vlm_res = {"raw": "", "parsed": None, "latency_ms": 0, "error": str(e)}

        parsed = vlm_res.get("parsed") or {}
        db = SessionLocal()
        rec = Prediction(
            filename=payload.get("filename", ""),
            yolo_total=yolo_res["total"],
            yolo_counts=json.dumps(yolo_res["counts"], ensure_ascii=False),
            vlm_total=parsed.get("total") if parsed else None,
            vlm_raw=json.dumps(parsed, ensure_ascii=False) if parsed else vlm_res.get("raw", ""),
            yolo_ms=yolo_res["latency_ms"],
            vlm_ms=vlm_res["latency_ms"],
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        db.close()

        queue.set_result(job_id, {
            "status": "done",
            "id": rec.id,
            "filename": payload.get("filename", ""),
            "traditional": {
                "counts": yolo_res["counts"],
                "total": yolo_res["total"],
                "image_base64": yolo_res["image_base64"],
                "latency_ms": yolo_res["latency_ms"],
            },
            "vlm": {
                "counts": parsed,
                "raw": vlm_res.get("raw", ""),
                "latency_ms": vlm_res["latency_ms"],
                "error": vlm_res.get("error"),
            },
        })
    except Exception as e:  # noqa: BLE001
        queue.set_result(job_id, {"status": "error", "error": str(e)})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main() -> None:
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

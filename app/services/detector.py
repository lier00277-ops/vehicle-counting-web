"""YOLO 检测 + SAM 分割。"""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

from app.config import settings

VEH = {"car", "bus", "truck", "motorcycle"}
_models: dict = {}


def get_models():
    if "yolo" not in _models:
        _models["yolo"] = YOLO(settings.yolo_weights)
        sam = sam_model_registry["vit_b"](checkpoint=settings.sam_weights)
        _models["sam"] = SamPredictor(sam)
    return _models["yolo"], _models["sam"]


def analyze(image_bgr: np.ndarray) -> dict:
    yolo, sam = get_models()
    t0 = time.time()

    result = yolo(image_bgr, conf=0.25, verbose=False)[0]
    dets = []
    for i in range(len(result.boxes)):
        name = yolo.names[int(result.boxes.cls[i])]
        if name not in VEH:
            continue
        dets.append({
            "class": name,
            "confidence": round(float(result.boxes.conf[i]), 3),
            "box": [round(float(v), 1) for v in result.boxes.xyxy[i]],
        })

    vis = result.orig_img.copy()
    if dets:
        sam.set_image(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
        for d in dets:
            masks, _, _ = sam.predict(box=np.array(d["box"]), multimask_output=False)
            d["mask_area"] = int(masks[0].sum())
            vis[masks[0]] = (vis[masks[0]] * 0.55 + np.array([255, 80, 0]) * 0.45).astype(np.uint8)

    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, f'{d["class"]} {d["confidence"]:.2f}', (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    counts: dict[str, int] = {}
    for d in dets:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    ok, buf = cv2.imencode(".jpg", vis)
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "detections": dets,
        "image_base64": base64.b64encode(buf).decode() if ok else "",
        "latency_ms": int((time.time() - t0) * 1000),
    }

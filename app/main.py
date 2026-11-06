"""车辆清点对比 Web API。"""
from __future__ import annotations

import json

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import create_token, current_user, verify_login
from app.db import Base, engine, get_db
from app.models import Prediction
from app.services import detector, vlm

app = FastAPI(title="车辆清点对比 API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Base.metadata.create_all(bind=engine)


class LoginIn(BaseModel):
    username: str
    password: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/login")
def login(body: LoginIn):
    verify_login(body.username, body.password)
    return {"token": create_token(body.username), "username": body.username}


@app.post("/api/predict")
def predict(file: UploadFile = File(...), user: str = Depends(current_user),
            db: Session = Depends(get_db)):
    data = file.file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="图片无法读取")

    yolo_res = detector.analyze(img)
    try:
        vlm_res = vlm.analyze(data)
    except RuntimeError as e:
        vlm_res = {"raw": "", "parsed": None, "latency_ms": 0, "error": str(e)}

    parsed = vlm_res.get("parsed") or {}
    rec = Prediction(
        filename=file.filename or "",
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

    return {
        "id": rec.id,
        "filename": rec.filename,
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
    }


@app.get("/api/history")
def history(limit: int = 20, user: str = Depends(current_user),
            db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
    return [{
        "id": r.id,
        "filename": r.filename,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "yolo_total": r.yolo_total,
        "vlm_total": r.vlm_total,
        "yolo_ms": r.yolo_ms,
        "vlm_ms": r.vlm_ms,
    } for r in rows]

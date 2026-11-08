"""车辆清点对比 Web API。"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import queue
from app.auth import create_token, current_user, verify_login
from app.config import settings
from app.db import Base, engine, get_db
from app.models import Prediction

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
def predict(file: UploadFile = File(...), user: str = Depends(current_user)):
    job_id = uuid4().hex
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"{job_id}{suffix}"
    path.write_bytes(file.file.read())

    queue.enqueue(job_id, {"path": str(path), "filename": file.filename or ""})
    queue.set_result(job_id, {"status": "pending"})
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, user: str = Depends(current_user)):
    result = queue.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return result


def _record_dict(r: Prediction) -> dict:
    return {
        "id": r.id,
        "job_id": r.job_id,
        "filename": r.filename,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "status": r.status,
        "error": r.error,
        "yolo_total": r.yolo_total,
        "yolo_counts": json.loads(r.yolo_counts or "{}"),
        "vlm_total": r.vlm_total,
        "vlm_counts": json.loads(r.vlm_counts or "{}"),
        "yolo_ms": r.yolo_ms,
        "vlm_ms": r.vlm_ms,
    }


@app.get("/api/history")
def history(limit: int = 20, user: str = Depends(current_user),
            db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
    return [_record_dict(r) for r in rows]


@app.get("/api/history/{record_id}")
def history_detail(record_id: int, user: str = Depends(current_user),
                   db: Session = Depends(get_db)):
    r = db.get(Prediction, record_id)
    if r is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    out = _record_dict(r)
    out["vlm_raw"] = r.vlm_raw
    image_base64 = ""
    if r.image_path and Path(r.image_path).is_file():
        image_base64 = base64.b64encode(Path(r.image_path).read_bytes()).decode()
    out["image_base64"] = image_base64
    return out

"""数据库表。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(32), unique=True, index=True, nullable=True)
    filename = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="done")  # done=两套都正常 partial=大模型失败/校验不过 failed=任务失败
    error = Column(Text, default="")
    yolo_total = Column(Integer, default=0)
    yolo_counts = Column(Text, default="{}")
    vlm_total = Column(Integer, nullable=True)
    vlm_counts = Column(Text, default="{}")
    vlm_raw = Column(Text, default="")
    image_path = Column(String(512), default="")
    yolo_ms = Column(Float, default=0)
    vlm_ms = Column(Float, default=0)

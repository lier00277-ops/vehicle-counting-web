"""数据库表。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    yolo_total = Column(Integer, default=0)
    yolo_counts = Column(Text, default="{}")
    vlm_total = Column(Integer, nullable=True)
    vlm_raw = Column(Text, default="")
    yolo_ms = Column(Float, default=0)
    vlm_ms = Column(Float, default=0)

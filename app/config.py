"""从环境变量读取配置。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class Settings:
    vlm_api_key: str = os.getenv("VLM_API_KEY", "")
    vlm_base_url: str = os.getenv("VLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
    vlm_model: str = os.getenv("VLM_MODEL", "glm-5.3-flashx")

    jwt_secret: str = os.getenv("JWT_SECRET", "dev-secret-change-me")
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))

    app_user: str = os.getenv("APP_USER", "demo")
    app_password_hash: str = os.getenv(
        "APP_PASSWORD_HASH", hashlib.sha256(os.getenv("APP_PASSWORD", "demo123").encode()).hexdigest()
    )

    yolo_weights: str = os.getenv("YOLO_WEIGHTS", r"D:\stage2\weights\yolo26n.pt")
    sam_weights: str = os.getenv("SAM_WEIGHTS", r"D:\stage2\weights\sam_vit_b_01ec64.pth")

    db_url: str = os.getenv("DB_URL", f"sqlite:///{(ROOT / 'data' / 'app.db').as_posix()}")
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(ROOT / "data" / "uploads")))


settings = Settings()

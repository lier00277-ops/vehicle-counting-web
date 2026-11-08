"""调用智谱 GLM 视觉模型。"""
from __future__ import annotations

import base64
import json
import re
import time

import requests

from app.config import settings

PROMPT = """你是车辆清点助手。请清点图中所有机动车（car、bus、truck、motorcycle），
不要统计非机动车和行人。只输出 JSON，格式如下，不要输出其他内容：
{"cars": 数字, "buses": 数字, "trucks": 数字, "motorcycles": 数字, "total": 数字, "description": "一句话描述场景"}"""


COUNT_KEYS = ("cars", "buses", "trucks", "motorcycles")


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate(parsed) -> str | None:
    """校验大模型返回的结构化结果，返回 None 表示通过，否则返回失败原因。"""
    if not isinstance(parsed, dict):
        return "返回内容不是 JSON 对象"
    for key in COUNT_KEYS + ("total",):
        if key not in parsed:
            return f"缺少字段 {key}"
        v = parsed[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or v < 0:
            return f"字段 {key}={v!r} 不是非负整数"
    subtotal = sum(int(parsed[k]) for k in COUNT_KEYS)
    if int(parsed["total"]) != subtotal:
        return f"total={int(parsed['total'])} 与分类之和 {subtotal} 不一致"
    return None


def normalize(parsed: dict) -> dict:
    """把数值统一转成 int，去掉多余的小数形式。"""
    out = dict(parsed)
    for key in COUNT_KEYS + ("total",):
        out[key] = int(out[key])
    return out


def analyze(image_bytes: bytes, timeout: int = 120, retries: int = 1) -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": settings.vlm_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {settings.vlm_api_key}", "Content-Type": "application/json"}

    t0 = time.time()
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{settings.vlm_base_url}/chat/completions", json=payload,
                              headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return {
                "raw": content,
                "parsed": parse_json(content),
                "latency_ms": int((time.time() - t0) * 1000),
                "usage": data.get("usage", {}),
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"大模型调用失败: {last_err}")

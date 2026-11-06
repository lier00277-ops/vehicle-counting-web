"""Redis 任务队列和结果缓存。"""
from __future__ import annotations

import json
import os

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_KEY = "stage3:jobs"
RESULT_PREFIX = "stage3:result:"
RESULT_TTL = 3600

_client = redis.from_url(REDIS_URL, decode_responses=True)


def enqueue(job_id: str, payload: dict) -> None:
    _client.rpush(QUEUE_KEY, json.dumps({"job_id": job_id, "payload": payload}, ensure_ascii=False))


def pop(timeout: int = 5):
    try:
        item = _client.blpop(QUEUE_KEY, timeout=timeout)
    except redis.exceptions.TimeoutError:
        return None
    return json.loads(item[1]) if item else None


def set_result(job_id: str, data: dict) -> None:
    _client.set(RESULT_PREFIX + job_id, json.dumps(data, ensure_ascii=False), ex=RESULT_TTL)


def get_result(job_id: str):
    raw = _client.get(RESULT_PREFIX + job_id)
    return json.loads(raw) if raw else None

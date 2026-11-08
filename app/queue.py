"""Redis 任务队列和结果缓存。"""
from __future__ import annotations

import json
import os

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_KEY = "stage3:jobs"
PROCESSING_KEY = "stage3:processing"
RESULT_PREFIX = "stage3:result:"
RESULT_TTL = 3600

_client = redis.from_url(REDIS_URL, decode_responses=True)


def enqueue(job_id: str, payload: dict) -> None:
    _client.rpush(QUEUE_KEY, json.dumps({"job_id": job_id, "payload": payload}, ensure_ascii=False))


def pop(timeout: int = 5):
    """取任务，同时把任务放进 processing 列表；处理完必须调用 ack 确认。"""
    try:
        raw = _client.brpoplpush(QUEUE_KEY, PROCESSING_KEY, timeout=timeout)
    except redis.exceptions.TimeoutError:
        return None
    if not raw:
        return None
    job = json.loads(raw)
    job["_raw"] = raw
    return job


def ack(job: dict) -> None:
    """任务处理完毕（无论成败），从 processing 列表移除。"""
    raw = job.get("_raw")
    if raw:
        _client.lrem(PROCESSING_KEY, 1, raw)


def recover() -> int:
    """把上次崩溃残留在 processing 列表里的任务挪回队列，返回恢复数量。"""
    moved = 0
    while True:
        item = _client.lmove(PROCESSING_KEY, QUEUE_KEY, "RIGHT", "LEFT")
        if item is None:
            break
        moved += 1
    return moved


def set_result(job_id: str, data: dict) -> None:
    _client.set(RESULT_PREFIX + job_id, json.dumps(data, ensure_ascii=False), ex=RESULT_TTL)


def get_result(job_id: str):
    raw = _client.get(RESULT_PREFIX + job_id)
    return json.loads(raw) if raw else None

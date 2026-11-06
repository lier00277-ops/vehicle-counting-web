"""Streamlit 前端：登录后上传图片，并排展示两套方案。"""
import base64
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_NAME = os.getenv("VLM_MODEL", "glm-5.3-flashx")

st.set_page_config(page_title="车辆清点对比", layout="wide")
st.title("车辆清点：YOLO+SAM vs 大模型")

if "token" not in st.session_state:
    st.session_state.token = None

with st.sidebar:
    st.subheader("登录")
    if st.session_state.token is None:
        user = st.text_input("用户名", value="demo")
        pwd = st.text_input("密码", type="password", value="")
        if st.button("登录"):
            r = requests.post(f"{API_BASE}/api/login",
                              json={"username": user, "password": pwd}, timeout=10)
            if r.status_code == 200:
                st.session_state.token = r.json()["token"]
                st.rerun()
            else:
                st.error("登录失败")
    else:
        st.success("已登录")
        if st.button("退出"):
            st.session_state.token = None
            st.rerun()

if st.session_state.token is None:
    st.info("先在左侧登录")
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}

up = st.file_uploader("上传一张道路/街景图片", type=["jpg", "jpeg", "png"])
if up:
    r = requests.post(
        f"{API_BASE}/api/predict",
        files={"file": (up.name, up.getvalue(), up.type or "image/jpeg")},
        headers=headers,
        timeout=30,
    )
    if r.status_code != 200:
        st.error(f"提交失败：{r.status_code} {r.text}")
        st.stop()

    job_id = r.json()["job_id"]
    box = st.empty()
    data = None
    for _ in range(180):
        jr = requests.get(f"{API_BASE}/api/jobs/{job_id}", headers=headers, timeout=10)
        result = jr.json()
        if result.get("status") in ("done", "error"):
            data = result
            break
        box.info(f"后台处理中（{result.get('status', 'pending')}）…")
        time.sleep(1)
    box.empty()

    if data is None:
        st.error("超时，任务还没有结果")
        st.stop()
    if data.get("status") == "error":
        st.error(f"处理失败：{data.get('error')}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("传统方案：YOLO + SAM")
        b64 = data["traditional"]["image_base64"]
        if b64:
            st.image(base64.b64decode(b64), use_container_width=True)
        st.write(f"检出 **{data['traditional']['total']}** 辆：{data['traditional']['counts']}")
        st.caption(f"耗时 {data['traditional']['latency_ms'] / 1000:.1f}s")

    with col2:
        st.subheader(f"大模型：{MODEL_NAME}")
        v = data["vlm"]
        if v.get("error"):
            st.error(v["error"])
        else:
            counts = v.get("counts") or {}
            st.write(f"识别 **{counts.get('total', '?')}** 辆")
            st.json(counts)
            st.caption(f"耗时 {v['latency_ms'] / 1000:.1f}s")
            with st.expander("原始返回"):
                st.code(v["raw"])

    with st.expander("历史记录"):
        r = requests.get(f"{API_BASE}/api/history", headers=headers, timeout=10)
        if r.status_code == 200:
            st.table(r.json())

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

for key, default in ("token", None), ("file_key", None), ("job_id", None), ("result", None):
    if key not in st.session_state:
        st.session_state[key] = default

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


def show_result(data: dict) -> None:
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
            st.error(f"大模型结果不可用：{v['error']}")
        else:
            counts = v.get("counts") or {}
            st.write(f"识别 **{counts.get('total', '?')}** 辆")
            st.json(counts)
        st.caption(f"耗时 {v['latency_ms'] / 1000:.1f}s")
        with st.expander("原始返回"):
            st.code(v.get("raw", ""))


up = st.file_uploader("上传一张道路/街景图片", type=["jpg", "jpeg", "png"])
if up:
    file_key = f"{up.name}:{up.size}"
    if st.session_state.file_key != file_key:
        # 换了新图片才提交新任务；页面重跑/重复上传同一张图会复用旧 job_id，不重复扣额度
        r = requests.post(
            f"{API_BASE}/api/predict",
            files={"file": (up.name, up.getvalue(), up.type or "image/jpeg")},
            headers=headers,
            timeout=30,
        )
        if r.status_code != 200:
            st.error(f"提交失败：{r.status_code} {r.text}")
            st.stop()
        st.session_state.file_key = file_key
        st.session_state.job_id = r.json()["job_id"]
        st.session_state.result = None

    job_id = st.session_state.job_id
    st.caption(f"任务编号：`{job_id}`（重复上传同一张图片会复用该任务，不重复消耗接口额度）")

    data = st.session_state.result
    if data is None:
        box = st.empty()
        for _ in range(180):
            jr = requests.get(f"{API_BASE}/api/jobs/{job_id}", headers=headers, timeout=10)
            result = jr.json()
            if result.get("status") in ("done", "error"):
                data = result
                st.session_state.result = data
                break
            box.info(f"后台处理中（{result.get('status', 'pending')}）…")
            time.sleep(1)
        box.empty()

    if data is None:
        st.error("超时，任务还没有结果，稍后可重新打开页面用任务编号等待结果")
        st.stop()
    if data.get("status") == "error":
        st.error(f"处理失败：{data.get('error')}")
        st.stop()
    show_result(data)

with st.expander("历史记录", expanded=not up):
    r = requests.get(f"{API_BASE}/api/history", headers=headers, timeout=10)
    if r.status_code == 200:
        rows = r.json()
        if not rows:
            st.caption("暂无记录")
        else:
            st.table([{
                "id": row["id"],
                "时间": row["created_at"],
                "文件": row["filename"],
                "状态": row["status"],
                "YOLO": row["yolo_total"],
                "大模型": row["vlm_total"],
                "YOLO耗时ms": row["yolo_ms"],
                "大模型耗时ms": row["vlm_ms"],
            } for row in rows])

            sel = st.selectbox("查看完整结果", ["（不选择）"] + [row["id"] for row in rows])
            if sel != "（不选择）":
                d = requests.get(f"{API_BASE}/api/history/{sel}", headers=headers, timeout=10).json()
                st.write(
                    f"任务编号 `{d['job_id']}`，状态 **{d['status']}**"
                    + (f"，失败原因：{d['error']}" if d.get("error") else "")
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.write("YOLO + SAM")
                    if d.get("image_base64"):
                        st.image(base64.b64decode(d["image_base64"]), use_container_width=True)
                    st.json(d.get("yolo_counts") or {})
                with c2:
                    st.write(f"大模型：{MODEL_NAME}")
                    st.json(d.get("vlm_counts") or {})
                    if d.get("vlm_raw"):
                        with st.expander("原始返回"):
                            st.code(d["vlm_raw"])
    else:
        st.error(f"历史记录加载失败：{r.status_code}")

# 车辆清点对比 Web 应用

第二阶段对比项目的 Web 版本：上传一张道路图片，后端同时跑 YOLO+SAM 和大模型，并排返回结果，记录到数据库。

## 结构

```
stage3/
├── app/
│   ├── main.py          # FastAPI 接口
│   ├── config.py        # 环境变量
│   ├── db.py            # SQLAlchemy
│   ├── models.py        # 数据表
│   ├── auth.py          # JWT 登录
│   ├── queue.py         # Redis 队列
│   ├── worker.py        # 后台任务
│   └── services/
│       ├── detector.py  # YOLO + SAM
│       └── vlm.py       # 智谱 GLM
├── frontend/app.py      # Streamlit 页面
├── docker/              # Dockerfile
├── nginx/               # 反向代理配置
├── docker-compose.yml
└── requirements.txt
```

## 本地运行

```powershell
# 1. 启动 Redis
docker run -d --name stage3-redis -p 6379:6379 redis:7-alpine

# 2. 装依赖
uv venv D:\stage3\venv --python 3.12
uv pip install --python D:\stage3\venv\Scripts\python.exe -r requirements.txt
copy .env.example .env   # 填 VLM_API_KEY

# 3. 后端
D:\stage3\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# 4. worker（另开一个终端）
D:\stage3\venv\Scripts\python.exe -m app.worker

# 5. 前端（另开一个终端）
D:\stage3\venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

上传后接口先返回 `job_id`，前端轮询 `/api/jobs/{job_id}` 拿结果。

浏览器打开 http://127.0.0.1:8501，默认账号 `demo`，密码看 `.env`。

## 接口

- `POST /api/login`：登录拿 token
- `POST /api/predict`：上传图片，返回两套方案结果
- `GET /api/history`：历史记录

## Docker

```powershell
docker compose up -d --build
```

浏览器打开 http://localhost。

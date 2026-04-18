FROM python:3.12-slim

WORKDIR /app

# 系統依賴
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    poppler-utils \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安裝 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 只複製 pyproject.toml 先安裝依賴（利用 Docker cache）
COPY python/pyproject.toml ./python/pyproject.toml
RUN uv pip install --system ./python/

# 複製完整程式碼
COPY python/ ./python/
COPY sync/ ./sync/

CMD ["python", "-m", "sync.sync"]

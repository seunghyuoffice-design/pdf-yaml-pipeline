# PDF-YAML Pipeline
# Document conversion service (CUDA 12.6)

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python 3.11 및 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    wget \
    git \
    unzip \
    build-essential \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    fonts-nanum \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# Python dependencies
COPY requirements.txt .

# 1. pip 업그레이드 및 PyTorch 온라인 설치
RUN pip install --upgrade pip && \
    pip install --retries 3 --timeout 300 --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.9.1 torchvision==0.24.1

# 2. 나머지 의존성 설치
RUN pip install --retries 3 --timeout 120 -r requirements.txt && \
    pip install --force-reinstall --no-deps pypdfium2==5.3.0

# 데이터 디렉토리 생성
RUN mkdir -p /data/input /data/output /data/cache

# Application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# 환경변수 설정
ENV PYTHONPATH=/app \
    INPUT_DIR=/data/input \
    OUTPUT_DIR=/data/output \
    CACHE_DIR=/data/cache \
    LOG_LEVEL=INFO

# 헬스체크
HEALTHCHECK --interval=60s --timeout=30s --retries=3 \
    CMD python -c "from src.pipeline.parsers import ParserFactory; print('OK')" || exit 1

# 기본 실행 명령어 (파이프라인 실행)
CMD ["python", "-m", "src.pipeline.orchestrator"]

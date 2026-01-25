#!/bin/bash
# PDF-YAML Pipeline - Quick Setup Script
# 한 번에 모든 설정을 완료합니다.

set -e

echo "=========================================="
echo "  PDF-YAML Pipeline Setup"
echo "=========================================="
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Docker 확인
echo -n "Checking Docker... "
if command -v docker &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}NOT FOUND${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# 2. GPU 확인 (선택)
echo -n "Checking NVIDIA GPU... "
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}Found: ${GPU_NAME}${NC}"
    USE_GPU=true
else
    echo -e "${YELLOW}Not found (will use CPU mode)${NC}"
    USE_GPU=false
fi

# 3. .env 파일 생성
echo -n "Creating .env file... "
if [ ! -f .env ]; then
    cp .env.example .env
    if [ "$USE_GPU" = false ]; then
        # CPU 모드로 설정
        sed -i 's/PIPELINE_DEVICE=cuda/PIPELINE_DEVICE=cpu/' .env 2>/dev/null || \
        sed -i '' 's/PIPELINE_DEVICE=cuda/PIPELINE_DEVICE=cpu/' .env
    fi
    echo -e "${GREEN}Created${NC}"
else
    echo -e "${YELLOW}Already exists${NC}"
fi

# 4. 디렉토리 생성
echo -n "Creating data directories... "
mkdir -p data/output
echo -e "${GREEN}OK${NC}"

# 5. 샘플 PDF 다운로드 (선택)
echo ""
read -p "Download sample PDF for testing? (y/N): " download_sample
if [[ "$download_sample" =~ ^[Yy]$ ]]; then
    echo -n "Downloading sample PDF... "
    # Public domain PDF (UN Universal Declaration of Human Rights)
    curl -sL "https://www.un.org/en/about-us/universal-declaration-of-human-rights/udhr.pdf" \
        -o data/sample_udhr.pdf 2>/dev/null && \
        echo -e "${GREEN}Downloaded: data/sample_udhr.pdf${NC}" || \
        echo -e "${YELLOW}Failed (you can add your own PDFs to data/)${NC}"
fi

# 6. Docker 이미지 빌드
echo ""
echo "Building Docker image (this may take 10-20 minutes on first run)..."
docker build -t pdf-pipeline:latest . || {
    echo -e "${RED}Build failed!${NC}"
    echo "Try: docker build --no-cache -t pdf-pipeline:latest ."
    exit 1
}

echo ""
echo -e "${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Add PDF files to: ./data/"
echo "  2. Start the pipeline: docker compose up -d"
echo "  3. Monitor progress: docker compose logs -f"
echo ""
if [ "$USE_GPU" = false ]; then
    echo -e "${YELLOW}Note: Running in CPU mode (slower but works without GPU)${NC}"
fi

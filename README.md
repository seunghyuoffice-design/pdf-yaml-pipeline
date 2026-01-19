# PDF-YAML Pipeline

GPU-accelerated document processing pipeline that converts PDF/HWP documents to structured YAML format.

## Features

- **PDF Parsing**: High-quality PDF text extraction using Docling
- **HWP Support**: Korean HWP/HWPX document parsing
- **Scan Detection**: Automatic triage of scanned vs digital PDFs
- **GPU Acceleration**: CUDA-optimized processing with multi-GPU support
- **CPU Mode**: Works without GPU (slower but functional)
- **Redis Queue**: Distributed worker architecture for scalability
- **YAML Output**: Structured, LLM-friendly output format

## Quick Start

### Option 1: One-Click Setup (Recommended)

```bash
git clone https://github.com/seunghyuoffice-design/pdf-yaml-pipeline.git
cd pdf-yaml-pipeline
./setup.sh
```

The setup script will:
- Detect GPU availability (falls back to CPU mode if no GPU)
- Create configuration files
- Build the Docker image
- Optionally download a sample PDF

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/seunghyuoffice-design/pdf-yaml-pipeline.git
cd pdf-yaml-pipeline

# Create configuration
cp .env.example .env
mkdir -p data/output

# Build and start
docker build -t pdf-pipeline:latest .
docker compose up -d
```

## Usage

1. **Add PDFs**: Place PDF files in `./data/`
2. **Start Pipeline**: `docker compose up -d`
3. **Monitor**: `docker compose logs -f`
4. **Results**: YAML files appear in `./data/output/`

## Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `/data` | Input directory containing PDFs |
| `OUTPUT_PATH` | `/data/output` | Output directory for YAML files |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `DOCLING_DEVICE` | `cuda` | `cuda` or `cpu` |
| `MAX_PDF_PAGES` | `100` | Maximum pages to process per PDF |

### CPU Mode (No GPU Required)

If you don't have an NVIDIA GPU, set in `.env`:

```bash
DOCLING_DEVICE=cpu
```

Note: CPU mode is ~5-10x slower than GPU mode.

### Worker Configuration

Adjust workers and threads in `.env`:

```bash
# 워커당 스레드 수 (CPU 코어에 맞게)
THREADS_PER_WORKER=2

# 워커당 메모리 (8G, 12G, 16G)
WORKER_MEMORY=8G
```

**Worker count:**

```bash
# 워커 1개 (기본)
docker compose up -d

# 워커 2개 (처리량 2배)
docker compose --profile scale up -d
```

**Recommended settings:**

| RAM | GPU VRAM | Workers | Threads |
|-----|----------|---------|---------|
| 16GB | 8GB | 1 | 2 |
| 32GB | 12GB+ | 2 | 4 |
| 64GB+ | 24GB+ | 2 | 8 |

## Architecture

```
PDF/HWP → Triage → Parser → YAML Converter → Output
              ↓
         Scan PDFs → OCR Queue (optional)
```

### Workers

- **GPU Workers**: PDF parsing with Docling (CUDA)
- **Redis**: Job queue and state management
- **Queue Monitor**: Progress tracking

## Testing

```bash
# Install test dependencies
pip install pytest fakeredis

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_lock_operations.py -v
```

## Troubleshooting

### "CUDA not available" error

Set CPU mode in `.env`:
```bash
DOCLING_DEVICE=cpu
```

### Docker build fails

Try building without cache:
```bash
docker build --no-cache -t pdf-pipeline:latest .
```

### Out of memory

Reduce worker count in `docker-compose.yml` or increase `MAX_PDF_PAGES` limit:
```bash
MAX_PDF_PAGES=50
```

### Pipeline stuck / no output

Check Redis connection:
```bash
docker compose logs redis
docker compose restart
```

### Permission denied on data/

Fix permissions:
```bash
chmod -R 777 data/
```

## Requirements

- Docker (with Docker Compose)
- **With GPU**: NVIDIA GPU + NVIDIA Container Toolkit
- **Without GPU**: Works in CPU mode (slower)
- 8GB+ RAM recommended

## License

MIT License

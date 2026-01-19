# PDF-YAML Pipeline

GPU-accelerated document processing pipeline that converts PDF/HWP documents to structured YAML format.

## Features

- **PDF Parsing**: High-quality PDF text extraction using Docling
- **HWP Support**: Korean HWP/HWPX document parsing
- **Scan Detection**: Automatic triage of scanned vs digital PDFs
- **GPU Acceleration**: CUDA-optimized processing with multi-GPU support
- **Redis Queue**: Distributed worker architecture for scalability
- **YAML Output**: Structured, LLM-friendly output format

## Quick Start

```bash
# Build the Docker image
docker build -t pdf-pipeline:latest .

# Start the pipeline (with GPU)
docker compose up -d

# Initialize the queue with PDF files
docker compose run --rm queue-init

# Monitor progress
docker compose run --rm queue-monitor
```

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `/data` | Input directory containing PDFs |
| `OUTPUT_PATH` | `/data/output` | Output directory for YAML files |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `DOCLING_DEVICE` | `cuda` | `cuda` or `cpu` |

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

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (RTX 3060+ recommended)
- 12GB+ RAM per worker

## License

MIT License

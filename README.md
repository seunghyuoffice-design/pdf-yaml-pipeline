# PDF-YAML Pipeline

GPU-accelerated document processing pipeline that converts PDF/HWP documents to structured YAML format, optimized for LLM training data preparation.

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Format](#output-format)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **PDF Parsing**: High-quality PDF text extraction using pypdfium2 + PaddleOCR
- **HWP Support**: Korean HWP/HWPX document parsing
- **Scan Detection**: Automatic triage of scanned vs digital PDFs
- **GPU Acceleration**: CUDA-optimized processing with multi-GPU support
- **CPU Mode**: Works without GPU (slower but functional)
- **Table Extraction**: Structured table data with cell-level bounding boxes
- **Redis Queue**: Distributed worker architecture for scalability
- **Fault Tolerant**: Automatic retry, dead letter queue, lock management
- **YAML Output**: Structured, LLM-friendly output format

## System Requirements

### Minimum (CPU Mode)

| Component | Requirement |
|-----------|-------------|
| OS | Linux (Ubuntu 20.04+), macOS, Windows with WSL2 |
| Docker | 20.10+ with Docker Compose v2 |
| RAM | 8GB |
| Disk | 10GB free space |

### Recommended (GPU Mode)

| Component | Requirement |
|-----------|-------------|
| OS | Linux (Ubuntu 20.04+ recommended) |
| Docker | 20.10+ with Docker Compose v2 |
| NVIDIA Driver | 525+ |
| NVIDIA Container Toolkit | Installed and configured |
| GPU | RTX 3060 (12GB) or better |
| RAM | 16GB+ |
| Disk | 20GB free space |

### Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Option 1: One-Click Setup (Recommended)

```bash
git clone https://github.com/seunghyuoffice-design/pdf-yaml-pipeline.git
cd pdf-yaml-pipeline
./setup.sh
```

The setup script automatically:
1. Detects GPU availability (falls back to CPU mode if no GPU)
2. Creates `.env` configuration file
3. Creates `data/` and `data/output/` directories
4. Builds the Docker image (~10-20 min on first run)
5. Optionally downloads a sample PDF for testing

### Option 2: Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/seunghyuoffice-design/pdf-yaml-pipeline.git
cd pdf-yaml-pipeline

# 2. Create configuration
cp .env.example .env
mkdir -p data/output

# 3. (Optional) Edit .env for your environment
# For CPU mode: set PIPELINE_DEVICE=cpu

# 4. Build Docker image
docker build -t pdf-pipeline:latest .

# 5. Start the pipeline
docker compose up -d
```

## Configuration

### All Environment Variables

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| **Data Paths** | | |
| `DATA_PATH` | `./data` | Input directory for PDF/HWP files |
| `OUTPUT_PATH` | `./data/output` | Output directory for YAML files |
| **GPU Settings** | | |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID(s), e.g., `0` or `0,1` |
| `PIPELINE_DEVICE` | `cuda` | `cuda` for GPU, `cpu` for CPU-only |
| **Worker Settings** | | |
| `THREADS_PER_WORKER` | `2` | CPU threads per worker |
| `WORKER_MEMORY` | `8G` | Memory limit per worker |
| **Processing** | | |
| `MAX_PDF_PAGES` | `100` | Max pages per PDF (truncates larger docs) |
| `TIMEOUT` | `600` | Processing timeout in seconds |
| `MAX_RETRIES` | `1` | Retry count before moving to DLQ |
| `SAFE_MODE` | `true` | Skip problematic files instead of crashing |
| `OCR_ENABLED` | `false` | Enable OCR for scanned documents |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |
| **Redis** | | |
| `REDIS_HOST` | `redis` | Redis hostname (use default in Docker) |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | (empty) | Redis password (optional) |

### CPU Mode (No GPU Required)

```bash
# In .env:
PIPELINE_DEVICE=cpu
```

> **Note:** CPU mode is approximately 5-10x slower than GPU mode.

### Worker Configuration

**Adjust resources in `.env`:**

```bash
# Workers use this many CPU threads
THREADS_PER_WORKER=2

# Memory per worker
WORKER_MEMORY=8G
```

**Scale workers:**

```bash
# 1 worker (default, lower memory usage)
docker compose up -d

# 2 workers (2x throughput, requires more memory)
docker compose --profile scale up -d
```

**Recommended settings by system:**

| System RAM | GPU VRAM | Workers | Threads | Memory |
|------------|----------|---------|---------|--------|
| 8GB | None (CPU) | 1 | 2 | 4G |
| 16GB | 8GB | 1 | 2 | 8G |
| 32GB | 12GB+ | 2 | 4 | 12G |
| 64GB+ | 24GB+ | 2 | 8 | 16G |

## Usage

### Basic Workflow

```bash
# 1. Add PDF files to data directory
cp /path/to/documents/*.pdf ./data/

# 2. Start the pipeline
docker compose up -d

# 3. Initialize the queue (scans data/ for new files)
docker compose run --rm queue-init

# 4. Monitor progress
docker compose logs -f worker-0

# 5. Check results
ls ./data/output/
```

### Useful Commands

```bash
# View worker logs
docker compose logs -f worker-0

# Check queue status
docker compose run --rm queue-monitor

# Stop pipeline
docker compose down

# Restart with fresh state
docker compose down -v  # Warning: clears Redis data
docker compose up -d

# Add more files to running pipeline
cp more_files/*.pdf ./data/
docker compose run --rm queue-init
```

### Processing States

Files move through these states:

```
file:queue → file:processing → file:done
                    ↓
              file:failed (DLQ)
```

## Output Format

### YAML Structure

Each processed PDF creates a `.yaml` file:

```yaml
document:
  source_path: "example.pdf"
  format: "pdf"
  parser: "cpu_pdf_parser"
  text_extractor: "pypdfium2"
  page_count: 15
  original_pages: 15        # Total pages in original PDF
  truncated: false          # true if exceeded MAX_PDF_PAGES
  max_pages_limit: null     # Limit that caused truncation (if any)
  ocr_enabled: false
  table_extraction: true
  encrypted: false

content:
  paragraphs:
    - "First paragraph text extracted from the document..."
    - "Second paragraph continues here with more content..."
    - "Each paragraph is a separate list item."

tables:
  - page: 1
    bbox: [100, 200, 500, 400]  # [x1, y1, x2, y2]
    cells:
      - text: "Header 1"
        row: 0
        col: 0
        bbox: [100, 200, 200, 220]
        confidence: 0.95
      - text: "Value 1"
        row: 1
        col: 0
        bbox: [100, 220, 200, 240]
        confidence: 0.98

assets:
  images: []  # Image metadata if extracted
```

### Key Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `document.page_count` | int | Number of pages actually processed |
| `document.original_pages` | int | Total pages in source PDF |
| `document.truncated` | bool | `true` if document was cut at MAX_PDF_PAGES |
| `content.paragraphs` | list | Extracted text paragraphs in reading order |
| `tables[].cells` | list | Cell-level table data with positions |
| `tables[].cells[].confidence` | float | OCR confidence score (0.0-1.0) |
| `tables[].cells[].bbox` | list | Bounding box [x1, y1, x2, y2] in pixels |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF-YAML Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │  Input   │───▶│  Redis   │───▶│ Workers  │               │
│  │  (data/) │    │  Queue   │    │ (GPU/CPU)│               │
│  └──────────┘    └──────────┘    └────┬─────┘               │
│                                       │                      │
│                       ┌───────────────┴───────────────┐      │
│                       ▼                               ▼      │
│                ┌──────────┐                    ┌──────────┐  │
│                │  Triage  │                    │  Parser  │  │
│                │(PDF type)│                    │(Paddle)  │  │
│                └────┬─────┘                    └────┬─────┘  │
│                     │                               │        │
│              ┌──────┴──────┐                        │        │
│              ▼             ▼                        ▼        │
│        ┌─────────┐  ┌──────────┐            ┌──────────┐    │
│        │ Digital │  │ Scanned  │            │  YAML    │    │
│        │  PDF    │  │ (→ OCR)  │            │ Converter│    │
│        └─────────┘  └──────────┘            └────┬─────┘    │
│                                                  │           │
│                                                  ▼           │
│                                           ┌──────────┐       │
│                                           │  Output  │       │
│                                           │(data/out)│       │
│                                           └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Redis** | Job queue, state management, distributed locking |
| **Worker** | PDF processing using pypdfium2 + PaddleOCR |
| **Triage** | Classifies PDFs as digital or scanned |
| **Parser** | Extracts text, tables, images |
| **Converter** | Transforms parsed data to YAML format |

### File Flow

1. PDFs placed in `data/` directory
2. `queue-init` scans and adds files to Redis queue
3. Workers pull files from queue with distributed locks
4. Parser processes PDF structure (GPU-accelerated with PaddleOCR)
5. YAML files written to `data/output/`
6. Completed files tracked in Redis `file:done` set

## Project Structure

For stable entrypoints and module boundaries, see `docs/STRUCTURE.md`.

## Testing

```bash
# Install test dependencies (run outside Docker)
pip install pytest fakeredis

# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_lock_operations.py -v    # Lock atomicity tests
pytest tests/test_deduplicator.py -v       # Deduplication tests
pytest tests/test_yaml_converter.py -v     # Output format tests
```

## Troubleshooting

### "CUDA not available" or GPU errors

**Solution:** Switch to CPU mode:
```bash
# In .env:
PIPELINE_DEVICE=cpu
```

### Docker build fails

**Solution 1:** Build without cache:
```bash
docker build --no-cache -t pdf-pipeline:latest .
```

**Solution 2:** Check disk space:
```bash
df -h
docker system prune -a  # Warning: removes all unused images
```

### Out of memory (OOM)

**Solution 1:** Reduce page limit:
```bash
# In .env:
MAX_PDF_PAGES=50
```

**Solution 2:** Reduce worker memory:
```bash
# In .env:
WORKER_MEMORY=4G
```

**Solution 3:** Use single worker only:
```bash
docker compose up -d  # Don't use --profile scale
```

### Pipeline stuck / no output

**Check Redis:**
```bash
docker compose logs redis
docker compose restart redis
```

**Check worker logs:**
```bash
docker compose logs worker-0
```

**Reset queue:**
```bash
docker compose down
docker compose up -d
docker compose run --rm queue-init
```

### Permission denied on data/

```bash
chmod -R 755 data/
# Or for Docker volume issues:
sudo chown -R $USER:$USER data/
```

### Files not being processed

Ensure files are in the queue:
```bash
docker compose run --rm queue-init --pattern "**/*.pdf"
```

### Worker crashes repeatedly

Check for problematic PDF in failed set - the pipeline will skip it on restart with `SAFE_MODE=true`.

## API Reference (For Programmatic Use)

### Direct Python Usage

```python
from pdf_yaml_pipeline.parsers import UnifiedParser

# Initialize parser
parser = UnifiedParser(
    ocr_enabled=False,
    table_extraction=True
)

# Parse a PDF
result = parser.parse("/path/to/document.pdf")

# Access results
print(result["document"]["page_count"])
print(result["content"]["paragraphs"])
print(result["tables"])
```

### Redis Queue Keys

| Key | Type | Description |
|-----|------|-------------|
| `file:queue` | List | Files waiting to be processed |
| `file:queue:set` | Set | Deduplication set for queue |
| `file:processing` | List | Files currently being processed |
| `file:done` | Set | Successfully processed files |
| `file:failed` | Set | Failed files (dead letter queue) |
| `file:lock:{hash}` | String | Per-file distributed lock |

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR engine for scanned documents
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) - Text extraction
- [Redis](https://redis.io/) - Queue management

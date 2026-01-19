# Examples

Sample files for testing the PDF-YAML pipeline.

## Getting Sample PDFs

Run the setup script to download a sample PDF:

```bash
./setup.sh
```

Or manually download any PDF file into the `data/` directory.

## Expected Output

After processing, YAML files will appear in `data/output/` with this structure:

```yaml
document:
  source_path: "sample.pdf"
  format: "pdf"
  page_count: 5
  truncated: false

content:
  paragraphs:
    - "First paragraph text..."
    - "Second paragraph text..."

tables:
  - cells:
      - {text: "Header", row: 0, col: 0}
      - {text: "Value", row: 1, col: 0}
```

## Test Files

| File | Description |
|------|-------------|
| `data/sample_udhr.pdf` | UN Human Rights Declaration (downloaded by setup.sh) |

Add your own PDFs to `data/` for processing.

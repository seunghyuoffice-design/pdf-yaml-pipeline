#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

fail=0

check_no_matches() {
  local pattern="$1"
  local path="$2"
  local label="$3"
  local allow_pattern="${4:-}"
  if rg -n --glob '*.py' "$pattern" "$path" > /tmp/structure_check.out; then
    if [[ -n "$allow_pattern" ]]; then
      rg -v "$allow_pattern" /tmp/structure_check.out > /tmp/structure_check.filtered || true
      mv /tmp/structure_check.filtered /tmp/structure_check.out
    fi
    if [[ -s /tmp/structure_check.out ]]; then
      echo "Structure violation: $label"
      cat /tmp/structure_check.out
      echo
      fail=1
    fi
  fi
}

# utils should not depend on higher layers
check_no_matches "from pdf_yaml_pipeline\\.(parsers|quality|qa|rag|ocr|deduplication|triage|security|converters|orchestrator)" \
  "src/pdf_yaml_pipeline/utils" \
  "utils imports higher layers"

# converters should not import parsers/quality/qa/orchestrator internals
check_no_matches "from pdf_yaml_pipeline\\.(parsers|quality|qa|orchestrator)" \
  "src/pdf_yaml_pipeline/converters" \
  "converters import higher layers" \
  "parsers\\.base import ParsedDocument"

# quality/qa should not import parsers internals
check_no_matches "from pdf_yaml_pipeline\\.parsers" \
  "src/pdf_yaml_pipeline/quality" \
  "quality imports parsers"
check_no_matches "from pdf_yaml_pipeline\\.parsers" \
  "src/pdf_yaml_pipeline/qa" \
  "qa imports parsers" \
  "parsers\\.special_clause_parser"

rm -f /tmp/structure_check.out

if [[ "$fail" -ne 0 ]]; then
  echo "Structure check failed."
  exit 1
fi

echo "Structure check passed."

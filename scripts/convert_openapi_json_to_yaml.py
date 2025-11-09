"""
Small utility to convert docs/openapi.json -> docs/openapi.yaml
Run: python scripts/convert_openapi_json_to_yaml.py
Requires: PyYAML (pip install pyyaml)
"""
import json
from pathlib import Path
import sys

try:
    import yaml
except Exception as e:
    print("PyYAML is required. Install with: pip install pyyaml")
    raise

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "docs" / "openapi.json"
OUT = ROOT / "docs" / "openapi.yaml"

if not IN.exists():
    print(f"Input file not found: {IN}")
    sys.exit(2)

with IN.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Use safe_dump with options to keep readability
with OUT.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

print(f"Wrote YAML to: {OUT}")

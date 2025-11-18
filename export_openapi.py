from pathlib import Path
from app import app

out = Path(__file__).with_name("docs").joinpath("openapi.json")
out.parent.mkdir(parents=True, exist_ok=True)
openapi = app.openapi()
with open(out, "w", encoding="utf-8") as f:
    import json

    json.dump(openapi, f, indent=2, ensure_ascii=False)
print(f"Wrote OpenAPI spec to: {out}")

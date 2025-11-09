from fastapi.testclient import TestClient
import sys
import os

# ensure repo root is on sys.path so `app` (app.py) can be imported when running tests
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from app import app as fastapi_app


def run_smoke():
    with TestClient(fastapi_app) as client:
        print("Calling /health...")
        r = client.get("/health")
        print("HEALTH", r.status_code, r.json())

        print("Calling /predict (single)...")
        payload = {"data": {"n_tokens_content": 100, "n_tokens_title": 10}}
        r = client.post("/predict", json=payload)
        print("PREDICT", r.status_code, r.json())

        print("Calling /predict_batch (json)...")
        batch = {"instances": [
            {"n_tokens_content": 100, "n_tokens_title": 10},
            {"n_tokens_content": 40, "n_tokens_title": 5}
        ]}
        r = client.post("/predict_batch", json=batch)
        print("BATCH_JSON", r.status_code, r.json())


if __name__ == "__main__":
    run_smoke()

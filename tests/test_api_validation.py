import sys
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from app import app as fastapi_app


def test_single_non_numeric_returns_400():
    client = TestClient(fastapi_app)
    resp = client.post("/predict", json={"data": {"n_tokens_content": "not-a-number"}})
    assert resp.status_code == 400
    body = resp.json()
    assert body.get("error") == "validation_error"
    assert body.get("details")
    # details may be a list of error dicts; ensure the feature name or message appears
    assert any("n_tokens_content" in str(d) or "not numeric" in str(d) for d in body["details"])


def test_batch_non_numeric_returns_400():
    client = TestClient(fastapi_app)
    resp = client.post("/predict_batch", json={"instances": [{"n_tokens_content": "xyz"}]})
    assert resp.status_code == 400
    body = resp.json()
    assert body.get("error") == "validation_error"
    assert body.get("details")

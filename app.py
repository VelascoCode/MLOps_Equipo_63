from typing import Any, Dict, List, Optional, Tuple, Type
from pathlib import Path
import io
import json
import pickle
import joblib
import numpy as np

import pandas as pd
from fastapi import Request
from fastapi import FastAPI, File, HTTPException, UploadFile
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, RootModel, Field, create_model
from pydantic import model_validator


MODEL_PATH = "models/final_model.pkl"



# Dynamically build the Features model from models/feature_names.json when available.
# This ensures the Pydantic schema matches the pipeline's expected input features.
def _guess_field_type(name: str) -> Type:
    # Heuristics: boolean-like column prefixes -> bool, id/url -> str, target -> int, otherwise float
    lower = name.lower()
    if lower.startswith("data_channel_is_") or lower.startswith("weekday_is_") or lower in ("is_weekend",):
        return Optional[bool]
    if lower in ("url", "link", "id") or "url" in lower:
        return Optional[str]
    if lower in ("shares", "target", "label"):
        return Optional[int]
    # fallback to numeric
    return Optional[float]


def _load_feature_names() -> List[str]:
    # Prefer models/feature_names.json; ignore errors and return empty list if not present
    p = Path("models").joinpath("feature_names.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


_feature_names = _load_feature_names()
if not _feature_names:
    # Fallback: keep the older static schema for compatibility
    class Features(BaseModel):
        url: Optional[str] = Field(None, description="Article URL / unique identifier")
        timedelta: Optional[int] = Field(None, description="Time delta (days) between article publication and dataset reference")
        n_tokens_title: Optional[int] = Field(None, description="Number of tokens in the article title")
        n_tokens_content: Optional[int] = Field(None, description="Number of tokens in the article content/body")
        # ...existing fields omitted for brevity - original static schema retained as fallback
        shares: Optional[int] = Field(None, description="Raw number of shares (target). Note: dataset uses popular = shares > 1400")

        model_config = {
            "json_schema_extra": {
                "example": {
                    "n_tokens_content": 100,
                    "n_tokens_title": 10,
                    "data_channel_is_entertainment": True,
                    "LDA_02": 0.44,
                    "shares": 1600,
                }
            }
        }
else:
    # Build model fields dynamically
    fields: Dict[str, Tuple[Type, None]] = {}
    for fname in _feature_names:
        ftype = _guess_field_type(fname)
        fields[fname] = (ftype, None)

    # ensure we include url and shares if present in docs
    # create Pydantic model
    Features = create_model("Features", __base__=BaseModel, **fields)

    # attach a helpful example to the generated model for OpenAPI
    try:
        # small example using the first few features
        example = {}
        for fn in _feature_names[:6]:
            t = _guess_field_type(fn)
            if t is Optional[bool]:
                example[fn] = False
            elif t is Optional[str]:
                example[fn] = "example"
            elif t is Optional[int]:
                example[fn] = 0
            else:
                example[fn] = 0.0
        Features.model_config = {"json_schema_extra": {"example": example}}
    except Exception:
        pass


class SinglePredictionRequest(BaseModel):
    data: Features

    model_config = {
        "json_schema_extra": {
            "example": {"data": {"n_tokens_content": 100, "n_tokens_title": 10}}
        }
    }


class BatchPredictionRequest(BaseModel):
    instances: List[Features]

    model_config = {
        "json_schema_extra": {
            "example": {
                "instances": [
                    {"n_tokens_content": 100, "n_tokens_title": 10},
                    {"n_tokens_content": 40, "n_tokens_title": 5},
                ]
            }
        }
    }


class SinglePredictionResponse(BaseModel):
    prediction: Any = Field(..., description="Model prediction (label or numeric).")
    probability: Optional[float] = Field(None, description="Probability for positive class (binary)")
    probabilities: Optional[List[float]] = Field(None, description="Probabilities for each class (multi-class)")


class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePredictionResponse]
    n: int = Field(..., description="Number of predictions returned")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize state and attempt to load model at startup
    app.state.model = None
    app.state.model_name = None
    try:
        app.state.model = load_model(MODEL_PATH)
        app.state.model_name = getattr(app.state.model, "__class__", type(app.state.model)).__name__
    except FileNotFoundError:
        app.state.model = None
        app.state.model_name = None
    except Exception:
        app.state.model = None
        app.state.model_name = None
    yield
    # nothing special to do on shutdown


app = FastAPI(title="MLOps Equipo63 - Prediction API", lifespan=lifespan)
# Ensure state keys exist so TestClient and other consumers can always access them
app.state.model = None
app.state.model_name = None


@app.exception_handler(RequestValidationError)
async def fastapi_validation_exception_handler(request: Request, exc: RequestValidationError):
    # Return structured JSON for validation errors. Sanitize details so they are JSON-serializable.
    try:
        raw = exc.errors()
    except Exception:
        raw = str(exc)

    def _sanitize(obj):
        # recursively convert non-serializable objects to strings
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        # basic types
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        try:
            # try to stringify as a last resort
            return str(obj)
        except Exception:
            return repr(obj)

    details = _sanitize(raw)
    return JSONResponse(status_code=400, content={"error": "validation_error", "details": details})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Catch-all to ensure structured 500 responses rather than raw tracebacks
    return JSONResponse(status_code=500, content={"error": "server_error", "details": str(exc)})


def load_model(path: str):
    """Try loading model using joblib (preferred) then pickle as a fallback."""
    try:
        # try joblib first (often used for sklearn objects)
        return joblib.load(path)
    except FileNotFoundError:
        raise
    except Exception:
        # fallback to pickle
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load model with joblib or pickle: {e}")


def align_dataframe(df: pd.DataFrame, model) -> pd.DataFrame:
    """Align a DataFrame to the model's expected features.

    If the model exposes `feature_names_in_` ( sklearn >= 1.0 ), use it
    to order and add missing columns. Otherwise return DataFrame as-is.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(getattr(model, "feature_names_in_"))
        # Add missing columns with pd.NA (will convert to np.nan below)
        for c in expected:
            if c not in df.columns:
                df[c] = pd.NA
        # Keep only expected columns and order them
        df = df[expected]

    # Convert pandas NA to numpy.nan so scikit-learn imputers accept them
    df = df.replace({pd.NA: np.nan})

    # Try converting columns to numeric where possible (leave others untouched)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def _to_python(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    try:
        # numpy scalar
        if isinstance(obj, (np.generic,)):
            return obj.item()
        # numpy array
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # list of numpy scalars
        if isinstance(obj, list):
            return [_to_python(x) for x in obj]
    except Exception:
        pass
    return obj




@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=SinglePredictionResponse, summary="Predict single sample")
def predict_single(payload: SinglePredictionRequest):
    """Predict a single sample.

    Body: {"data": {"feature1": value1, "feature2": value2, ...}}
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not available on server")

    # payload.data is a Features instance - extract dict
    instance = payload.data.model_dump()
    if not isinstance(instance, dict) or not instance:
        raise HTTPException(status_code=400, detail="Request `data` must be a non-empty dict of features")

    df = pd.DataFrame([instance])
    df = align_dataframe(df, app.state.model)

    try:
        pred = app.state.model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    raw_pred = pred[0] if hasattr(pred, "__len__") else pred
    result: Dict[str, Any] = {"prediction": _to_python(raw_pred)}

    # Add probability when available
    try:
        if hasattr(app.state.model, "predict_proba"):
            proba = app.state.model.predict_proba(df)
            # if binary, return probability for positive class
            if proba.shape[1] == 2:
                result["probability"] = _to_python(proba[0, 1])
            else:
                result["probabilities"] = _to_python(proba[0].tolist())
    except Exception:
        # silently ignore probability errors
        pass

    return result


@app.post("/predict_batch", response_model=BatchPredictionResponse, summary="Batch predictions (JSON or CSV upload)")
async def predict_batch(request: Request, instances: Optional[BatchPredictionRequest] = None, file: Optional[UploadFile] = File(None)):
    """Batch predictions.

    Accepts either JSON body: {"instances": [{...}, {...}, ...]}
    or a CSV file upload (form-data) where each row is a sample.
    """
    # Accept either an uploaded CSV (form-data) or a JSON body {"instances": [...]}
    if file is not None:
        # read CSV into DataFrame
        try:
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded CSV: {e}")
    else:
        # Try to parse JSON body manually (helps when mixing file and body params)
        try:
            payload = await request.json()
        except Exception:
            payload = None

        instances_list = None
        if payload and isinstance(payload, dict):
            instances_list = payload.get("instances")

        # also accept the pydantic model if FastAPI already parsed it
        if instances is not None and getattr(instances, "instances", None):
            # instances.instances is a list of Features -> extract dict from each
            instances_list = [f.model_dump() for f in instances.instances]

        if not instances_list:
            raise HTTPException(status_code=400, detail="Provide either a JSON body `instances` or a CSV file upload")

        # If we got raw instances from the JSON payload (FastAPI didn't parse them),
        # run Pydantic validation manually so we can return the same structured error.
        validated_instances = []
        if instances is None:
            from pydantic import ValidationError

            details = []
            for idx, rec in enumerate(instances_list):
                try:
                    fm = Features.model_validate(rec)
                    validated_instances.append(fm.model_dump())
                except Exception as e:
                    # Collect a readable error for this record
                    details.append({"loc": ("body", "instances", idx), "msg": str(e)})

            if details:
                return JSONResponse(status_code=400, content={"error": "validation_error", "details": details})
        else:
            validated_instances = instances_list

        df = pd.DataFrame(validated_instances)

        # Check model availability after validation/parsing so clients get validation errors (400)
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not available on server")

        df = align_dataframe(df, app.state.model)

    try:
        preds = app.state.model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    results: List[Dict[str, Any]] = []

    # Try to obtain probabilities
    proba = None
    try:
        if hasattr(app.state.model, "predict_proba"):
            proba = app.state.model.predict_proba(df)
    except Exception:
        proba = None

    for i, p in enumerate(preds):
        item: Dict[str, Any] = {"prediction": _to_python(p)}
        if proba is not None:
            if proba.shape[1] == 2:
                item["probability"] = _to_python(proba[i, 1])
            else:
                item["probabilities"] = _to_python(proba[i].tolist())
        results.append(item)

    return {"predictions": results, "n": len(results)}

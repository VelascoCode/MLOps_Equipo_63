from typing import Any, Dict, List, Optional
import io
import pickle
import joblib
import numpy as np

import pandas as pd
from fastapi import Request
from fastapi import FastAPI, File, HTTPException, UploadFile
from contextlib import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, RootModel, Field
from pydantic import model_validator


MODEL_PATH = "models/final_model.pkl"


class Features(BaseModel):
    """Typed feature set for the Online News Popularity dataset.

    All fields are optional to allow partial inputs; types and descriptions are set
    to improve OpenAPI documentation and runtime validation.
    """
    url: Optional[str] = Field(None, description="Article URL / unique identifier")
    timedelta: Optional[int] = Field(None, description="Time delta (days) between article publication and dataset reference")

    n_tokens_title: Optional[int] = Field(None, description="Number of tokens in the article title")
    n_tokens_content: Optional[int] = Field(None, description="Number of tokens in the article content/body")
    n_unique_tokens: Optional[float] = Field(None, description="Fraction or count of unique tokens in the content")
    n_non_stop_words: Optional[float] = Field(None, description="Number or fraction of non-stop-words in the content")
    n_non_stop_unique_tokens: Optional[float] = Field(None, description="Number or fraction of unique non-stop-words in the content")

    num_hrefs: Optional[int] = Field(None, description="Number of external hyperlinks in the article")
    num_self_hrefs: Optional[int] = Field(None, description="Number of self-referential hyperlinks (same site)")
    num_imgs: Optional[int] = Field(None, description="Number of images in the article")
    num_videos: Optional[int] = Field(None, description="Number of videos in the article")
    average_token_length: Optional[float] = Field(None, description="Average token length in characters")
    num_keywords: Optional[int] = Field(None, description="Number of keywords associated with the article")

    data_channel_is_lifestyle: Optional[bool] = Field(None, description="Is the article from the Lifestyle channel?")
    data_channel_is_entertainment: Optional[bool] = Field(None, description="Is the article from the Entertainment channel?")
    data_channel_is_bus: Optional[bool] = Field(None, description="Is the article from the Business channel?")
    data_channel_is_socmed: Optional[bool] = Field(None, description="Is the article from the Social Media channel?")
    data_channel_is_tech: Optional[bool] = Field(None, description="Is the article from the Tech channel?")
    data_channel_is_world: Optional[bool] = Field(None, description="Is the article from the World channel?")

    kw_min_min: Optional[float] = Field(None, description="Keyword metric: minimum of minimums")
    kw_max_min: Optional[float] = Field(None, description="Keyword metric: maximum of minimums")
    kw_avg_min: Optional[float] = Field(None, description="Keyword metric: average of minimums")
    kw_min_max: Optional[float] = Field(None, description="Keyword metric: minimum of maximums")
    kw_max_max: Optional[float] = Field(None, description="Keyword metric: maximum of maximums")
    kw_avg_max: Optional[float] = Field(None, description="Keyword metric: average of maximums")
    kw_min_avg: Optional[float] = Field(None, description="Keyword metric: minimum of averages")
    kw_max_avg: Optional[float] = Field(None, description="Keyword metric: maximum of averages")
    kw_avg_avg: Optional[float] = Field(None, description="Keyword metric: average of averages")

    self_reference_min_shares: Optional[float] = Field(None, description="Min shares among self-referenced articles")
    self_reference_max_shares: Optional[float] = Field(None, description="Max shares among self-referenced articles")
    self_reference_avg_sharess: Optional[float] = Field(None, description="Avg shares among self-referenced articles (note: original dataset name includes typo 'sharess')")

    weekday_is_monday: Optional[bool] = Field(None, description="Published on Monday?")
    weekday_is_tuesday: Optional[bool] = Field(None, description="Published on Tuesday?")
    weekday_is_wednesday: Optional[bool] = Field(None, description="Published on Wednesday?")
    weekday_is_thursday: Optional[bool] = Field(None, description="Published on Thursday?")
    weekday_is_friday: Optional[bool] = Field(None, description="Published on Friday?")
    weekday_is_saturday: Optional[bool] = Field(None, description="Published on Saturday?")
    weekday_is_sunday: Optional[bool] = Field(None, description="Published on Sunday?")
    is_weekend: Optional[bool] = Field(None, description="Published on weekend?")

    LDA_00: Optional[float] = Field(None, description="LDA topic 0 proportion")
    LDA_01: Optional[float] = Field(None, description="LDA topic 1 proportion")
    LDA_02: Optional[float] = Field(None, description="LDA topic 2 proportion")
    LDA_03: Optional[float] = Field(None, description="LDA topic 3 proportion")
    LDA_04: Optional[float] = Field(None, description="LDA topic 4 proportion")

    global_subjectivity: Optional[float] = Field(None, description="Subjectivity score of the content (0 objective - 1 subjective)")
    global_sentiment_polarity: Optional[float] = Field(None, description="Sentiment polarity of content (negative to positive)")
    global_rate_positive_words: Optional[float] = Field(None, description="Rate of positive words in content")
    global_rate_negative_words: Optional[float] = Field(None, description="Rate of negative words in content")
    rate_positive_words: Optional[float] = Field(None, description="Rate of positive words (normalized)")
    rate_negative_words: Optional[float] = Field(None, description="Rate of negative words (normalized)")

    avg_positive_polarity: Optional[float] = Field(None, description="Average polarity across positive words")
    min_positive_polarity: Optional[float] = Field(None, description="Minimum positive-word polarity")
    max_positive_polarity: Optional[float] = Field(None, description="Maximum positive-word polarity")
    avg_negative_polarity: Optional[float] = Field(None, description="Average polarity across negative words")
    min_negative_polarity: Optional[float] = Field(None, description="Minimum negative-word polarity")
    max_negative_polarity: Optional[float] = Field(None, description="Maximum negative-word polarity")

    title_subjectivity: Optional[float] = Field(None, description="Subjectivity score of the title")
    title_sentiment_polarity: Optional[float] = Field(None, description="Sentiment polarity of the title")
    abs_title_subjectivity: Optional[float] = Field(None, description="Absolute subjectivity of the title")
    abs_title_sentiment_polarity: Optional[float] = Field(None, description="Absolute sentiment polarity of the title")

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

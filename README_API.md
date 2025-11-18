# Prediction API (FastAPI)

This service exposes a small FastAPI app that loads `models/final_model.pkl` and provides endpoints for single and batch predictions.

Files added:

- `app.py` — FastAPI application. Loads the model at startup (path: `models/final_model.pkl`).

How to run (development):

1. From the repository root, activate your Python environment and install dependencies (the repository already includes `fastapi` and `uvicorn` in `requirements.txt`).

2. Start the server with uvicorn:

```powershell
# from repo root
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints

- GET /health
  - Returns simple health info and whether the model was loaded.

- POST /predict
  - Body: JSON { "data": {"feature1": value1, ...} }
  - Returns: { "prediction": ..., "probability": ... }

- POST /predict_batch
  - Accepts either:
    - JSON body: { "instances": [ {..}, {..}, ... ] }
    - form-data file upload (CSV): field name `file`
  - Returns: { "predictions": [ {"prediction":..., "probability":...}, ... ], "n": <count> }

Notes & assumptions

- The API attempts to align incoming data to the model's `feature_names_in_` if the model provides it. Missing columns are added as NaN and extra columns are dropped.
- The model is loaded via pickle from `models/final_model.pkl`. If the model is not present, the API will start but prediction endpoints will return an HTTP 503.
- The API will call `predict_proba` if available and include probabilities in responses.

Example requests (PowerShell):

Single prediction:

```powershell
$body = @{ data = @{ n_tokens_content = 100; n_tokens_title = 10 } } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType 'application/json'
```

Batch prediction (JSON):

```powershell
$instances = @{ instances = @( @{n_tokens_content=100; n_tokens_title=10}, @{n_tokens_content=40; n_tokens_title=5} ) } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict_batch -Method Post -Body $instances -ContentType 'application/json'
```

Batch prediction (CSV upload):

```powershell
# form upload using curl (PowerShell):
cURL -F "file=@samples.csv" http://localhost:8000/predict_batch
```

If you want, I can add a small test to import `app` and ensure `app.state.model` loads successfully on your environment.
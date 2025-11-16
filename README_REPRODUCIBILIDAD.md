# Reproducibilidad

1 - Crea un ambiente virtual con Python=3.11.14 e instala las dependencias:

```bash
pip install -r requirements.txt
```

2 - Descarga los datos con DVC:

```bash
dvc pull
```

3 - Ejecuta el pipeline

```bash
dvc repro
```

4 - Valida los resultados con MLflow:

```bash
mlflow ui
```

Los resultados del modelo deben ser:

- final_auc=0.7273749684423125
- final_accuracy=0.6623539756312021
- max_depth=30
- n_estimators=399
- classifier=RandomForest

5 - Levanta la API (FastAPI) de manera local:

```bash
uvicorn app:app --reload
```

6 - Utiliza los siguientes payloads de ejemplo para testear los endpoints en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs):

/predict

```json
{
  "data": {
    "kw_avg_avg": 584.6888403650005,
    "kw_max_avg": 0,
    "LDA_04": 0.682188294,
    "LDA_02": 0.033351425,
    "kw_avg_max": 0,
    "kw_avg_min": 0,
    "self_reference_min_shares": 918,
    "LDA_03": 0.033333536,
    "LDA_01": 0.033334457,
    "global_subjectivity": 0.6766310953824999,
    "self_reference_avg_sharess": 918,
    "n_non_stop_unique_tokens": 0.663865541,
    "LDA_00": 0.217792289,
    "n_unique_tokens": 0.575129531,
    "average_token_length": 4.393364929,
    "global_rate_positive_words": 0.056872038,
    "global_sentiment_polarity": 0.323333333,
    "avg_positive_polarity": 0.495833333,
    "kw_max_min": 0,
    "kw_min_avg": 0
  }
}
```

/predict_batch

```json
{
  "instances": [
    {
      "kw_avg_avg": 584.6888403650005,
      "kw_max_avg": 0,
      "LDA_04": 0.682188294,
      "LDA_02": 0.033351425,
      "kw_avg_max": 0,
      "kw_avg_min": 0,
      "self_reference_min_shares": 918,
      "LDA_03": 0.033333536,
      "LDA_01": 0.033334457,
      "global_subjectivity": 0.6766310953824999,
      "self_reference_avg_sharess": 918,
      "n_non_stop_unique_tokens": 0.663865541,
      "LDA_00": 0.217792289,
      "n_unique_tokens": 0.575129531,
      "average_token_length": 4.393364929,
      "global_rate_positive_words": 0.056872038,
      "global_sentiment_polarity": 0.323333333,
      "avg_positive_polarity": 0.495833333,
      "kw_max_min": 0,
      "kw_min_avg": 0
    },
    {
      "kw_avg_avg": 584.6888403650005,
      "kw_max_avg": 0,
      "LDA_04": 0.682188294,
      "LDA_02": 0.033351425,
      "kw_avg_max": 0,
      "kw_avg_min": 0,
      "self_reference_min_shares": 918,
      "LDA_03": 0.033333536,
      "LDA_01": 0.033334457,
      "global_subjectivity": 0.6766310953824999,
      "self_reference_avg_sharess": 918,
      "n_non_stop_unique_tokens": 0.663865541,
      "LDA_00": 0.217792289,
      "n_unique_tokens": 0.575129531,
      "average_token_length": 4.393364929,
      "global_rate_positive_words": 0.056872038,
      "global_sentiment_polarity": 0.323333333,
      "avg_positive_polarity": 0.495833333,
      "kw_max_min": 0,
      "kw_min_avg": 0
    }
  ]
}
```

/predict_url

```json
{
  "url": "https://mashable.com/article/landline-phone-brain-rot"
}
```

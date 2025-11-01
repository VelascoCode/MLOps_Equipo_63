# MLOps_Equipo_63

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este proyecto tiene como propósito experimentar de manera práctica cómo se construye, organiza y despliega un sistema de Machine Learning en producción, siguiendo los principios de MLOps.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_equipo_63 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops_equipo_63   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_equipo_63 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# News Popularity Prediction - ML Project

Proyecto de Machine Learning para predecir la popularidad de artículos de noticias online basado en características del contenido.

## 🎯 Objetivo

Clasificar artículos como "populares" o "no populares" basándose en el número de shares usando técnicas de ML y mejores prácticas de MLOps.

## 🏗️ Estructura del Proyecto

news-popularity-ml/
├── data/ # Datos raw y procesados (versionados con DVC)
├── models/ # Modelos entrenados (versionados con DVC)
├── src/ # Código fuente modularizado
├── notebooks/ # Jupyter notebooks para exploración
├── reports/ # Reportes, métricas y visualizaciones
├── dvc.yaml # Pipeline reproducible
└── params.yaml # Parámetros configurables


## 🚀 Configuración Inicial

### 1. Clonar el repositorio

git clone https://github.com/tu-usuario/news-popularity-ml.git
cd news-popularity-ml


### 2. Crear entorno virtual

python -m venv venv
source venv/bin/activate # En Windows: venv\Scripts\activate
pip install -r requirements.txt


### 3. Configurar AWS S3

Copiar archivo de ejemplo
cp .env.example .env

Editar .env con tus credenciales de AWS
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_DEFAULT_REGION=us-east-1


### 4. Configurar DVC con S3

Configurar remote de DVC
dvc remote add -d s3remote s3://tu-bucket/news-popularity-ml

Descargar datos y modelos desde S3
dvc pull


## 🔄 Ejecutar Pipeline Completo

Reproducir todo el pipeline
dvc repro

Ver métricas
dvc metrics show

Comparar con experimentos anteriores
dvc metrics diff


## 📊 Visualizar Experimentos con MLflow

Iniciar MLflow UI
mlflow ui

Abrir en el navegador: http://localhost:5000


## 🔬 Experimentación

### Modificar hiperparámetros

Edita `params.yaml` y ejecuta:

dvc repro


### Trackear cambios

Commit de código
git add src/ params.yaml dvc.yaml dvc.lock
git commit -m "Update hyperparameters"

Push de datos y modelos
dvc push

Push de código
git push origin main


## 📈 Resultados

- **Mejor modelo**: [Se actualiza automáticamente]
- **AUC-ROC**: [Ver en MLflow]
- **Accuracy**: [Ver en MLflow]

## 🛠️ Desarrollo

### Agregar nuevas features

1. Modifica `src/preprocessing.py` o `src/feature_engineering.py`
2. Ejecuta `dvc repro`
3. Compara métricas con `dvc metrics diff`

### Tests

pytest tests/


## 📝 Licencia

MIT License

## 👥 Autores

- Equipo 63

🚀 Comandos de Configuración Inicial

# 1. Inicializar Git
git init
git add .
git commit -m "Initial commit: modular ML project structure"

# 2. Inicializar DVC
dvc init

# 3. Configurar S3 como remote
dvc remote add -d s3remote s3://tu-bucket/news-popularity-ml
dvc remote modify s3remote region us-east-1

# 4. Agregar datos a DVC
dvc add data/raw/online_news_modified.csv
git add data/raw/online_news_modified.csv.dvc data/raw/.gitignore
git commit -m "Add raw data to DVC"

# 5. Ejecutar pipeline por primera vez
dvc repro

# 6. Agregar outputs a DVC
dvc add data/processed/cleaned_data.csv
dvc add models/best_model.pkl
git add data/processed/cleaned_data.csv.dvc models/best_model.pkl.dvc
git commit -m "Add processed data and model to DVC"

# 7. Push a S3
dvc push

# 8. Push a GitHub
git remote add origin https://github.com/tu-usuario/news-popularity-ml.git
git push -u origin main





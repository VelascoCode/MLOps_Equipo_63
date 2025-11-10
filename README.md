# MLOps Equipo 63 — Proyecto y despliegue de un modelo de clasificación de noticias

Este repositorio recoge el trabajo del Equipo 63: limpieza y preparación de datos, experimentación, búsqueda de hiperparámetros, versionado de modelos y despliegue de una API para servir predicciones.

Contenido del README:

- Instalación mínima
- Estructura del repositorio y origen (Cookiecutter)
- Entorno Conda / Pip
- Git / GitHub
- AWS (uso para DVC remoto)
- DVC (versionado de datos)
- MLflow (experimentos y modelos)
- Optuna (búsqueda de hiperparámetros)
- Semilla aleatoria estable
- Testing (pytest y tests incluidos)
- FastAPI (servicio y endpoints)
- Docker y contenerización

## 1) Instalación mínima

Requisitos mínimos para ejecutar y desarrollar:

- Python 3.12.x (recomendado)
- Git
- Docker (opcional, para producción/contenerización)

Instalar dependencias (desde la raíz del repo):

```powershell
# crear/activar entorno (opcional con conda)
conda create -n env_mlops python=3.12 -y; conda activate env_mlops

# instalar dependencias del proyecto
pip install -r requirements.txt
```

## 2) Estructura del repositorio y Cookiecutter

Resumen de archivos y carpetas importantes:

- `app.py` — FastAPI app que sirve el modelo
- `Dockerfile` — imagen multi-stage para producción
- `dvc.yaml`, `params.yaml` — pipelines y parámetros
- `train.py` — script de entrenamiento
- `mlops_equipo_63/` — paquete con código fuente (preprocesado, pipeline, utilidades)
- `models/` — artefactos de modelo (no siempre incluidos en el repo)
- `mlruns/` — runs de MLflow (local)
- `tests/` — suite de tests (pytest)

Este proyecto sigue la estructura generada por Cookiecutter Data Science (plantilla para proyectos de ciencia de datos). La carpeta `mlops_equipo_63/` contiene la lógica de negocio: carga y preparación de datos (`load_and_preparation.py`), extracción de features (`feature_extraction_from_url.py`), pipeline (`pipeline.py`) y utilidades (MLflow/Optuna helpers).

## 3) Entorno Conda / Pip (recomendado)

Comandos recomendados para desarrollo local en Windows PowerShell:

```powershell
conda create -n env_mlops python=3.12 -y
conda activate env_mlops
pip install -r requirements.txt
```

Si prefieres no usar Conda, crear un virtualenv estándar y luego `pip install -r requirements.txt` también funciona.

## 4) Git / GitHub

Buenas prácticas usadas en el proyecto:

- Ramas por feature/issue
- Tests automatizados en `tests/`
- Versionado de artefactos grandes con DVC

Comandos básicos:

```powershell
git checkout -b feature/mi-feature
git add .
git commit -m "Descripción breve"
git push origin feature/mi-feature
```

## 5) AWS (uso principal: DVC remoto)

El repositorio está preparado para usar S3 como remoto de DVC. No incluye credenciales.

Configuración rápida:

```powershell
aws configure
dvc remote add -d storage s3://<tu-bucket>
dvc push
```

## 6) DVC — Versionado de datos

DVC se utiliza para versionar conjuntos de datos y conectar con remotos (S3). El pipeline reproducible está descrito en `dvc.yaml`.

Ejemplos:

```powershell
dvc init                     # si aún no está inicializado
dvc add data/raw/online_news_modified.csv
git add data/raw/online_news_modified.csv.dvc .dvcignore
git commit -m "Añadido raw data dvc"
dvc push
```

Para recuperar datos versionados:

```powershell
dvc pull
```

## 7) MLflow — Experimentos y modelos

MLflow se usa para rastrear experimentos y almacenar modelos (carpeta `mlruns/`).

Arrancar un servidor de tracking local (opcional):

```powershell
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Los artefactos y modelos guardados por MLflow pueden exportarse o copiarse a la carpeta `models/` para servirlos desde la API o empaquetarlos en Docker.

## 8) Optuna — Búsqueda de hiperparámetros

La optimización de hiperparámetros se implementa con Optuna (ver `mlops_equipo_63/Optuna_Study.py`). Los resultados se registran en MLflow para comparar runs.

Ejecutar estudio Optuna (ejemplo):

```powershell
python mlops_equipo_63/Optuna_Study.py
```

## 9) Semilla aleatoria (reproducibilidad)

Para asegurar reproducibilidad el proyecto incluye utilidades para fijar la semilla global (ver `mlops_equipo_63/seed.py`). Usar la semilla consistente en entrenamiento y evaluación ayuda a obtener resultados comparables.

## 10) Testing (pytest)

La suite de pruebas se encuentra en `tests/`. Ejecutar las pruebas con:

```powershell
pytest -q
```

Incluye tests para componentes de carga/preparación, EDA, utilidades, MLflow y la API.

## 11) FastAPI — API y endpoints

La API se implementa en `app.py`. Endpoints principales:

- `GET /health` — Estado del servicio y carga del modelo
- `POST /predict` — Predicción para una sola instancia
- `POST /predict_batch` — Predicciones en lote (JSON o CSV)
- `POST /predict_url` — Extrae características desde una URL y devuelve predicción

Levantar la API en desarrollo:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Comprobación rápida (PowerShell):

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

Notas de implementación:

- `mlops_equipo_63/feature_extraction_from_url.py` contiene la lógica para extraer features desde páginas web.
- La API carga el modelo desde `models/` al iniciar. En desarrollo, monte `models/` desde el host para poder reemplazar el modelo sin rebuild.

## 12) Docker y contenerización

El `Dockerfile` es multi-stage y optimizado para reducir tamaño final.

Opciones de uso:

- Desarrollo (montar `models/` desde host):

```powershell
docker build -t ml-service:latest .
docker run --rm -p 8000:8000 -v ${PWD}\models:/app/models:ro ml-service:latest
```

- Producción (incluir modelo en la imagen):

1. Copiar los artefactos del modelo (`models/final_model.pkl`, `models/feature_names.json`) en `models/`.
2. `docker build -t ml-service:latest .`
3. `docker run --rm -p 8000:8000 ml-service:latest`

Consideraciones:

- Montar `models/` facilita iteración en desarrollo.
- Incluir el modelo en la imagen es aconsejable para despliegues inmutables.

## Cómo reproducir lo esencial (rápido)

1. Crear entorno e instalar dependencias.
2. Recuperar datos si están en DVC: `dvc pull`.
3. Entrenar: `python train.py` o ejecutar el pipeline DVC (`dvc repro`).
4. Revisar resultados en MLflow (`mlruns/`).
5. Levantar API: `uvicorn app:app --reload`.

## Archivos y rutas clave

- `app.py` — API
- `train.py` — Entrenamiento
- `dvc.yaml`, `params.yaml` — Pipeline y parámetros
- `mlops_equipo_63/` — Código fuente principal
- `models/` — Modelos (puede no estar incluido en git)
- `mlruns/` — Experimentos MLflow locales
- `tests/` — Pruebas

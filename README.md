# Predicción de popularidad de una publicación​

<img src="docs\images\logo_tec.png" alt="Logo Tecnológico de Monterrey" width="300"/>

# Maestría en Inteligencia Artificial Aplicada  
## Curso: Operaciones de Aprendizaje Automático
#### **Profesor Titular: Dr. Gerardo Rodríguez Hernández**  
#### **Prof Tutor: Iván Reyes Amezcua**

**Nombres y matrículas:**
| Nombre Completo | Matrícula |
| :-------------- | :-------- |
| Jhamyr Arnulfo Alcalde Oballe | A01795401 |
| Alberto Aquino Mendoza | A01796857 |
| Diego Andres Bernal Diaz | A01795975 |
| Rafael Fernando Olmedo Aguilar | A01796862 |
| Carlos Leopoldo Velasco Bautista | A01796699 |

**Equipo: 63**
-----

### El Problema:​
Mashable, un líder en noticias digitales, enfrenta un desafío clave: una gran desproporción entre el alto volumen de artículos que publica y los pocos que logran volverse virales. Esta impredictibilidad conduce a una asignación de recursos (tiempo y presupuesto) que no siempre es eficiente.​

### La Oportunidad de Negocio:​
Proponemos transformar la predicción de popularidad en acciones de negocio medibles para:​
-   Priorizar Contenido: Identificar y destacar artículos con alta probabilidad de éxito antes de publicarlos.​
-   Optimizar la Inversión: Enfocar los esfuerzos de marketing y promoción únicamente en el contenido de mayor potencial.​
-   Maximizar el Alcance: Determinar los horarios de publicación más efectivos para cada tipo de noticia.​

### Objetivo
Clasificar publicaciones como "populares" o "no populares" basándose en el número de shares usando técnicas de ML y mejores prácticas de MLOps.

### Machine Learning Canvas
<img src="docs\images\Machine Learning Canvas.png" alt="ML Canvas del proyecto" width="1200"/>

-----

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este proyecto tiene como propósito experimentar de manera práctica cómo se construye, organiza y despliega un sistema de Machine Learning en producción, siguiendo los principios de MLOps.

### Conda/Pip Environment

Instalamos [Conda](https://www.anaconda.com/docs/getting-started/miniconda/main) y creamos un ambiente virtual (mna-mlops) para gestionar todas las librerias de Python con Pip.

```bash
conda --version
conda create -n mna-mlops python=3.12.0
conda activate mna-mlops
```

-----

### Cookiecutter Data Science

Una vez dentro del ambiente virtual de Conda, instalamos la librería de [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org) para estructurar el trabajo de acuerdo a los estándares de ciencia de datos:

```bash
pip install cookiecutter-data-science
ccds
```

Utilizamos la siguiente configuración para la estructura del proyecto:

```bash
project_name (project_name): mlops_equipo_63
repo_name (mlops_equipo_63): mlops_equipo_63
module_name (mlops_equipo_63): mlops_equipo_63
author_name (Your name (or your organization/company/team)): Equipo 63
description (A short description of the project.): Este proyecto tiene como proposito experimentar de manera practica como se construye, organiza y despliega un sistema de Machine Learning en producion, siguiendo los principios de MLOps.
python_version_number (3.10): 3.12.0
Select dataset_storage
    1 - none
    2 - azure
    3 - s3
    4 - gcs
    Choose from [1/2/3/4] (1): 1
Select environment_manager
    1 - virtualenv
    2 - conda
    3 - pipenv
    4 - uv
    5 - pixi
    6 - poetry
    7 - none
    Choose from [1/2/3/4/5/6/7] (1): 2
Select dependency_file
    1 - requirements.txt
    2 - pyproject.toml
    3 - environment.yml
    4 - Pipfile
    5 - pixi.toml
    Choose from [1/2/3/4/5] (1): 1
Select pydata_packages
    1 - none
    2 - basic
    Choose from [1/2] (1): 1
Select testing_framework
    1 - none
    2 - pytest
    3 - unittest
    Choose from [1/2/3] (1): 1
Select linting_and_formatting
    1 - ruff
    2 - flake8+black+isort
    Choose from [1/2] (1): 1
Select open_source_license
    1 - No license file
    2 - MIT
    3 - BSD-3-Clause
    Choose from [1/2/3] (1): 2
Select docs
    1 - mkdocs
    2 - none
    Choose from [1/2] (1): 2
Select include_code_scaffold
    1 - Yes
    2 - No
    Choose from [1/2] (1): 2
```

El resultado es un folder (mlops_equipo_63) con la siguiente estructura:

```txt
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
│                         article-popularity and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── mlops_equipo_63   <- Source code for use in this project.
    │
    └── __init__.py             <- Makes article-popularity a Python module
```
-----

### Git/GitHub

Para gestionar el versionamiento del código, utilizamos [Git](https://git-scm.com/install) y lo vinculamos con nuestra cuenta de [GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github) para poder hacer cambios de manera local y empujarlos al repositorio remoto. Para esto un miembro del equipo creó el repositorio remoto [MLOps_Equipo_63](https://github.com/VelascoCode/MLOps_Equipo_63) (publico) y proporcionó permisos de lectura y escritura a los demás miembros del equipo.

Una vez que creada la estructura del proyecto con CoockieCutters, inicializamos Git en la raíz del proyecto (mlops_equipo_63), añadimos todos los archivos, creamos el commit, y empujamos todos los cambios a la rama principal del repositorio remoto:

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

-----
### Pipeline Automatizado con DVC
El proyecto utiliza un **pipeline automatizado con DVC (Data Version Control)** para organizar y versionar el flujo completo de Machine Learning, desde la preparación de datos hasta la evaluación de modelos.

**¿Por qué usar DVC?**
- Permite automatizar todo el proceso de datos y modelos.
- Garantiza que los resultados sean reproducibles: cualquier persona puede ejecutar el pipeline y obtener exactamente los mismos resultados si los datos y los parámetros no cambian.
- Facilita el trabajo colaborativo, la trazabilidad y la gestión de versiones en equipo.
- El archivo `dvc.yaml` define las etapas (stages) clave que se ejecutan automáticamente.

**¿Cómo ejecutar el pipeline?**
Para ejecutar el pipeline completo y actualizar solo las etapas necesarias, usa:

```bash
dvc repro
```

DVC revisa los cambios en datos, scripts y parámetros, y solo ejecuta las etapas que realmente necesitan actualizarse.

**Visualización y trazabilidad**

-   Puedes visualizar el flujo del pipeline con:

```
dvc dag
```

-   Todos los archivos generados y versionados por DVC pueden enviarse al almacenamiento remoto (como S3 o Google Drive) con:

```
dvc push
```

**Beneficios**
-   Reproducibilidad garantizada
-   Versionado y control eficiente de datos/modelos/métricas
-   Colaboración real y segura
-   Resultados fácilmente comparables y auditables


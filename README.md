# MLOps_Equipo_63

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este proyecto tiene como propósito experimentar de manera práctica cómo se construye, organiza y despliega un sistema de Machine Learning en producción, siguiendo los principios de MLOps.

## 🎯 Objetivo

Clasificar artículos como "populares" o "no populares" basándose en el número de shares usando técnicas de ML y mejores prácticas de MLOps.


## 📁 Estructura del Proyecto

La organización de carpetas sigue las mejores prácticas de proyectos de Machine Learning para facilitar la reutilización, el mantenimiento y la automatización.

```
mlops_equipo_63/
├── data/                  # Almacena datos crudos y procesados
├── models/                # Archivos de modelos entrenados
├── reports/               # Reportes y visualizaciones de métricas
├── mlops_equipo_63/                   # Módulos reutilizables de Python (lógica principal)
│   ├── data_loader.py         # Funciones para carga de datos
│   ├── preprocessing.py       # Funciones para limpiar y transformar datos
│   ├── model_training.py      # Funciones para entrenamiento de modelos
│   └── ...                    # Otros módulos de utilidades
│
│   └── scripts/               # Scripts ejecutables para pipeline y DVC
│       ├── prepare_data.py        # Script para preparar y limpiar datos
│       ├── train_model.py         # Script para entrenamiento de modelos
│       └── evaluate_model.py      # Script para evaluación y reportes finales
├── notebooks/             # Cuadernos interactivos para exploración y análisis rápido
├── dvc.yaml               # Pipeline de DVC (define las etapas del flujo automático)
├── params.yaml            # Parámetros configurables utilizados en el pipeline
└── README.md              # Documentación principal del proyecto

```

### Detalle de carpetas y módulos
- `mlops_equipo_63/`: Contiene el código modular (funciones y clases) reutilizable para manipulación de datos, procesamiento y entrenamiento. Estos módulos se pueden importar en otros scripts y notebooks.

- `mlops_equipo_63/scripts/`: Incluye los scripts ejecutables principales que orquestan cada fase concreta del pipeline (preparación de datos, entrenamiento y evaluación). Estos scripts son los que DVC ejecuta en cada stage.

- `data/, models/ y reports/`: Espacios para los archivos de datos, modelos guardados y reportes/resultados generados automáticamente.

- `notebooks/`: Uso recomendado para explorar datos, verificar hipótesis o realizar análisis adicionales de manera interactiva.

- `dvc.yaml y params.yaml`: Definen las etapas, dependencias, outputs y parámetros del pipeline del proyecto, asegurando trazabilidad y reproducibilidad.

- `README.md`: Esta documentación central que describe la finalidad y estructura del proyecto.

Esta división permite trabajar ordenadamente, con flujos reproducibles y facilita el trabajo en equipo tanto en experimentación como en producción.

--------
### ⚙️ Pipeline Automatizado con DVC
El proyecto utiliza un **pipeline automatizado con DVC (Data Version Control)** para organizar y versionar el flujo completo de Machine Learning, desde la preparación de datos hasta la evaluación de modelos.

**¿Por qué usar DVC?**
- Permite automatizar todo el proceso de datos y modelos.
- Garantiza que los resultados sean reproducibles: cualquier persona puede ejecutar el pipeline y obtener exactamente los mismos resultados si los datos y los parámetros no cambian.
- Facilita el trabajo colaborativo, la trazabilidad y la gestión de versiones en equipo.

**Etapas principales del pipeline**
El archivo `dvc.yaml` define las etapas (stages) clave que se ejecutan automáticamente:

1. **Preparación de datos (`prepare_data`):**

    -   Limpia, transforma y prepara los datos crudos.
    -   Entradas: datos originales y scripts de preprocesamiento.
    -   Salida: archivo de datos limpios y métricas de calidad.

2. **Entrenamiento de modelo (train_model):**

    -   Ejecuta la optimización y el entrenamiento del mejor modelo.
    -   Entradas: datos procesados y scripts de entrenamiento.
    -   Salida: modelo entrenado y métricas de entrenamiento.

3.  **Evaluación de modelo (evaluate_model):**

    -   Realiza la evaluación final del modelo sobre el conjunto de test.
    -   Entradas: modelo entrenado y datos procesados.
    -   Salida: reportes, métricas y visualizaciones.

**¿Cómo ejecutar el pipeline?**
Para ejecutar el pipeline completo y actualizar solo las etapas necesarias, usa:

`bash
> dvc repro`

DVC revisa los cambios en datos, scripts y parámetros, y solo ejecuta las etapas que realmente necesitan actualizarse.

**Visualización y trazabilidad**

-   Puedes visualizar el flujo del pipeline con:

`> dvc dag`

-   Todos los archivos generados y versionados por DVC pueden enviarse al almacenamiento remoto (como S3 o Google Drive) con:

`dvc push`

**Beneficios**
-   Reproducibilidad garantizada
-   Versionado y control eficiente de datos/modelos/métricas
-   Colaboración real y segura
-   Resultados fácilmente comparables y auditables


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





# MLOps Equipo 63

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

```markdown
# MLOps Equipo 63

### Entorno Conda/Pip

Este proyecto usa Conda para gestionar el entorno. A continuación se muestran comandos de ejemplo para crear y activar un entorno:

```bash
conda --version
conda create -n mna-mlops python=3.12.0
conda activate mna-mlops
```

-----

### Cookiecutter Data Science

Dentro del entorno de Conda instalamos la plantilla Cookiecutter Data Science para crear la estructura del proyecto según buenas prácticas:

```bash
pip install cookiecutter-data-science
ccds
```

Durante la creación se emplearon estas opciones para nuestro repositorio:

```bash
project_name (project_name): mlops_equipo_63
repo_name (mlops_equipo_63): mlops_equipo_63
module_name (mlops_equipo_63): mlops_equipo_63
author_name (Your name (or your organization/company/team)): Equipo 63
description (A short description of the project.): Este proyecto tiene como propósito practicar la construcción, organización y despliegue de un sistema de Machine Learning en producción, siguiendo principios de MLOps.
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

El resultado es una carpeta (`mlops_equipo_63`) con la siguiente estructura:

```txt
├── LICENSE            <- Licencia open-source (si se escogió una)
├── Makefile           <- Comandos de conveniencia como `make data` o `make train`
├── README.md          <- README principal para desarrolladores
├── data
│   ├── external       <- Datos de terceros
│   ├── interim        <- Datos intermedios transformados
│   ├── processed      <- Conjuntos de datos finales para modelado
   └── raw            <- Datos originales e inmutables
│
├── docs               <- Documentación (mkdocs)
├── models             <- Modelos entrenados y artefactos relacionados
├── notebooks          <- Notebooks Jupyter (nombres numerados para orden)
├── pyproject.toml     <- Configuración del proyecto y herramientas
├── references         <- Diccionarios de datos y documentación adicional
├── reports            <- Informes generados (HTML, PDF, etc.)
│   └── figures        <- Gráficos e imágenes
├── requirements.txt   <- Dependencias del proyecto
└── mlops_equipo_63    <- Código fuente del proyecto
    └── __init__.py    <- Hace el paquete importable
```

-----

### Git/GitHub

Para el control de versiones usamos Git y alojamos el repositorio en GitHub: https://github.com/VelascoCode/MLOps_Equipo_63. Después de generar la estructura con Cookiecutter inicializamos Git y empujamos el código:

```bash
git --version

cd mlops_equipo_63

git init
git add .
git commit -m "CCDS defaults"
git remote add origin https://github.com/VelascoCode/MLOps_Equipo_63
git branch -M main
git push -u origin main
```

-----

### Amazon Web Services (AWS): IAM, S3

Para versionar datos utilizamos un bucket de S3 (p. ej. `s3://mlops-equipo-63`). Se creó un rol IAM con permisos de lectura/escritura y usuarios con access keys para acceso por línea de comando.

Instalación y configuración de AWS CLI:

```bash
pip install awscli

aws --version

aws configure

AWS Access Key ID [****************5XMO]: 
AWS Secret Access Key [****************JlOz]:
Default region name [us-east-1]: us-east-1
Default output format [json]: json
```

Comandos útiles para verificar conexión y listar buckets:

```bash
aws sts get-caller-identity
aws s3 ls
```

-----

### Data Version Control (DVC)

Usamos DVC para versionar los datos y conectar con el bucket S3. Pasos básicos:

```bash
pip install dvc

dvc --version

cd mlops_equipo_63

dvc init
git commit -m "Initialize DVC"
```

Configurar remote en S3:

```bash
pip install dvc-s3

dvc remote add -d storage s3://mlops-equipo-63
```

Ejemplo de uso (añadir y empujar un archivo):

```bash
mkdir -p data/raw && echo -e "id,name,age,city\n1,Alice,25,New York" > data/raw/dummy.csv # (v1)

dvc add data/raw/test.csv # se genera data/raw/dummy.csv.dvc
git commit -m 'added dummy.csv.dvc file'
dvc push
```

Para deshacer cambios o volver a una versión anterior:

```bash
git log --oneline
git checkout <commit-hash>

dvc pull
```

-----

### MLflow

Usamos MLflow para el versionamiento de modelos. Ejemplo para ejecutar un servidor local de tracking:

```bash
pip install mlflow
mlflow server --host 127.0.0.1 --port 8080
```

``` 


Para el versionamiento de los modelos de Machine Learning utilizamos MLflow. Instalamos MLflow en el ambiente virtual de Conda y lo ejecutamos dentro del folder */notebooks* para almacenar los experimentos en esta ruta:

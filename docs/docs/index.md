# MLOps_Equipo_63 documentation!

## Description

Este proyecto tiene como propósito experimentar de manera práctica cómo se construye, organiza y despliega un sistema de Machine Learning en producción, siguiendo los principios de MLOps.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://s3://mlops-equipo-63/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://s3://mlops-equipo-63/data/` to `data/`.



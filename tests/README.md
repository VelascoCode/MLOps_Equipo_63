# Tests

This folder contains the project's unit tests (pytest) and small helpers used to make tests hermetic when optional heavy dependencies are not installed.

## Organization

- `tests/` — test files follow the pattern `test_*.py` and mostly map one-to-one to modules in the `mlops_equipo_63` package. Example mappings:
  - `test_load_and_preparation.py` → `mlops_equipo_63/load_and_preparation.py`
  - `test_split_and_dummy.py` → `mlops_equipo_63/Split_and_Dummy.py`
  - `test_retrain_and_evaluate.py` → `mlops_equipo_63/Retrain_and_Evaluate.py`
  - `test_optuna_study.py` → `mlops_equipo_63/Optuna_Study.py`
  - `test_pipeline.py` → `mlops_equipo_63/pipeline.py`

There is also `conftest.py` which sets up test-time shims/stubs and adjusts `sys.path` so the package can be imported during CI or local runs without installing the package.

## Notes about dependencies and stubs

- The project uses `mlflow` and `optuna` in production code. To keep tests fast and avoid requiring those packages in CI, `tests/conftest.py` provides minimal runtime stubs for `mlflow` and `optuna.integration.mlflow` when they are missing. This means the test suite can run without installing heavy optional deps.
- Plotting tests set the non-interactive Matplotlib backend (Agg) so tests won't try to open GUI windows.

## Run tests locally

From the repository root on Windows PowerShell:

```powershell
# run all tests (quiet)
pytest -q

# run a single test file
pytest tests/test_load_and_preparation.py -q

# run a single test function
pytest tests/test_load_and_preparation.py::test_prepare_numeric_df_and_imputation -q
```

If you prefer more verbose output or to see print() statements, omit `-q`.

## CI (GitHub Actions) example

You can use a minimal workflow that installs dev/test requirements and runs pytest. Because we include test-time stubs, installing `mlflow`/`optuna` is not required for the tests to pass.

Example job snippet (GitHub Actions):

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest
      - name: Run tests
        run: pytest -q
```

If you prefer to install the optional heavy dependencies in CI (to exercise MLflow or Optuna integrations), add them to the install step.

## Extending tests

- Add more unit tests for edge cases and failure modes.
- Add integration tests that run against a small local MLflow store or real Optuna runs if you want to validate full behavior (these should be placed under `tests/integration/` and run conditionally in CI).

## Troubleshooting

- If imports fail inside tests, make sure you're running pytest from the repository root. The `tests/conftest.py` adds the repo root to `sys.path` for convenience.
- If CI needs the real `mlflow`/`optuna`, install them in the workflow or adjust `conftest.py` to skip stubbing when you want full integration testing.

If you'd like, I can add a small `requirements-dev.txt` and a GitHub Actions workflow file to the repo next.

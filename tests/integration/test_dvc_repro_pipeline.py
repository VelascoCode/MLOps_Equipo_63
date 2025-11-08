import os
import shutil
import subprocess
from pathlib import Path
import pytest


@pytest.mark.integration
def test_dvc_repro_pipeline_runs_when_enabled():
    """
    Integration test that reproduces the DVC pipeline defined in `dvc.yaml`.

    Safety checks / behavior:
    - By default this test is SKIPPED unless the environment variable
      `RUN_DVC_INTEGRATION` is set to '1'. This prevents heavy / networked
      operations during normal unit-test runs or CI where DVC isn't desired.
    - The test also skips if the `dvc` CLI is not on PATH.
    - If both conditions are met it runs `dvc repro` in the repository root
      and asserts a zero return code.
    """

    # Enable only when explicitly requested
    if os.environ.get("RUN_DVC_INTEGRATION", "0") != "1":
        pytest.skip("DVC integration not enabled. Set RUN_DVC_INTEGRATION=1 to run this test.")

    # Ensure dvc is available
    if shutil.which("dvc") is None:
        pytest.skip("`dvc` CLI not found on PATH. Install DVC to run this integration test.")

    repo_root = Path(__file__).resolve().parents[2]
    # Basic sanity: dvc.yaml and params.yaml should exist
    assert (repo_root / "dvc.yaml").exists(), "dvc.yaml not found in repo root"
    assert (repo_root / "params.yaml").exists(), "params.yaml not found in repo root"

    # Run dvc repro. This can be long depending on the pipeline; set a generous timeout.
    proc = subprocess.run(["dvc", "repro"], cwd=str(repo_root), capture_output=True, text=True, timeout=60 * 60)

    msg = (
        f"dvc repro failed (exit {proc.returncode})\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert proc.returncode == 0, msg

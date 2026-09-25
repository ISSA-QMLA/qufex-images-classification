"""Exercise the actual Bash launcher with a disposable worker (Linux only)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Slurm launcher requires Linux")
LAUNCHER = Path(__file__).parents[1] / "scripts/slurm/compression-sweep.sbatch"


@pytest.fixture
def launch_env(tmp_path):
    repo = tmp_path / "repo"
    for directory in ("qmla", "scripts", "configs", "notebooks"):
        (repo / directory).mkdir(parents=True)
    for filename in ("configs/compression_sweep.toml", "notebooks/05-compression-sweep.ipynb",
                     "pyproject.toml", "uv.lock", "README.md"):
        (repo / filename).write_text("test")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("scontrol", "nvidia-smi", "git"):
        path = binaries / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    worker = tmp_path / "worker"
    worker.write_text(f"#!{sys.executable}\n" + '''import json, os, signal, sys, time
from pathlib import Path
if "-c" in sys.argv:
    sys.exit(0)
if "--dry-run" in sys.argv:
    print("{}")
    sys.exit(0)
Path(os.environ["QMLA_TEST_ARGS"]).write_text(json.dumps(sys.argv[1:]))
if os.environ.get("QMLA_TEST_WAIT"):
    stopped = False
    def stop(*args):
        global stopped
        stopped = True
    signal.signal(signal.SIGUSR1, stop)
    Path(os.environ["QMLA_TEST_READY"]).touch()
    while not stopped:
        time.sleep(0.01)
    sys.exit(75)
''')
    worker.chmod(0o755)
    return {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "SLURM_SUBMIT_DIR": str(repo), "SLURM_JOB_ID": "123",
            "QMLA_REPO": str(repo), "QMLA_CONFIG": str(repo / "configs/compression_sweep.toml"),
            "QMLA_STAGE": "all", "QMLA_PYTHON": str(worker),
            "QMLA_BUNDLE": str(tmp_path / "bundle"), "QMLA_DATA_ROOT": str(tmp_path / "data"),
            "QMLA_TEST_ARGS": str(tmp_path / "args.json"), "QMLA_TEST_READY": str(tmp_path / "ready")}


def test_launcher_resumes_with_saved_configuration(launch_env):
    subprocess.run(["bash", str(LAUNCHER)], env=launch_env, check=True, capture_output=True, timeout=20)
    args = json.loads(Path(launch_env["QMLA_TEST_ARGS"]).read_text())
    assert "--resume" not in args and "--checkpoints-dir" in args
    root = Path(launch_env["QMLA_BUNDLE"]) / "study"
    root.mkdir()
    (root / "study.json").write_text("{}")
    (root / "resolved_sweep.toml").write_text("test")
    subprocess.run(["bash", str(LAUNCHER)], env=launch_env, check=True, capture_output=True, timeout=20)
    args = json.loads(Path(launch_env["QMLA_TEST_ARGS"]).read_text())
    assert "--resume" in args and "--checkpoints-dir" not in args
    assert args[args.index("--config") + 1] == str(root / "resolved_sweep.toml")


def test_launcher_forwards_signal_and_waits_for_worker(launch_env):
    launch_env["QMLA_TEST_WAIT"] = "1"
    process = subprocess.Popen(["bash", str(LAUNCHER)], env=launch_env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 15
        while not Path(launch_env["QMLA_TEST_READY"]).exists():
            assert process.poll() is None, process.communicate()[0]
            assert time.monotonic() < deadline, "Worker failed to start"
            time.sleep(0.02)
        # Concurrent submissions must not start another writer.
        duplicate = subprocess.run(["bash", str(LAUNCHER)], env=launch_env,
                                   capture_output=True, text=True, timeout=10)
        assert duplicate.returncode == 2 and "Another process" in duplicate.stderr
        process.send_signal(signal.SIGUSR1)
        output = process.communicate(timeout=15)[0]
        assert process.returncode == 75, output
        assert "Resubmit" in output
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)

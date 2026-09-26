"""Execute the readable Slurm templates with a disposable worker (Linux only)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Slurm launcher requires Linux")
TEMPLATES = Path(__file__).parents[1] / "scripts/slurm"


@pytest.fixture
def launch_env(tmp_path):
    repo = tmp_path / "repo"
    binaries = repo / ".venv/bin"
    binaries.mkdir(parents=True)
    nvidia = binaries / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\nexit 0\n")
    nvidia.chmod(0o755)
    worker = binaries / "python"
    worker.write_text(f"#!{sys.executable}\n" + '''import json, os, signal, sys, time
from pathlib import Path
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
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
           "QMLA_TEST_ARGS": str(tmp_path / "args.json"),
           "QMLA_TEST_READY": str(tmp_path / "ready")}
    launchers = {}
    for name in ("sweep", "prepare"):
        source = (TEMPLATES / f"compression-{name}.sbatch.example").read_text()
        source = source.replace("/users/famato/QMLA/code/qmla", str(repo))
        target = tmp_path / f"{name}.sbatch"
        target.write_text(source)
        launchers[name] = target
    return env, launchers


@pytest.mark.parametrize("name,stage", [("sweep", "all"), ("prepare", "prepare")])
def test_template_invokes_correct_stage(launch_env, name, stage):
    env, launchers = launch_env
    subprocess.run(["bash", str(launchers[name])], env=env, check=True,
                   capture_output=True, timeout=20)
    args = json.loads(Path(env["QMLA_TEST_ARGS"]).read_text())
    assert args[args.index("--stage") + 1] == stage
    assert args[args.index("--config") + 1] == "configs/compression_sweep.toml"
    assert "--resume" not in args
    if name == "sweep":
        assert args[args.index("--run-dir") + 1].endswith("/compression-hpc/study")
        assert args[args.index("--checkpoints-dir") + 1].endswith("/compression-hpc/checkpoints")


def test_exec_delivers_warning_directly_to_python(launch_env):
    env, launchers = launch_env
    env["QMLA_TEST_WAIT"] = "1"
    process = subprocess.Popen(["bash", str(launchers["sweep"])], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 15
        while not Path(env["QMLA_TEST_READY"]).exists():
            assert process.poll() is None, process.communicate()[0]
            assert time.monotonic() < deadline, "Worker failed to start"
            time.sleep(0.02)
        process.send_signal(signal.SIGUSR1)
        output = process.communicate(timeout=15)[0]
        assert process.returncode == 75, output
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=15)

#!/usr/bin/env bash
#SBATCH --job-name=qmla-check-models
#SBATCH --partition=master
#SBATCH --nodelist=treachery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /users/famato/QMLA/code/qmla
export PATH="$HOME/.local/bin:$PATH"
hostname
nvidia-smi
# Synthetic model check only. Use the real-image pilot to measure resources.
uv run --no-sync python -u - <<'PY'
import torch
from qmla.config import load_config, MODELS
from qmla.model import GalaxyClassifier

assert torch.cuda.is_available(), "CUDA is unavailable in this allocation"
cfg = load_config("configs/hpc.toml", profile="full128", device="cuda")
torch.set_num_threads(cfg.training.cpu_threads)
torch.manual_seed(42)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("Image size:", cfg.data.image_size, "Training batch:", cfg.training.batch_size)

for name in MODELS:
    print(f"Testing {name}...", flush=True)
    model = GalaxyClassifier(cfg, mode=name).cuda().train()
    x = torch.randn(cfg.training.batch_size, 3, cfg.data.image_size,
                    cfg.data.image_size, device="cuda")
    y = torch.arange(len(x), device="cuda") % len(cfg.evaluation.class_names)
    with torch.autocast("cuda", enabled=cfg.training.amp):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
    assert logits.shape == (len(x), len(cfg.evaluation.class_names))
    assert torch.isfinite(loss).item(), f"{name}: non-finite loss"
    loss.backward()
    torch.cuda.synchronize()
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name}/{parameter_name}: missing gradient"
            assert torch.isfinite(parameter.grad).all().item(), f"{name}/{parameter_name}: non-finite gradient"
    print(f"{name}: PASS", flush=True)
    del model, parameter, x, y, logits, loss
    torch.cuda.empty_cache()

print("All three models passed the 128x128 CUDA forward/backward check.")
PY

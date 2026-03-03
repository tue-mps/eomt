"""
inferency.py — Post-training efficiency metric capture for EoMT.

Usage:
    python inferency.py \
        --ckpt_path /path/to/checkpoint.ckpt \
        --backbone swin_tiny \
        --img_size 512 512 \
        --num_classes 150 \
        --num_q 100 \
        --batch_size 8 \
        --device cuda \
        [--all_metrics] \
        [--amp] \
        [--output metrics_output.json]

Backbones are resolved by name from models/backbones/__init__.py.
All Lightning wrappers are bypassed — only the raw nn.Module is loaded.
"""

import argparse
import json
import importlib
import inspect

from pathlib import Path

import torch
import torch.nn as nn

# ── project imports ────────────────────────────────────────────────────────
from models.eomt import EoMT
from metrics import calculate_metrics


 

# ── checkpoint loading (Lightning-free) ────────────────────────────────────

def _strip_lightning_prefix(state_dict: dict) -> dict:
    """Remove 'network.' prefix that LightningModule wraps weights under."""
    return {
        (k[len("network."):] if k.startswith("network.") else k): v
        for k, v in state_dict.items()
    }


def _remove_compiled_prefix(state_dict: dict) -> dict:
    """torch.compile wraps params under '_orig_mod.*'; unwrap that."""
    return {
        k.replace("._orig_mod", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }


def load_eomt_from_checkpoint(
    ckpt_path: str,
    backbone: nn.Module,
    num_classes: int,
    num_q: int,
    num_blocks: int = 4,
    self_a: bool = False,
    masked_attn_enabled: bool = True,
    device: torch.device = torch.device("cpu"),
) -> EoMT:
    """Instantiate EoMT and load weights from a checkpoint without Lightning."""

    model = EoMT(
        encoder=backbone,
        num_classes=num_classes,
        num_q=num_q,
        num_blocks=num_blocks,
        self_a=self_a,
        masked_attn_enabled=masked_attn_enabled,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Unwrap Lightning checkpoint envelope
    state_dict = ckpt.get("state_dict", ckpt)

    # Clean keys produced by Lightning + torch.compile
    state_dict = _strip_lightning_prefix(state_dict)
    state_dict = _remove_compiled_prefix(state_dict)

    # Drop loss-criterion buffers that are not part of the network
    state_dict = {
        k: v for k, v in state_dict.items()
        if "criterion.empty_weight" not in k
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[WARNING] Missing keys ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARNING] Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return model.to(device)


# ── dummy args shim ─────────────────────────────────────────────────────────

class _MetricArgs:
    """
    Minimal args namespace consumed by calculate_metrics / throughput / etc.
    Mirrors only the fields actually read inside metrics.py.
    """
    def __init__(
        self,
        batch_size: int,
        cuda: bool,
        amp: bool,
        eval_amp: bool,
        compile_model: bool = False,
        tqdm: bool = True,
        max_grad_norm: float = 0.0,
    ):
        self.batch_size = batch_size
        self.cuda = cuda
        self.amp = amp
        self.eval_amp = eval_amp
        self.compile_model = compile_model
        self.tqdm = tqdm
        self.max_grad_norm = max_grad_norm


def build_backbone(
    model_name: str,
    img_size: tuple[int, int],
    **kwargs,
) -> nn.Module:
    # Discover top-level model modules in the `models/` directory dynamically
    models_dir = Path(__file__).resolve().parent / "models"
    _model_modules = []
    if models_dir.exists() and models_dir.is_dir():
        # Add any .py modules (except __init__.py)
        for p in sorted(models_dir.iterdir()):
            if p.is_file() and p.suffix == ".py" and p.name != "__init__.py":
                _model_modules.append(f"models.{p.stem}")
        # Add any subpackages (directories with __init__.py)
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and (p / "__init__.py").exists():
                _model_modules.append(f"models.{p.name}")
    else:
        raise RuntimeError(f"models directory not found: {models_dir}")

    for module_path in _model_modules:
        mod = importlib.import_module(module_path)
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name.lower() == model_name.lower() and issubclass(obj, nn.Module):
                print(f"[inferency] Found '{name}' in {module_path}")
                return obj(img_size=img_size, **kwargs)

    all_classes = []
    for module_path in _model_modules:
        mod = importlib.import_module(module_path)
        all_classes += [
            name for name, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, nn.Module) and obj.__module__ == module_path
        ]

    raise ValueError(
        f"Backbone class '{model_name}' not found.\n"
        f"Available: {sorted(all_classes)}"
    )


def run_inferency(
    ckpt_path: str,
    backbone_name: str,
    img_size: tuple[int, int],
    num_classes: int,
    num_q: int,
    batch_size: int,
    device_str: str = "cuda",
    all_metrics: bool = True,
    amp: bool = False,
    num_blocks: int = 4,
    self_a: bool = False,
    masked_attn_enabled: bool = True,
    output_path: str | None = None,
    backbone_kwargs: dict | None = None,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"[inferency] Running on device: {device}")

    # 1. build backbone ──────────────────────────────────────────────────────
    print(f"[inferency] Building backbone: {backbone_name}")
    backbone = build_backbone(backbone_name, img_size, **(backbone_kwargs or {}))

    # 2. load EoMT (no Lightning) ────────────────────────────────────────────
    print(f"[inferency] Loading checkpoint: {ckpt_path}")
    model = load_eomt_from_checkpoint(
        ckpt_path=ckpt_path,
        backbone=backbone,
        num_classes=num_classes,
        num_q=num_q,
        num_blocks=num_blocks,
        self_a=self_a,
        masked_attn_enabled=masked_attn_enabled,
        device=device,
    )
    model.eval()

    # 3. synthetic input (B, 3, H, W) ────────────────────────────────────────
    #    EoMT.forward applies its own (pixel_mean / pixel_std) normalisation,
    #    so randn in [0, 1] range is fine here.
    dummy_input = torch.randn(
        batch_size, 3, img_size[0], img_size[1], device=device,
    )

    # 4. assemble metric args ────────────────────────────────────────────────
    metric_args = _MetricArgs(
        batch_size=batch_size,
        cuda=use_cuda,
        amp=amp,
        eval_amp=amp,
    )

    # 5. run calculate_metrics ───────────────────────────────────────────────
    print("[inferency] Calculating metrics…")
    metrics = calculate_metrics(
        args=metric_args,
        model=model,
        rank=0,
        input=dummy_input,
        device=device if use_cuda else None,
        did_training=False,
        world_size=1,
        all_metrics=all_metrics,
        n_ims=batch_size,
        key_start="inferency/",
    )

    # 6. pretty-print results ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EoMT Efficiency Metrics  |  backbone: {backbone_name}")
    print("=" * 60)

    serialisable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_k, sub_v in value.items():
                print(f"      {sub_k}: {sub_v}")
            serialisable[key] = value
        elif isinstance(value, torch.Tensor):
            v = value.item()
            print(f"  {key}: {v}")
            serialisable[key] = v
        else:
            print(f"  {key}: {value}")
            serialisable[key] = value

    print("=" * 60 + "\n")

    # 7. optional JSON dump ──────────────────────────────────────────────────
    if output_path:
        Path(output_path).write_text(json.dumps(serialisable, indent=2))
        print(f"[inferency] Metrics saved → {output_path}")

    return metrics


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_backbone_kwargs(raw: list[str] | None) -> dict:
    if not raw:
        return {}
    kwargs = {}
    for item in raw:
        k, _, v = item.partition("=")
        if v.lower() in ("true", "false"):
            kwargs[k] = v.lower() == "true"
        else:
            try:
                kwargs[k] = int(v)
            except ValueError:
                try:
                    kwargs[k] = float(v)
                except ValueError:
                    kwargs[k] = v
    return kwargs


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Post-training EoMT efficiency metric capture (Lightning-free).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt_path",    required=True)
    p.add_argument("--backbone",     required=True,
                   help="Backbone class name (case-insensitive).")
    p.add_argument("--num_classes",  required=True, type=int)
    p.add_argument("--num_q",        required=True, type=int,
                   help="Number of EoMT query slots.")
    p.add_argument("--img_size",     nargs=2, type=int, default=[512, 512],
                   metavar=("H", "W"))
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--device",       default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_blocks",   type=int, default=4)
    p.add_argument("--self_a",       action="store_true")
    p.add_argument("--no_masked_attn", action="store_true")
    p.add_argument("--all_metrics",  action="store_true", default=True)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--output",       default=None,
                   help="Path to write JSON metrics output.")
    p.add_argument("--backbone_kwargs", nargs="*", metavar="KEY=VALUE",
                   help="Extra kwargs forwarded to the backbone constructor.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_inferency(
        ckpt_path=args.ckpt_path,
        backbone_name=args.backbone,
        img_size=tuple(args.img_size),
        num_classes=args.num_classes,
        num_q=args.num_q,
        batch_size=args.batch_size,
        device_str=args.device,
        all_metrics=args.all_metrics,
        amp=args.amp,
        num_blocks=args.num_blocks,
        self_a=args.self_a,
        masked_attn_enabled=not args.no_masked_attn,
        output_path=args.output,
        backbone_kwargs=_parse_backbone_kwargs(args.backbone_kwargs),
    )

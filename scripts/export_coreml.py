"""Export a trained RAVE model to an ExecuTorch program for ``neural.live~``.

This is RAVE's single, new-protocol exporter. It reuses ``neural_tilde.LiveModule``
(the audio-rate exporter that backs ``neural.live~``) so the ``.pte`` + sidecar
JSON come from the protocol's single source of truth, ``LiveModule._nn_metadata``
(see ``neural_tilde``'s ``EXECUTORCH_PROTOCOL.md``). It produces a **pair of files
sharing a basename**:

    <name>.pte    ExecuTorch program (encode / decode / forward methods)
    <name>.json   sidecar metadata (``kind: "live"``, method channel/ratio/label table)

The RAVE-specific model building (encoder family wrappers, branch neutralization,
latent sizing, cached_conv warmup) lives in ``scripts/export.py`` and is imported
here; only the ExecuTorch lowering + sidecar emission are delegated to
``neural_tilde``.

The exported model is **streaming** (``cached_conv``): each layer keeps the
previous block's tail so ``neural.live~`` can process contiguous audio blocks
without buffer-boundary clicks. With ``--backend coreml`` the Core ML partitioner
takes the cached_conv mutable buffers over as native Core ML *state*, so that
state persists across ``execute()`` calls (needs macOS 15+/iOS 18 + the ANE at
run time). The export is deterministic (no ``randn``): ``encode`` returns the
latent mean and ``decode`` pads discarded latent dims with zeros, giving pure,
runtime-portable methods.

Usage (from the RAVE repo root):
    uv run python scripts/export_coreml.py --run <run_dir_or_ckpt> \
        [--backend coreml|xnnpack|mlx|portable] [--name NAME] [--output DIR] \
        [--buffer-size N] [--fidelity F] [--ema-weights]
"""

import argparse
import logging
import math
import os
import sys
from typing import List

logging.basicConfig(level=logging.INFO)

import torch

torch.set_grad_enabled(False)

import cached_conv as cc
import gin

try:
    import rave
except ImportError:
    sys.path.append(os.path.abspath("."))
    import rave
import rave.core

# RAVE model-building helpers (the reusable domain logic lives in export.py).
from scripts.export import (compute_latent_size, get_export_class,
                            load_pretrained, neutralize_for_export,
                            zero_streaming_caches)

from neural_tilde import LiveModule


# --------------------------------------------------------------------------- #
# neural_tilde.LiveModule wrapper: exposes RAVE encode/decode/forward methods  #
# --------------------------------------------------------------------------- #
class RaveLiveModule(LiveModule):
    """Wrap a built ``RaveExport`` base so ``neural_tilde.LiveModule`` can register
    and export its ``encode`` / ``decode`` / ``forward`` methods."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.base.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.decode(self.base.encode(x))


def channel_labels(n: int) -> List[str]:
    return [f"(signal) Channel {d}" for d in range(1, n + 1)]


def latent_labels(n: int) -> List[str]:
    return [f"(signal) Latent dimension {i + 1}" for i in range(n)]


def build_live_module(run_arg: str, fidelity: float, ema_weights: bool,
                      buffer_size: int):
    """Build the streaming RAVE model, warm up its caches, and register the three
    methods on a ``RaveLiveModule``. Returns ``(module, buffer_size, run_ckpt)``
    (the buffer size may be snapped up to a multiple of the encode ratio)."""
    # streaming => cached_conv (stateful). Must precede rave.RAVE() construction.
    cc.use_cached_conv(True)

    config_file = rave.core.search_for_config(run_arg)
    if config_file is None:
        raise SystemExit(f"Config file not found in {run_arg}")
    gin.parse_config_file(config_file)
    run_ckpt = rave.core.search_for_run(run_arg)
    if run_ckpt is None:
        raise SystemExit(f"No checkpoint found in {run_arg}")

    pretrained = load_pretrained(run_ckpt, ema_weights)
    export_class = get_export_class(pretrained.encoder)
    latent_size = compute_latent_size(pretrained, fidelity)
    base = export_class(pretrained, latent_size, stochastic=False).eval()
    neutralize_for_export(base)
    logging.info("export mode: streaming deterministic")

    # fixed batch -> cache buffers become [1, c, pad] instead of [MAX_BATCH_SIZE, ...].
    # Set AFTER load (AdaIN registers [MAX_BATCH_SIZE,...] buffers that must match the
    # checkpoint) but BEFORE warmup, where init_cache allocates the caches.
    cc.convs.MAX_BATCH_SIZE = 1

    # discover the encode temporal ratio with a dummy pass (also a cache-init warmup)
    probe_len = 2 ** 14
    z = base.encode(torch.zeros(1, base.n_channels, probe_len))
    ratio_encode = probe_len // z.shape[-1]
    logging.info("latent_size=%d  encode ratio=%d", latent_size, ratio_encode)

    # buffer_size must be an integer multiple of the encode ratio
    if buffer_size % ratio_encode:
        buffer_size = ratio_encode * max(1, round(buffer_size / ratio_encode))
        logging.warning("buffer_size adjusted to %d (multiple of ratio %d)",
                        buffer_size, ratio_encode)
    latent_len = buffer_size // ratio_encode

    # shape sanity before export (these encode/decode passes double as cache warmup;
    # LiveModule.export_to_pte also runs a no_grad warmup forward per method before
    # tracing, so no separate forward-path warmup is needed here).
    enc_in = torch.zeros(1, base.n_channels, buffer_size)
    dec_in = torch.zeros(1, latent_size, latent_len)
    assert tuple(base.encode(enc_in).shape) == (1, latent_size, latent_len)
    assert tuple(base.decode(dec_in).shape) == (1, base.n_channels, buffer_size)

    zero_streaming_caches(base)  # serialize a defined zero initial state

    n_channels = base.n_channels
    module = RaveLiveModule(base).eval()
    module.register_method(
        "encode", in_channels=n_channels, in_ratio=1,
        out_channels=latent_size, out_ratio=ratio_encode,
        input_labels=channel_labels(n_channels),
        output_labels=latent_labels(latent_size))
    module.register_method(
        "decode", in_channels=latent_size, in_ratio=ratio_encode,
        out_channels=n_channels, out_ratio=1,
        input_labels=latent_labels(latent_size),
        output_labels=channel_labels(n_channels))
    module.register_method(
        "forward", in_channels=n_channels, in_ratio=1,
        out_channels=n_channels, out_ratio=1,
        input_labels=channel_labels(n_channels),
        output_labels=channel_labels(n_channels))
    return module, buffer_size, run_ckpt


def resolve_name(run_arg: str, run_ckpt: str, name: "str | None") -> str:
    if name is not None:
        return name
    normalized = os.path.normpath(run_arg)
    name = os.path.basename(normalized)
    if name.endswith(".ckpt"):
        name = run_ckpt.split(os.sep)[-4]
    return name


def main():
    ap = argparse.ArgumentParser(
        description="Export a trained RAVE model to ExecuTorch for neural.live~ ")
    ap.add_argument("--run", required=True,
                    help="path to the run directory or checkpoint to export")
    ap.add_argument("--name", default=None,
                    help="custom model name (default: run name)")
    ap.add_argument("--output", default=None,
                    help="output directory (default: the run/checkpoint dir)")
    ap.add_argument("--backend", default="coreml",
                    choices=["coreml", "xnnpack", "mlx", "portable"],
                    help="ExecuTorch delegate to lower to (default: coreml)")
    ap.add_argument("--buffer-size", type=int, default=4096,
                    help="audio-rate block size the program is exported at")
    ap.add_argument("--fidelity", type=float, default=0.95,
                    help="fidelity used to size the latent (variational models only)")
    ap.add_argument("--ema-weights", action="store_true",
                    help="use EMA weights if available")
    args = ap.parse_args()

    module, buffer_size, run_ckpt = build_live_module(
        args.run, args.fidelity, args.ema_weights, args.buffer_size)

    if args.backend == "mlx":
        # registers leaky_relu for the MLX delegate; neural_tilde's partitioner
        # factory does not import this (RAVE's old get_partitioners did).
        from scripts import mlx_ops_patch  # noqa: F401

    name = resolve_name(args.run, run_ckpt, args.name)
    output_dir = os.path.abspath(args.output or os.path.dirname(run_ckpt))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, name + ".pte")

    logging.info("exporting %s methods to %s (backend=%s, buffer_size=%d)",
                 module.get_methods(), out_path, args.backend, buffer_size)
    # strict=True: TorchDynamo specializes RAVE's buffer-conditioned branches
    # (AdaIN.learn_*, VariationalEncoder.warmed_up) on their frozen values, and the
    # fake-tensor trace keeps the live cached_conv caches at the zeroed initial state
    # across all three method exports.
    pte_path = module.export_to_pte(out_path, delegate=args.backend,
                                    buffer_size=buffer_size, batch=1, strict=True)
    logging.info("exported %s (+ sidecar %s.json)", pte_path, pte_path[:-4])


if __name__ == "__main__":
    main()

"""Verify a RAVE ExecuTorch export against the nn~ contract and PyTorch numerics.

Checks, against an exported ``<name>.pte`` + ``<name>.json`` pair:

  1. contract conformance  -- method set matches the sidecar; v1 invariants hold;
                              buffer_size divisible by every ratio; label counts.
  2. runtime execution     -- each method runs and returns the contracted shape,
                              all-finite (requires a runtime that supports the
                              delegate; portable / xnnpack work from the pip wheel,
                              mlx needs the source-built runtime).
  3. real audio            -- a real waveform reconstructs through ``forward`` with
                              a sane RMS.
  4. numeric parity        -- ExecuTorch output matches eager PyTorch (only valid
                              for a deterministic export -- the default).

Usage:
    uv run python scripts/verify_export.py <run_dir> <export_dir> \
        [--name NAME] [--wav PATH] [--buffer-size N] [--fidelity F] [--skip-parity]

``<export_dir>`` is where ``scripts/export_coreml.py`` wrote the ``.pte`` / ``.json``.
"""
import argparse
import json
import os
import sys

import gin
import torch
import torch.nn as nn
import cached_conv as cc

sys.path.insert(0, os.path.abspath("."))
import rave
import rave.core
from scripts.export import (compute_latent_size, get_export_class,
                            neutralize_for_export, zero_streaming_caches)
from executorch.runtime import Runtime


def load_eager_base(run_dir, fidelity):
    """Rebuild the eager export model (for parity comparison). Exports are always the
    stateful cached_conv model, so the cached_conv toggle is enabled before
    construction to match the exported program."""
    cc.use_cached_conv(True)  # must precede rave.RAVE()
    gin.parse_config_file(rave.core.search_for_config(run_dir))
    pretrained = rave.RAVE()
    state = torch.load(rave.core.search_for_run(run_dir), map_location="cpu")["state_dict"]
    pretrained.load_state_dict(state, strict=False)
    pretrained.eval()
    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    latent_size = compute_latent_size(pretrained, fidelity)
    base = get_export_class(pretrained.encoder)(pretrained, latent_size).eval()
    neutralize_for_export(base)
    return base, latent_size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="trained run folder (config.gin + checkpoint)")
    ap.add_argument("export_dir", help="folder containing the exported .pte / .json")
    ap.add_argument("--name", default=None, help="basename (default: run_dir basename)")
    ap.add_argument("--wav", default=None, help="audio file for the reconstruction test")
    ap.add_argument("--buffer-size", type=int, default=None,
                    help="block size (default: read from sidecar)")
    ap.add_argument("--fidelity", type=float, default=.95)
    ap.add_argument("--skip-parity", action="store_true",
                    help="skip numeric parity (block-by-block ET vs eager)")
    args = ap.parse_args()

    name = args.name or os.path.basename(os.path.normpath(args.run_dir))
    pte = os.path.join(args.export_dir, name + ".pte")
    json_path = os.path.join(args.export_dir, name + ".json")
    sidecar = json.load(open(json_path))
    buffer_size = args.buffer_size or sidecar["buffer_size"]

    base, latent_size = load_eager_base(args.run_dir, args.fidelity)
    ratio = 2 ** 14 // base.encode(torch.zeros(1, base.n_channels, 2 ** 14)).shape[-1]
    latent_len = buffer_size // ratio
    print("mode: live (stateful cached_conv)")

    ok = True
    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    prog = Runtime.get().load_program(pte)

    # ---------------------------------------------------------- 1. conformance
    print("\n[1] protocol conformance")
    check("method_names == JSON methods == {encode,decode,forward}",
          set(prog.method_names) == set(sidecar["methods"]) == {"encode", "decode", "forward"})
    check("kind == 'live'", sidecar.get("kind") == "live")
    check("no legacy streaming/latency keys (new protocol)",
          "streaming" not in sidecar
          and all("latency" not in m for m in sidecar["methods"].values()))
    check("attributes == []", sidecar["attributes"] == [])
    check("buffer_size % every ratio == 0",
          all(buffer_size % m["in_ratio"] == 0 and buffer_size % m["out_ratio"] == 0
              for m in sidecar["methods"].values()))
    for n, m in sidecar["methods"].items():
        check(f"{n}: label counts match channels",
              len(m["input_labels"]) == m["in_channels"]
              and len(m["output_labels"]) == m["out_channels"])

    # ----------------------------------------------------- 2. runtime exec
    print("\n[2] runtime execution (shape + finite)")
    expected = {
        "encode": ((1, base.n_channels, buffer_size), (1, latent_size, latent_len)),
        "decode": ((1, latent_size, latent_len), (1, base.n_channels, buffer_size)),
        "forward": ((1, base.n_channels, buffer_size), (1, base.n_channels, buffer_size)),
    }
    for n, (in_shape, out_shape) in expected.items():
        out = prog.load_method(n).execute([torch.zeros(*in_shape)])[0]
        check(f"{n}: out {tuple(out.shape)} == {out_shape} & finite",
              tuple(out.shape) == out_shape and bool(torch.isfinite(out).all()))

    # ------------------------------------------------------ 3. real audio
    if args.wav and os.path.isfile(args.wav):
        print("\n[3] real audio through forward")
        import soundfile as sf
        wav, _ = sf.read(args.wav, dtype="float32", always_2d=True)
        x = torch.from_numpy(wav.T)[:base.n_channels, :buffer_size].unsqueeze(0).contiguous()
        y = prog.load_method("forward").execute([x])[0]
        in_rms, out_rms = float(x.pow(2).mean().sqrt()), float(y.pow(2).mean().sqrt())
        check("forward(real audio) finite", bool(torch.isfinite(y).all()))
        check(f"reconstruction RMS sane (in={in_rms:.4f} out={out_rms:.4f})",
              0.02 * in_rms < out_rms < 50 * in_rms)

    # ----------------------- 4. state persistence + numeric parity vs eager
    print("\n[4] state persistence + block-by-block equivalence")
    # state carryover: same block twice => different output (mutable buffers persist)
    fwd = prog.load_method("forward")
    blk = torch.randn(1, base.n_channels, buffer_size)
    o1 = fwd.execute([blk.contiguous()])[0].clone()
    o2 = fwd.execute([blk.contiguous()])[0].clone()
    check("state persists across execute() (o1 != o2)",
          not torch.allclose(o1, o2, atol=1e-6))
    if not args.skip_parity:
        # ExecuTorch streaming (fresh method) vs eager cached_conv, block-by-block
        nb = 8
        torch.manual_seed(0)
        sig = torch.randn(1, base.n_channels, buffer_size * nb) * 0.1
        fwd2 = Runtime.get().load_program(pte).load_method("forward")  # fresh => zero state
        et = torch.cat([fwd2.execute([sig[..., i*buffer_size:(i+1)*buffer_size].contiguous()])[0].clone()
                        for i in range(nb)], -1)
        with torch.no_grad():
            zero_streaming_caches(base)
            eager = torch.cat([base.decode(base.encode(sig[..., i*buffer_size:(i+1)*buffer_size]))
                               for i in range(nb)], -1)
        maxdiff = float((eager - et).abs().max())
        check(f"ET streaming == eager streaming (maxdiff={maxdiff:.2e})",
              torch.allclose(eager, et, atol=1e-4, rtol=1e-3))

    print("\n==>", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

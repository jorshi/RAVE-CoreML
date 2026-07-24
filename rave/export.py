"""RAVE export-model helpers (library).

This module holds the RAVE-specific machinery for turning a trained ``rave.RAVE``
into a stateless encode/decode/forward model ready for ExecuTorch lowering:
encoder-family wrappers (variational / discrete / wasserstein / spherical),
branch neutralization for ``torch.export``, latent sizing, cached_conv cache
zeroing, and checkpoint loading.

The CLI that drives an actual export now lives in ``scripts/export_coreml.py``,
which reuses ``neural_tilde.LiveModule`` to emit the ExecuTorch ``.pte`` + sidecar
(see ``neural_tilde``'s ``EXECUTORCH_PROTOCOL.md``). The functions here are
imported by ``export_coreml.py`` and ``verify_export.py``.
"""

import logging
import math
import os
import sys
import types

logging.basicConfig(level=logging.INFO)

import torch

torch.set_grad_enabled(False)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

try:
    import rave
except ImportError:
    sys.path.append(os.path.abspath("."))
    import rave
import rave.blocks


# --------------------------------------------------------------------------- #
# Pure encode / decode pipeline (no stereo / adain-update / resampler / state) #
# --------------------------------------------------------------------------- #
class RaveExport(nn.Module):
    """Stateless encode/decode wrapper around a pretrained ``rave.RAVE``.

    Subclasses implement ``post_process_latent`` / ``pre_process_latent`` for the
    specific encoder family (variational, discrete, wasserstein, spherical),
    mirroring the old ``ScriptedRAVE`` subclasses.
    """

    def __init__(self, pretrained: "rave.RAVE", latent_size: int,
                 stochastic: bool = False) -> None:
        super().__init__()
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        self.pqmf = pretrained.pqmf
        self.input_mode = pretrained.input_mode
        self.output_mode = pretrained.output_mode
        self.n_channels = pretrained.n_channels
        self.latent_size = latent_size
        self.full_latent_size = pretrained.latent_size
        self.stochastic = stochastic
        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)

    def _augmentation_noise(self, n_batch: int, n_dims: int, length: int,
                            ref: torch.Tensor) -> torch.Tensor:
        """Padding/augmentation tensor: randn when stochastic, else zeros."""
        if self.stochastic:
            return torch.randn(n_batch, n_dims, length).type_as(ref)
        return torch.zeros(n_batch, n_dims, length).type_as(ref)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        elif self.input_mode == "mel":
            x = self.encoder.spectrogram(x)[..., :-1]
            x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
        z = self.encoder(x)
        return self.post_process_latent(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.pre_process_latent(z)
        batch_size = z.shape[:-2]
        y = self.decoder(z)
        if self.output_mode == "pqmf":
            y = y.reshape(y.shape[0] * self.n_channels, -1, y.shape[-1])
            y = self.pqmf.inverse(y)
            y = y.reshape(batch_size + (self.n_channels, -1))
        return y

    def post_process_latent(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pre_process_latent(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VariationalRaveExport(RaveExport):

    def post_process_latent(self, z):
        if self.stochastic:
            z = self.encoder.reparametrize(z)[0]
        else:
            z = z.chunk(2, 1)[0]  # deterministic: latent mean (no sampling)
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        return z

    def pre_process_latent(self, z):
        pad = self._augmentation_noise(z.shape[0], self.full_latent_size - z.shape[1],
                                       z.shape[-1], z)
        z = torch.cat([z, pad], 1)
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        return z


class DiscreteRaveExport(RaveExport):

    def post_process_latent(self, z):
        return self.encoder.rvq.encode(z).float()

    def pre_process_latent(self, z):
        z = torch.clamp(z, 0, self.encoder.rvq.layers[0].codebook_size - 1).long()
        z = self.encoder.rvq.decode(z)
        if self.encoder.noise_augmentation:
            noise = self._augmentation_noise(z.shape[0], self.encoder.noise_augmentation,
                                             z.shape[-1], z)
            z = torch.cat([z, noise], 1)
        return z


class WasserteinRaveExport(RaveExport):

    def post_process_latent(self, z):
        return z

    def pre_process_latent(self, z):
        if self.encoder.noise_augmentation:
            noise = self._augmentation_noise(z.shape[0], self.encoder.noise_augmentation,
                                             z.shape[-1], z)
            z = torch.cat([z, noise], 1)
        return z


class SphericalRaveExport(RaveExport):

    def post_process_latent(self, z):
        return rave.blocks.unit_norm_vector_to_angles(z)

    def pre_process_latent(self, z):
        return rave.blocks.angles_to_unit_norm_vector(z)


_EXPORT_CLASSES = {
    rave.blocks.VariationalEncoder: VariationalRaveExport,
    rave.blocks.DiscreteEncoder: DiscreteRaveExport,
    rave.blocks.WasserteinEncoder: WasserteinRaveExport,
    rave.blocks.SphericalEncoder: SphericalRaveExport,
}


def _adain_identity(self, x):
    return x


def _encoder_passthrough(self, x):
    return self.encoder(x)


def neutralize_for_export(module: nn.Module) -> None:
    """Replace inference-irrelevant, buffer-conditioned branches with branch-free
    equivalents so ``torch.export`` (Dynamo) can trace the graph.

    Dynamo treats ``if <tensor buffer>:`` as data-dependent control flow and refuses
    to trace it, even when the buffer is frozen. The two offenders in RAVE's
    encode/decode path are both no-ops at (stateless v1) inference time:

    - ``AdaptiveInstanceNormalization`` -> identity. v1 has no timbre transfer, and
      with the default (zeroed) ``num_update`` buffers AdaIN already returns its input.
    - encoder wrappers (``VariationalEncoder`` etc.) -> drop the
      ``if self.warmed_up: z = z.detach()`` branch (``detach`` is a no-op here).
    """
    patched = []
    for m in module.modules():
        if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
            m.forward = types.MethodType(_adain_identity, m)
            patched.append("AdaIN")
        elif hasattr(m, "warmed_up") and hasattr(m, "encoder"):
            m.forward = types.MethodType(_encoder_passthrough, m)
            patched.append(type(m).__name__)
    logging.info("neutralized buffer-conditioned branches for export: %s", patched)


def get_export_class(encoder) -> type:
    for encoder_type, export_class in _EXPORT_CLASSES.items():
        if isinstance(encoder, encoder_type):
            return export_class
    raise ValueError(f"Encoder type {type(encoder).__name__} not supported for export.")


def compute_latent_size(pretrained, fidelity: float) -> int:
    """Latent dimensionality kept at export, per encoder family."""
    encoder = pretrained.encoder
    if isinstance(encoder, rave.blocks.VariationalEncoder):
        include_latent = pretrained.fidelity.numpy() > fidelity
        if np.all(~include_latent):
            latent_size = len(pretrained.fidelity)
        else:
            latent_size = max(np.argmax(pretrained.fidelity.numpy() > fidelity), 1)
        return int(2 ** math.ceil(math.log2(latent_size)))
    if isinstance(encoder, rave.blocks.DiscreteEncoder):
        return int(encoder.num_quantizers)
    if isinstance(encoder, rave.blocks.WasserteinEncoder):
        return int(pretrained.latent_size)
    if isinstance(encoder, rave.blocks.SphericalEncoder):
        return int(pretrained.latent_size - 1)
    raise ValueError(f"Encoder type {type(encoder).__name__} not supported for export.")


def zero_streaming_caches(module: nn.Module) -> None:
    """Zero every cached_conv state buffer so the exported mutable buffers serialize a
    defined zero initial state (paired with InitializedMutableBufferPass)."""
    for m in module.modules():
        pad = getattr(m, "pad", None)
        if torch.is_tensor(pad):
            pad.zero_()
        cache = getattr(m, "cache", None)
        if torch.is_tensor(cache):
            cache.zero_()


def load_pretrained(run_ckpt: str, ema_weights: bool) -> "rave.RAVE":
    """Instantiate rave.RAVE() (under the current cached_conv toggle), load weights, eval,
    and fold weight_norm into plain weights. gin must already be parsed."""
    pretrained = rave.RAVE()
    checkpoint = torch.load(run_ckpt, map_location="cpu")
    if ema_weights and "EMA" in checkpoint.get("callbacks", {}):
        pretrained.load_state_dict(checkpoint["callbacks"]["EMA"], strict=False)
    else:
        pretrained.load_state_dict(checkpoint["state_dict"], strict=False)
    pretrained.eval()
    for module in pretrained.modules():
        if hasattr(module, "weight_g"):
            nn.utils.remove_weight_norm(module)
    return pretrained

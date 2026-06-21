"""Extra ExecuTorch MLX-delegate op handlers, registered at import time.

RAVE streaming models click on the MLX delegate because cached_conv state doesn't
persist across execute(). Root cause: the MLX partitioner can't handle
``aten.leaky_relu`` (RAVE's activations, ~176 nodes), so it shatters each method
into dozens of subgraphs; a cached_conv buffer then gets written-as-mutable in one
subgraph but read-as-constant (zero) in another, so state never flows.

Registering a ``leaky_relu`` handler that decomposes into existing MLX IR nodes
(maximum / minimum / multiply / add — all already implemented in the C++ runtime)
makes the partitioner delegate those nodes, collapsing the fragmentation so the
cached_conv read+write stay in the same subgraph and the per-subgraph mutable
buffers persist. No schema change, no generate.py, no runtime rebuild, and no
change to the nn~ contract or the .pte method signatures.

Import this module before lowering (scripts/export.py does, for --backend mlx).
"""
import logging

import torch
from executorch.backends.mlx.builder.op_helpers import emit_lifted_constant
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    AddNode, MaximumNode, MinimumNode, MultiplyNode)

logger = logging.getLogger(__name__)


@REGISTRY.register(target=[torch.ops.aten.leaky_relu.default])
def _leaky_relu_handler(P, n):
    """leaky_relu(x, s) = maximum(x, 0) + s * minimum(x, 0).

    Built from existing MLX nodes so the runtime needs no new op. The negative
    slope is read from the node (RAVE uses its own value, not the 0.01 default).
    """
    args = P.args(n)
    x = args[0]
    if len(args) > 1:
        slope = args[1]
    else:
        slope = P.kwargs(n).get("negative_slope", 0.01)

    val = n.args[0].meta.get("val")
    dtype = val.dtype if val is not None else torch.float32

    zero = emit_lifted_constant(P, 0.0, dtype)
    slope_const = emit_lifted_constant(P, float(slope), dtype)

    _, pos = P.make_tmp_slot()
    P.emit(MaximumNode(a=P.slot_to_tid(x), b=P.slot_to_tid(zero),
                       out=P.slot_to_tid(pos)))
    _, neg = P.make_tmp_slot()
    P.emit(MinimumNode(a=P.slot_to_tid(x), b=P.slot_to_tid(zero),
                       out=P.slot_to_tid(neg)))
    _, scaled = P.make_tmp_slot()
    P.emit(MultiplyNode(a=P.slot_to_tid(neg), b=P.slot_to_tid(slope_const),
                        out=P.slot_to_tid(scaled)))
    out = P.make_or_get_slot(n)
    P.emit(AddNode(a=P.slot_to_tid(pos), b=P.slot_to_tid(scaled),
                   out=P.slot_to_tid(out)))
    return out


logger.info("nn~ MLX op patch: registered aten.leaky_relu")

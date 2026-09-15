"""Canonical flat prototype-weight layout.

ONE source of truth for how a prototype row packs per-layer conv weights and
the classifier head::

    [layer i]   w1: max_dim*in_c*9 | b1: max_dim      (both block modes share this)
                w2: max_dim*max_dim*9 | b2: max_dim   (residual branch; C_in = max_dim)
    [head]      W: (16*max_dim)*num_classes | b: num_classes

v2 layout: every layer also carries the SECOND 3x3 convolution of the real
ResNet block (plain = ReLU(GN(conv1(h))); residual = ReLU(GN(conv2(plain)) + skip)),
so the two block modes are genuinely different in parameters and FLOPs and
ModelStructure.effective_params_formula can charge them apart. The w2/b2
slots exist for EVERY layer, including layer 0 (whose residual skip is taken
from the conv1 output, matching the executor).

Consumers that must agree byte-for-byte: PrototypeLibrary init/slicing
(mgn.generator), AcceleratedCVNetwork + fused CUDA kernels via w_offsets/
b_offsets (AML_Core_Code -- the kernels consume the offsets and index conv1
slots only; get_second_conv_layout feeds the second conv), extract_static_model,
and the eager reference in validate_ext_in. Any layout change here MUST be
mirrored in mgn/csrc kernel indexing and the fused autograd wrappers. Old (v1)
checkpoints are incompatible and fail the row-width assert at load time.
"""

def _walk_layout(max_layers: int, max_dim: int, num_classes: int, in_channels: int):
    """Single packing walk; both public accessors derive from this."""
    offsets = []
    offsets2 = []
    idx = 0
    in_c = in_channels
    for _ in range(max_layers):
        out_c = max_dim
        w_size, b_size = out_c * in_c * 9, out_c
        w_start, w_end = idx, idx + w_size
        idx += w_size
        b_start, b_end = idx, idx + b_size
        idx += b_size
        offsets.append((w_start, w_end, b_start, b_end, in_c))

        w2_size, b2_size = out_c * max_dim * 9, out_c
        w2_start, w2_end = idx, idx + w2_size
        idx += w2_size
        b2_start, b2_end = idx, idx + b2_size
        idx += b2_size
        offsets2.append((w2_start, w2_end, b2_start, b2_end))

        in_c = out_c

    wo_offset = idx
    bo_offset = idx + (16 * max_dim) * num_classes
    total_dim = bo_offset + num_classes
    return offsets, offsets2, wo_offset, bo_offset, total_dim


def get_prototype_layout(max_layers: int, max_dim: int, num_classes: int,
                         in_channels: int = 3):
    """Return (total_dim, offsets, layer_info, wo_offset, bo_offset).

    offsets[i]    = (w_start, w_end, b_start, b_end, in_channels_i)
    layer_info[i] = (w_start, w_end, b_start, b_end, in_channels_i, out_channels_i)
    wo/bo_offset  = slice bounds of the classifier weight/bias blocks.

    These are the FIRST-convolution slots. The second (residual-branch) conv
    slots interleaved between layers come from get_second_conv_layout.
    """
    offsets, _offsets2, wo_offset, bo_offset, total_dim = _walk_layout(
        max_layers, max_dim, num_classes, in_channels)
    layer_info = [(ws, we, bs, be, in_c, max_dim)
                  for (ws, we, bs, be, in_c) in offsets]
    return total_dim, offsets, layer_info, wo_offset, bo_offset


def get_second_conv_layout(max_layers: int, max_dim: int, num_classes: int,
                           in_channels: int = 3):
    """Return [(w_start, w_end, b_start, b_end)] per layer for the SECOND conv.

    Must be walked with the SAME (num_classes, in_channels) that produced the
    row via get_prototype_layout: layer 0's smaller input width shifts every
    later offset, and the classifier tail sits after all layers.
    """
    _offsets, offsets2, _wo, _bo, _total = _walk_layout(
        max_layers, max_dim, num_classes, in_channels)
    return offsets2

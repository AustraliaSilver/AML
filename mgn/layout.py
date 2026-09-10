"""Canonical flat prototype-weight layout.

ONE source of truth for how a prototype row packs per-layer conv weights and
the classifier head::

    [layer 0]   w: max_dim*in_channels*9 | b: max_dim
    [layer i>0] w: max_dim*max_dim*9     | b: max_dim
    [head]      W: (16*max_dim)*num_classes | b: num_classes

The 16* slots in the head come from AdaptiveAvgPool2d((4, 4)). Consumers that
must agree byte-for-byte: PrototypeLibrary init/slicing (mgn.generator),
AcceleratedCVNetwork + fused CUDA kernels via w_offsets/b_offsets
(AML_Core_Code), extract_static_model, and the eager reference in
validate_ext_in. Any layout change here MUST be mirrored in mgn/csrc
kernel indexing (they consume the offsets, not the formula) and the fused
autograd wrappers.
"""

def get_prototype_layout(max_layers: int, max_dim: int, num_classes: int,
                         in_channels: int = 3):
    """Return (total_dim, offsets, layer_info, wo_offset, bo_offset).

    offsets[i]    = (w_start, w_end, b_start, b_end, in_channels_i)
    layer_info[i] = (w_start, w_end, b_start, b_end, in_channels_i, out_channels_i)
    wo/bo_offset  = slice bounds of the classifier weight/bias blocks.

    Signature and return shapes are identical to the historical
    train_cv_cifar_18.get_prototype_layout (which now re-exports this), so
    CUDA offset vectors and slicing call sites stay byte-compatible.
    """
    offsets = []
    layer_info = []
    idx = 0
    in_c = in_channels
    for _ in range(max_layers):
        out_c = max_dim
        w_size = out_c * in_c * 9
        b_size = out_c
        w_start, w_end = idx, idx + w_size
        idx += w_size
        b_start, b_end = idx, idx + b_size
        idx += b_size

        layer_info.append((w_start, w_end, b_start, b_end, in_c, out_c))
        offsets.append((w_start, w_end, b_start, b_end, in_c))

        in_c = out_c

    wo_offset = idx
    bo_offset = idx + (16 * max_dim) * num_classes
    total_dim = bo_offset + num_classes
    return total_dim, offsets, layer_info, wo_offset, bo_offset

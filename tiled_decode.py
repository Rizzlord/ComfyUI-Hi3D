"""Memory-light ("tiled") decode for the Hi3D sparse VAE.

The decoder's GroupNorm normalizes over every point of the shape, so splitting the
shape into spatial tiles would change its statistics and leave seams. Instead each
large layer is evaluated in chunks of points: GroupNorm accumulates its statistics
over all chunks before normalizing, sparse convolutions gather neighbours from the
full feature matrix one chunk of rows at a time, and upsampling projects and prunes
children chunk by chunk. The math is unchanged; only fewer full-size copies of the
feature matrix are alive at once.

:func:`tiled_decode` swaps the forwards of the VAE's modules for the duration of a
decode and restores the originals afterwards.
"""

from __future__ import annotations

import contextlib
import types
from math import prod

import torch

from sparc3d.models.autoencoders import blocks
from sparc3d.ops import sparse_backend
from sparc3d.structures.point_modules import PointSequential
from sparc3d.utils.misc import batch2offset

CHUNK_ROWS = 262_144


def _row_chunks(n: int):
    for start in range(0, n, CHUNK_ROWS):
        yield start, min(start + CHUNK_ROWS, n)


def _group_norm(self, point, out_dtype=None):
    """PointGroupNorm with two-pass global statistics and chunked normalization."""
    feat = point.feat
    if int(point.batch[-1]) != 0:
        return type(self).forward(self, point)
    n, channels = feat.shape
    groups, per_group = self.group, self.channels_per_group
    sums = torch.zeros(groups, dtype=torch.float64, device=feat.device)
    sums_sq = torch.zeros_like(sums)
    for start, stop in _row_chunks(n):
        x = feat[start:stop].float().view(-1, groups, per_group)
        sums += x.sum(dim=(0, 2))
        sums_sq += x.square().sum(dim=(0, 2))
    count = n * per_group
    mean = sums / (count + self.eps)
    var = sums_sq / (count + self.eps) - mean.square()
    mean = mean.float().view(1, groups, 1)
    inv_std = torch.rsqrt(var.float() + self.eps).view(1, groups, 1)

    out = torch.empty((n, channels), dtype=out_dtype or feat.dtype, device=feat.device)
    with torch.autocast(device_type="cuda", enabled=False):
        for start, stop in _row_chunks(n):
            x = feat[start:stop].float().view(-1, groups, per_group)
            x = ((x - mean) * inv_std).view(-1, channels)
            out[start:stop] = x.mul_(self.weights).add_(self.bias)
    point.update_feat(out)
    return point


def _swish_inplace(self, x):
    """Swish written into its input; only installed where the input is a fresh GroupNorm output."""
    for start, stop in _row_chunks(x.shape[0]):
        chunk = x[start:stop]
        chunk.mul_(torch.sigmoid(chunk))
    return x


def _subm_conv(self, input):
    """Submanifold convolution that casts and gathers features one chunk of rows at a time.

    Same pair traversal as ``SubMConv3d._large_forward`` in ``sparc3d.ops.sparse_backend``,
    but the input stays in its own dtype instead of being copied to the weight dtype.
    """
    from spconv.pytorch import ConvAlgo, ops
    from spconv.pytorch.core import IndiceData

    indice_key = f"{self.indice_key}_large" if self.indice_key is not None else None
    data = input.find_indice_pair(indice_key)
    if data is None:
        out_indices, pairs, pair_counts = ops.get_indice_pairs(
            input.indices, input.batch_size, input.spatial_shape, ConvAlgo.Native,
            self.kernel_size, self.stride, self.padding, self.dilation, self.output_padding,
            True, False,
        )
        data = IndiceData(
            out_indices, input.indices, pairs, pair_counts, input.spatial_shape, input.spatial_shape,
            is_subm=True, algo=ConvAlgo.Native, ksize=self.kernel_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
        )
    else:
        out_indices, pairs, pair_counts = data.out_indices, data.indice_pairs, data.indice_pair_num

    features = input.features
    kernel_volume = prod(self.kernel_size)
    weights = self.weight.reshape(self.out_channels, kernel_volume, self.in_channels)
    center = kernel_volume // 2
    with torch.autocast(device_type="cuda", enabled=False):
        output = torch.empty((features.shape[0], self.out_channels), dtype=weights.dtype, device=features.device)
        for start, stop in _row_chunks(len(features)):
            output[start:stop] = features[start:stop].to(weights.dtype) @ weights[:, center, :].T

        def accumulate(source, target, weight):
            for start, stop in _row_chunks(len(source)):
                contribution = features[source[start:stop]].to(weight.dtype) @ weight.T
                output.index_add_(0, target[start:stop], contribution)

        for offset in range(center):
            count = int(pair_counts[offset])
            source = pairs[0, offset, :count].long()
            target = pairs[1, offset, :count].long()
            accumulate(source, target, weights[:, offset, :])
            accumulate(target, source, weights[:, kernel_volume - 1 - offset, :])
            del source, target
        if self.bias is not None:
            output.add_(self.bias)
        if output.dtype != features.dtype:
            output = output.to(features.dtype)

    result = input.replace_feature(output)
    if indice_key is not None:
        result.indice_dict[indice_key] = data
    result.indices = out_indices
    return result


def _resnet_block(self, point):
    """SparseResnetBlock adding the shortcut into the branch output in place."""
    if point.feat.shape[0] > self.skip_points:
        return type(self).forward(self, point)
    shortcut = self.res_conv(point.feat) if self.res_conv is not None else point.feat
    out = self.block(point).feat
    if out.dtype == torch.promote_types(out.dtype, shortcut.dtype):
        out.add_(shortcut)
    else:
        out = shortcut + out
    point.update_feat(out)
    return point


def _unpooling(self, point):
    """SerializedUnpooling that projects and prunes the eight children chunk by chunk."""
    feat_in = point.feat
    keep = self.cls_head(point).feat > 0
    parents = feat_in.shape[0]
    channels = self.proj.out_features // 8
    total = int(keep.sum())

    feat = None
    filled = 0
    for start, stop in _row_chunks(parents):
        children = self.proj(feat_in[start:stop]).view(-1, channels)[keep[start:stop].flatten()]
        if feat is None:
            feat = torch.empty((total, channels), dtype=children.dtype, device=children.device)
        feat[filled:filled + len(children)] = children
        filled += len(children)
    if feat is None:
        feat = feat_in.new_empty((0, channels))
    del feat_in

    keep = keep.flatten()
    grid_coord = (point.grid_coord.unsqueeze(1) * 2 + self.inv_grid_coord.unsqueeze(0)).view(-1, 3)[keep]
    batch = point.batch.unsqueeze(1).expand(-1, 8).flatten()[keep]
    point = point.clone(
        feat=feat,
        grid_coord=grid_coord,
        resolution=point.resolution * 2,
        batch=batch,
        offset=batch2offset(batch),
    )
    point.sparsify()
    return point


def _conv_out(self, point):
    """ConvOut normalizing straight into float32 instead of copying the input to float32 first."""
    norm, act, conv = self.conv._modules.values()
    _group_norm(norm, point, out_dtype=torch.float32)
    _swish_inplace(act, point.feat)
    point.sparse_conv_feat = _subm_conv(conv, point.sparse_conv_feat)
    point.feat = sparse_backend.get_feats(point.sparse_conv_feat)
    point["sdf_sign"] = point.feat[:, :8]
    point.update_feat(point.feat[:, 8:])
    return point


@contextlib.contextmanager
def tiled_decode(vae):
    """Run every decode inside this context with the chunked layer implementations."""
    patched = []

    def patch(module, fn):
        module.forward = types.MethodType(fn, module)
        patched.append(module)

    for module in vae.modules():
        if isinstance(module, blocks.PointGroupNorm):
            patch(module, _group_norm)
        elif isinstance(module, blocks.SparseResnetBlock):
            patch(module, _resnet_block)
        elif isinstance(module, blocks.SerializedUnpooling):
            patch(module, _unpooling)
        elif isinstance(module, blocks.ConvOut):
            patch(module, _conv_out)
        elif sparse_backend.is_sparse_conv(module):
            patch(module, _subm_conv)
        elif isinstance(module, PointSequential):
            children = list(module._modules.values())
            for previous, child in zip(children, children[1:]):
                if isinstance(child, blocks.Swish) and isinstance(previous, blocks.PointGroupNorm):
                    patch(child, _swish_inplace)
    try:
        yield
    finally:
        for module in patched:
            del module.forward

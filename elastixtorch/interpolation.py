"""B-spline interpolation of a 3D volume at arbitrary (continuous, differentiable) voxel coordinates.

Order 1 is trilinear interpolation, order 3 is cubic B-spline interpolation (as the final interpolator of
Elastix). Coordinates are in voxel units: ``(0, 0, 0)`` is the centre of the first voxel of the tensor.
Gradients flow to the coordinates, which is what the registration needs.
"""
import itertools
import math
from typing import Tuple

import torch
import torch.nn.functional as F

_POLE = math.sqrt(3.0) - 2.0
_TAPS = 14                                    # |pole|**14 ~ 1e-8: the truncated inverse filter is exact in float32


def bspline_weights(t: torch.Tensor, order: int) -> torch.Tensor:
    """Weights of the ``order + 1`` B-spline coefficients around a position, for the fractional part
    ``t`` in [0, 1) (shape ``(N,)``) -> ``(N, order + 1)``. The coefficients are the ones at
    ``floor(x) - (order - 1) // 2 + (0 .. order)``."""
    if order == 1:
        return torch.stack([1 - t, t], dim=-1)
    if order == 3:
        t2, t3 = t * t, t * t * t
        return torch.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6,
                            (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], dim=-1)
    raise ValueError(f"the B-spline order must be 1 or 3, got {order}")


def cubic_prefilter(volume: torch.Tensor) -> torch.Tensor:
    """B-spline coefficients of a volume, such that interpolating the coefficients with the cubic B-spline
    kernel passes through the voxel values. The inverse of the ``[1, 4, 1] / 6`` filter is a symmetric
    exponential kernel, applied here as a truncated convolution along each axis (mirror boundaries)."""
    gain = 6 * _POLE / (_POLE ** 2 - 1)
    taps = gain * _POLE ** torch.arange(-_TAPS, _TAPS + 1, device=volume.device).abs().to(volume.dtype)
    kernel = taps.to(volume.dtype)[None, None]
    out = volume
    for axis in range(volume.ndim):
        moved = out.movedim(axis, -1)
        length = moved.shape[-1]
        pad = min(_TAPS, length - 1)
        flat = moved.reshape(-1, 1, length)
        flat = F.pad(flat, (pad, pad), mode="reflect") if pad > 0 else flat
        k = kernel[..., _TAPS - pad:_TAPS + pad + 1]
        filtered = F.conv1d(flat, k)
        out = filtered.reshape(moved.shape).movedim(-1, axis)
    return out


def _mirror(index: torch.Tensor, n: int) -> torch.Tensor:
    """Index of the neighbour at the borders: whole-sample mirror (the boundary of the prefilter)."""
    if n == 1:
        return torch.zeros_like(index)
    period = 2 * (n - 1)
    index = index.abs() % period
    return torch.where(index > n - 1, period - index, index)


def sample(coefficients: torch.Tensor, coordinates: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Interpolated values at ``coordinates`` (``(N, 3)``, in voxels). ``coefficients`` is the volume itself
    for order 1 and ``cubic_prefilter(volume)`` for order 3. Beyond the border the volume is mirrored."""
    n_points = coordinates.shape[0]
    shape = coefficients.shape
    base = torch.floor(coordinates.detach())
    frac = coordinates - base
    base = base.long()
    offsets = torch.arange(order + 1, device=coordinates.device) - (order - 1) // 2
    weights, indices = [], []
    for axis in range(3):
        weights.append(bspline_weights(frac[:, axis], order))
        indices.append(_mirror(base[:, axis, None] + offsets, shape[axis]))
    flat_index = (indices[0][:, :, None, None] * shape[1] + indices[1][:, None, :, None]) * shape[2] \
        + indices[2][:, None, None, :]
    weight = weights[0][:, :, None, None] * weights[1][:, None, :, None] * weights[2][:, None, None, :]
    values = coefficients.reshape(-1)[flat_index.reshape(n_points, -1)]
    return (values * weight.reshape(n_points, -1)).sum(dim=1)


def inside(coordinates: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Boolean mask of the coordinates (``(N, 3)``, in voxels) that lie within the volume."""
    upper = torch.tensor([s - 1 for s in shape], device=coordinates.device, dtype=coordinates.dtype)
    return ((coordinates >= 0) & (coordinates <= upper)).all(dim=1)


def sample_linear(volume: torch.Tensor, coordinates: torch.Tensor, gradient: bool = False):
    """Trilinear interpolation for positions inside the volume (``inside(coordinates, volume.shape)``): a single
    gather of the 8 neighbours, with no border handling. Same values as ``sample(volume, coordinates, 1)``
    inside the volume, faster. With ``gradient=True``, also returns the spatial gradient ``(N, 3)`` (per voxel)
    of the interpolated image, computed from the derivative of the weights.

    The work is laid out component by component (``(3, N)`` and ``(8, N)`` tensors, contiguous along the points):
    with a few thousand points, broadcasting over small trailing dimensions is several times slower."""
    shape = volume.shape
    device = coordinates.device
    c = coordinates.T                                                   # (3, N)
    upper = torch.tensor([max(n - 2, 0) for n in shape], device=device)[:, None]
    base = torch.minimum(torch.floor(c.detach()).long().clamp_min(0), upper)
    t = c - base                                                        # (3, N), in [0, 1] inside the volume
    strides = [shape[1] * shape[2] if shape[0] > 1 else 0, shape[2] if shape[1] > 1 else 0, 1 if shape[2] > 1 else 0]
    offsets = torch.tensor([a * strides[0] + b * strides[1] + k * strides[2]
                            for a, b, k in itertools.product((0, 1), repeat=3)], device=device)
    index = base[0] * (shape[1] * shape[2]) + base[1] * shape[2] + base[2]
    values = torch.take(volume, offsets[:, None] + index[None, :]).reshape(2, 2, 2, -1)    # (2, 2, 2, N)
    one = 1 - t
    wx, wy, wz = torch.stack([one[0], t[0]]), torch.stack([one[1], t[1]]), torch.stack([one[2], t[2]])  # (2, N)
    wyz = wy[:, None] * wz[None, :]                                     # (2, 2, N)
    out = (values * (wx[:, None, None] * wyz[None])).sum((0, 1, 2))
    if not gradient:
        return out
    # derivative of the weights along one axis: (-1, +1) in place of (1 - t, t)
    dx = values[1] - values[0]                                          # (2, 2, N)
    gx = (dx * wyz).sum((0, 1))
    vy = (values * wx[:, None, None]).sum(0)                            # (2, 2, N), axis 0 interpolated
    gy = ((vy[1] - vy[0]) * wz).sum(0)
    vz = (vy * wy[:, None]).sum(0)                                      # (2, N), axes 0 and 1 interpolated
    gz = vz[1] - vz[0]
    return out, torch.stack([gx, gy, gz], dim=1)


def bspline_derivative_weights(t: torch.Tensor) -> torch.Tensor:
    """Derivative of the cubic B-spline weights of `bspline_weights` w.r.t. the position, ``(N, 4)``."""
    t2 = t * t
    return torch.stack([-(1 - t) ** 2 / 2, 1.5 * t2 - 2 * t, -1.5 * t2 + t + 0.5, t2 / 2], dim=-1)

"""Multi-resolution pyramid: Gaussian smoothing followed by a downsampling, as in Elastix."""
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from .interpolation import sample_linear


@dataclass
class Level:
    """A volume on its own grid: the voxel ``i`` is at the physical position ``origin + spacing * i``."""
    image: torch.Tensor
    spacing: torch.Tensor          # (3,)
    origin: torch.Tensor           # (3,)


def gaussian_smooth(volume: torch.Tensor, sigma: Sequence[float]) -> torch.Tensor:
    """Separable Gaussian smoothing, ``sigma`` in voxels for each axis (0 leaves the axis unchanged)."""
    out = volume
    for axis, s in enumerate(sigma):
        if s <= 0:
            continue
        radius = max(1, int(math.ceil(3 * s)))
        x = torch.arange(-radius, radius + 1, device=volume.device, dtype=volume.dtype)
        kernel = torch.exp(-0.5 * (x / s) ** 2)
        kernel = (kernel / kernel.sum())[None, None]
        moved = out.movedim(axis, -1)
        length = moved.shape[-1]
        pad = min(radius, length - 1)
        flat = F.pad(moved.reshape(-1, 1, length), (pad, pad), mode="replicate")
        k = kernel[..., radius - pad:radius + pad + 1]
        k = k / k.sum()
        out = F.conv1d(flat, k).reshape(moved.shape).movedim(-1, axis)
    return out


def schedule(num_resolutions: int, shape: Sequence[int], min_size: int = 8) -> Tuple[Tuple[int, ...], ...]:
    """Downsampling factor of each axis at each resolution (coarsest first): ``2 ** (R - 1 - r)``, limited so
    that no axis has fewer than ``min_size`` voxels."""
    levels = []
    for r in range(num_resolutions):
        factor = 2 ** (num_resolutions - 1 - r)
        levels.append(tuple(max(1, min(factor, int(n) // min_size)) for n in shape))
    return tuple(levels)


def build_pyramid(volume: torch.Tensor, spacing: torch.Tensor, origin: torch.Tensor, factors):
    """All the levels of the pyramid (coarsest first) for the per-level `factors` of `schedule`.

    Each level is the volume smoothed (sigma = 0.5 * factor voxels, as the Elastix smoothing pyramid) and
    resampled on a grid ``factor`` times coarser, whose voxels are centred on the blocks of ``factor`` fine voxels.
    It is computed from the next finer level: the extra smoothing is ``sqrt(sigma_coarse**2 - sigma_fine**2)``
    (Gaussian smoothings add in quadrature), on a volume that is already smaller, so the cost of the coarse levels
    is negligible."""
    levels = [None] * len(factors)
    previous, previous_factors = Level(volume, spacing.clone(), origin.clone()), (1, 1, 1)
    for r in reversed(range(len(factors))):
        f = tuple(int(x) for x in factors[r])
        if f == previous_factors:
            levels[r] = previous
            continue
        sigma_of = lambda k: 0.5 * k if k > 1 else 0.0
        sigma = [math.sqrt(max(sigma_of(a) ** 2 - sigma_of(g) ** 2, 0.0)) / g for a, g in zip(f, previous_factors)]
        smooth = gaussian_smooth(previous.image, sigma)
        sizes = [int(n) // a for n, a in zip(volume.shape, f)]
        axes = [(torch.arange(n, device=volume.device, dtype=volume.dtype) * a + (a - 1) / 2 - (g - 1) / 2) / g
                for n, a, g in zip(sizes, f, previous_factors)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        grid = torch.minimum(grid, torch.tensor(smooth.shape, device=grid.device, dtype=grid.dtype) - 1)
        coarse = sample_linear(smooth, grid).reshape(sizes)
        fa = torch.tensor(f, device=volume.device, dtype=volume.dtype)
        levels[r] = previous = Level(coarse, spacing * fa, origin + spacing * (fa - 1) / 2)
        previous_factors = f
    return levels

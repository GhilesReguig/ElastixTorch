"""Resampling of the moving image on the grid of the fixed image."""
from typing import Sequence

import torch

from .interpolation import cubic_prefilter, inside, sample


def grid_points(shape: Sequence[int], spacing: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """Physical position of every voxel of a grid, ``(prod(shape), 3)``, in C order."""
    axes = [torch.arange(n, device=spacing.device, dtype=spacing.dtype) for n in shape]
    index = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    return origin + index * spacing


@torch.no_grad()
def resample(moving: torch.Tensor, matrix: torch.Tensor, moving_spacing: torch.Tensor, moving_origin: torch.Tensor,
             shape: Sequence[int], spacing: torch.Tensor, origin: torch.Tensor, order: int = 3,
             default_value: float = 0.0, chunk: int = 200_000) -> torch.Tensor:
    """Image of ``moving`` on the grid ``(shape, spacing, origin)``: the value at a point ``p`` of the grid is
    the moving image at ``matrix @ p`` (the 4x4 matrix maps the fixed to the moving physical space).
    Points that fall outside the moving image get ``default_value``."""
    coefficients = cubic_prefilter(moving) if order == 3 else moving
    points = grid_points(shape, spacing, origin)
    out = torch.empty(points.shape[0], dtype=moving.dtype, device=moving.device)
    linear, offset = matrix[:3, :3], matrix[:3, 3]
    for start in range(0, points.shape[0], chunk):
        coordinates = ((points[start:start + chunk] @ linear.T + offset) - moving_origin) / moving_spacing
        values = sample(coefficients, coordinates, order)
        out[start:start + chunk] = torch.where(inside(coordinates, moving.shape), values,
                                               torch.full_like(values, default_value))
    return out.reshape(tuple(shape))

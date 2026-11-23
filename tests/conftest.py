import math

import pytest
import torch
import torch.nn.functional as F


def make_phantom(shape=(48, 48, 48), spacing=(1.0, 1.0, 1.0), seed=0):
    """Smooth asymmetric volume: a few Gaussian blobs of different sizes plus an ellipsoid."""
    generator = torch.Generator().manual_seed(seed)
    axes = [(torch.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, spacing)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
    image = torch.zeros(shape)
    extent = torch.tensor([n * s for n, s in zip(shape, spacing)])
    for _ in range(14):
        centre = (torch.rand(3, generator=generator) - 0.5) * extent * 0.5
        width = (0.04 + 0.08 * torch.rand(3, generator=generator)) * extent
        image += (0.5 + torch.rand(1, generator=generator)) * torch.exp(
            -0.5 * (((grid - centre[:, None, None, None]) / width[:, None, None, None]) ** 2).sum(0))
    ellipsoid = ((grid / (0.32 * extent)[:, None, None, None]) ** 2).sum(0) < 1
    return image + 0.8 * ellipsoid.float()


def rotation(angles_deg):
    """Rz Rx Ry rotation (same convention as the library), written independently."""
    rx, ry, rz = (math.radians(a) for a in angles_deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def warp(fixed, matrix, spacing):
    """Moving image such that ``moving(matrix @ p) = fixed(p)``, built with ``grid_sample`` (independent of the
    library): ``moving(q) = fixed(matrix^-1 q)``."""
    shape = fixed.shape
    spacing = torch.tensor(spacing)
    axes = [(torch.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, spacing)]     # physical, centred origin
    q = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    inverse = torch.linalg.inv(matrix)
    p = q @ inverse[:3, :3].T + inverse[:3, 3]
    half = (torch.tensor(shape) - 1) / 2 * spacing
    normalized = (p / half).flip(-1)                    # grid_sample wants (x, y, z) = (last axis first), in [-1, 1]
    out = F.grid_sample(fixed[None, None], normalized[None], mode="bilinear", padding_mode="zeros",
                        align_corners=True)
    return out[0, 0]


@pytest.fixture(scope="session")
def phantom():
    return make_phantom()

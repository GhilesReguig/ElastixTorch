"""Parametric transforms in physical coordinates, mapping the fixed image to the moving image.

A transform maps a physical point ``p`` of the fixed image to ``A p + b = A (p - c) + c + t`` in the moving
image, where ``c`` is the centre of rotation (the geometrical centre of the fixed image) and ``t`` the
translation. The axes are the ones of the tensors: the point coordinates are ``(axis 0, axis 1, axis 2)``.
Rotations are Euler angles in radians, ``R = Rz Rx Ry`` (the convention of the ITK ``Euler3DTransform``).

======================  ===========================  =====================================================
name                    parameters                   
======================  ===========================  =====================================================
``translation``         3                            translation
``euler`` / ``rigid``   6                            3 rotation angles, 3 translations
``similarity``          7                            3 rotation angles, 3 translations, log of the scale
``affine``              12                           A - I (row by row), 3 translations
======================  ===========================  =====================================================
"""
from typing import Dict

import torch


def euler_matrix(angles: torch.Tensor) -> torch.Tensor:
    """3x3 rotation ``Rz Rx Ry`` from the angles (rx, ry, rz) in radians; differentiable."""
    cx, cy, cz = torch.cos(angles)
    sx, sy, sz = torch.sin(angles)
    one, zero = torch.ones_like(cx), torch.zeros_like(cx)
    rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx]).reshape(3, 3)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy]).reshape(3, 3)
    rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one]).reshape(3, 3)
    return rz @ rx @ ry


class Transform:
    """Base class: subclasses define `num_parameters` and `_linear_and_translation`."""

    name = ""
    num_parameters = 0

    def __init__(self, center: torch.Tensor):
        self.center = center

    def identity(self) -> torch.Tensor:
        parameters = torch.zeros(self.num_parameters, dtype=self.center.dtype, device=self.center.device)
        return parameters

    def _linear_and_translation(self, parameters):
        raise NotImplementedError

    def linear_and_offset(self, parameters: torch.Tensor):
        """``(A, b)`` such that the point ``p`` maps to ``A p + b``."""
        linear, translation = self._linear_and_translation(parameters)
        return linear, self.center - linear @ self.center + translation

    def apply(self, parameters: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Transformed points, ``(N, 3)`` in physical units."""
        linear, offset = self.linear_and_offset(parameters)
        return points @ linear.T + offset

    def matrix(self, parameters: torch.Tensor) -> torch.Tensor:
        """4x4 homogeneous matrix of the transform (fixed physical point -> moving physical point)."""
        linear, offset = self.linear_and_offset(parameters)
        out = torch.eye(4, dtype=linear.dtype, device=linear.device)
        out[:3, :3], out[:3, 3] = linear, offset
        return out

    def parameter_scales(self, points: torch.Tensor, step: float = 1e-3) -> torch.Tensor:
        """Physical shift (mm) of the points per unit of each parameter (RMS over the points). The registration
        works on ``parameter * scale``, so that all the parameters move the image by comparable amounts
        (a rotation of 1 rad moves the points by about the radius of the image, a translation of 1 by 1 mm)."""
        identity = self.identity()
        reference = self.apply(identity, points)
        scales = []
        for i in range(self.num_parameters):
            moved = identity.clone()
            moved[i] += step
            shift = (self.apply(moved, points) - reference) / step
            scales.append(shift.pow(2).sum(dim=1).mean().sqrt())
        return torch.stack(scales).clamp_min(1e-6)


class Translation(Transform):
    name, num_parameters = "translation", 3

    def _linear_and_translation(self, parameters):
        return torch.eye(3, dtype=parameters.dtype, device=parameters.device), parameters


class Euler(Transform):
    name, num_parameters = "euler", 6

    def _linear_and_translation(self, parameters):
        return euler_matrix(parameters[:3]), parameters[3:6]


class Similarity(Transform):
    name, num_parameters = "similarity", 7

    def _linear_and_translation(self, parameters):
        return torch.exp(parameters[6]) * euler_matrix(parameters[:3]), parameters[3:6]


class Affine(Transform):
    name, num_parameters = "affine", 12

    def _linear_and_translation(self, parameters):
        eye = torch.eye(3, dtype=parameters.dtype, device=parameters.device)
        return eye + parameters[:9].reshape(3, 3), parameters[9:12]


TRANSFORMS: Dict[str, type] = {"translation": Translation, "euler": Euler, "rigid": Euler,
                               "similarity": Similarity, "affine": Affine}

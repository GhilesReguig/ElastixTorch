"""Elastix-style intensity-based registration in native PyTorch.

The loop is the one of Elastix: a multi-resolution pyramid (smoothing + downsampling), and at each resolution
a stochastic optimisation (adaptive stochastic gradient descent) where every iteration evaluates the metric
on a new random subset of continuous positions of the fixed image, interpolated linearly, with the moving
image resampled at the transformed positions. The final image is resampled with cubic B-splines.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch

from .interpolation import inside, sample_linear
from .metrics import METRICS
from .optimizer import AdaptiveStochasticGradientDescent, estimate_gain
from .pyramid import build_pyramid, schedule
from .resample import resample
from .transforms import TRANSFORMS

SIGMOID_SCALE = 0.1


def _vector(value, default: Optional[torch.Tensor], device, dtype) -> torch.Tensor:
    if value is None:
        return default.clone()
    value = torch.as_tensor(value, dtype=dtype, device=device)
    return value.expand(3).clone() if value.ndim == 0 else value.clone()


def _per_level(value, count: int, name: str) -> List:
    values = list(value) if isinstance(value, (list, tuple)) else [value] * count
    if len(values) != count:
        raise ValueError(f"{name} must be a number or have one value per resolution ({count})")
    return values


@dataclass
class RegistrationResult:
    """Outcome of a registration.

    Attributes:
        parameters: final transform parameters (see `elastixtorch.transforms`).
        matrix: 4x4 matrix mapping a physical point of the fixed image to the moving image.
        losses: metric value of each iteration, one list per resolution (noisy: new samples every iteration).
    """
    transform: str
    parameters: torch.Tensor
    matrix: torch.Tensor
    losses: List[List[float]] = field(default_factory=list)
    fixed_shape: tuple = ()
    fixed_spacing: Optional[torch.Tensor] = None
    fixed_origin: Optional[torch.Tensor] = None
    moving_spacing: Optional[torch.Tensor] = None
    moving_origin: Optional[torch.Tensor] = None
    final_order: int = 3

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        """Map ``(N, 3)`` physical points of the fixed image to the moving image."""
        points = points.to(self.matrix)
        return points @ self.matrix[:3, :3].T + self.matrix[:3, 3]

    def resample(self, moving: torch.Tensor, order: Optional[int] = None, default_value: float = 0.0) -> torch.Tensor:
        """The moving image on the grid of the fixed image, with B-splines of order ``order`` (default: the
        ``final_order`` of the registration, cubic unless changed)."""
        order = self.final_order if order is None else order
        return resample(moving.to(self.matrix), self.matrix, self.moving_spacing, self.moving_origin,
                        self.fixed_shape, self.fixed_spacing, self.fixed_origin, order, default_value)


class Registration:
    """Registration of a moving volume onto a fixed volume (both 3D tensors, intensities as floats).

    Args:
        transform: ``"translation"``, ``"euler"`` (alias ``"rigid"``), ``"similarity"`` or ``"affine"``.
        metric: ``"mattes"`` (Mattes mutual information, default), ``"mse"`` or ``"ncc"``.
        num_resolutions: number of pyramid levels (Elastix default: 4).
        max_iterations: iterations per resolution (int or one per resolution; Elastix: 250).
        num_samples: random positions per iteration (int or one per resolution; Elastix: 2048).
        num_bins: histogram bins of the mutual information (Elastix: 32).
        final_order: B-spline order used by `RegistrationResult.resample` (1 or 3; Elastix: 3).
        max_step_length: size in mm of the first step; also the largest step (shorter steps than Elastix's
            default, the mean voxel spacing, because the gradients are noisy); default: a quarter of the mean
            voxel spacing of the level.
        A, alpha: ASGD gain parameters (Elastix defaults 20 and 1).
        initial_parameters: starting parameters (default: identity).
        seed: seed of the random sampling.
        device: torch device (default: the one of the fixed image).
        verbose: print the progress.

    The moving image is interpolated linearly during the optimisation (Elastix: ``BSplineInterpolationOrder 1``)
    and the gradient of the metric is analytic, as in Elastix: derivative of the metric w.r.t. the moving values
    x gradient of the interpolated image x Jacobian of the transform (autograd is only used on the 3x3 matrix).
    """

    def __init__(self, transform: str = "euler", metric: str = "mattes", num_resolutions: int = 4,
                 max_iterations: Union[int, Sequence[int]] = 250, num_samples: Union[int, Sequence[int]] = 2048,
                 num_bins: int = 32, final_order: int = 3, max_step_length: Optional[float] = None,
                 A: float = 20.0, alpha: float = 1.0, initial_parameters=None, seed: int = 0, device=None,
                 verbose: bool = False):
        if transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {transform!r}, choose among {sorted(TRANSFORMS)}")
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}, choose among {sorted(METRICS)}")
        if final_order not in (1, 3):
            raise ValueError("final_order must be 1 or 3")
        if num_resolutions < 1:
            raise ValueError("num_resolutions must be at least 1")
        self.transform_name, self.metric_name = transform, metric
        self.num_resolutions = num_resolutions
        self.max_iterations = _per_level(max_iterations, num_resolutions, "max_iterations")
        self.num_samples = _per_level(num_samples, num_resolutions, "num_samples")
        self.num_bins, self.final_order = num_bins, final_order
        self.max_step_length, self.A, self.alpha = max_step_length, A, alpha
        self.initial_parameters, self.seed, self.device, self.verbose = initial_parameters, seed, device, verbose

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _evaluate(transform, parameters, metric, fixed, moving, count, generator):
        """Metric on ``count`` random positions of the fixed level, and its gradient w.r.t. the parameters.
        Returns ``(loss, gradient)``, or ``(None, None)`` if too few positions fall inside the moving image."""
        shape = torch.tensor(fixed.image.shape, device=fixed.image.device, dtype=parameters.dtype)
        coordinates = torch.rand(count, 3, generator=generator, device=fixed.image.device,
                                 dtype=parameters.dtype) * (shape - 1)
        fixed_values = sample_linear(fixed.image, coordinates)
        points = fixed.origin + coordinates * fixed.spacing
        with torch.enable_grad():
            theta = parameters.detach().clone().requires_grad_(True)
            linear, offset = transform.linear_and_offset(theta)
            with torch.no_grad():
                mapped = (points @ linear.T + offset - moving.origin) / moving.spacing
                valid = inside(mapped, moving.image.shape)
                if int(valid.sum()) < max(32, count // 20):
                    return None, None
                moving_values, spatial = sample_linear(moving.image, mapped[valid], gradient=True)
                loss, d_values = metric.value_and_gradient(fixed_values[valid], moving_values)
                d_points = d_values[:, None] * spatial / moving.spacing            # dL/dq, physical units
                d_linear, d_offset = d_points.T @ points[valid], d_points.sum(0)
            (gradient,) = torch.autograd.grad((linear * d_linear).sum() + (offset * d_offset).sum(), theta)
        return float(loss), gradient

    def _optimizer(self, parameters, scales, step, evaluate, generator, level):
        """ASGD whose gain and sigmoid are estimated from gradients measured before the optimisation."""
        device, dtype = parameters.device, parameters.dtype
        norms = []                                       # scaled gradients around the current parameters
        for k in range(8):
            theta = parameters
            if k > 0:
                direction = torch.randn(parameters.shape, generator=generator, device=device, dtype=dtype)
                theta = parameters + 2 * step * direction / direction.norm() / scales
            _, gradient = evaluate(theta)
            if gradient is not None:
                norms.append(float((gradient / scales).norm()))
        if not norms:
            raise RuntimeError(f"resolution {level}: the images do not overlap with the initial transform")
        norm = sum(norms) / len(norms)
        # Sigmoid from the squared norm of the mean gradient (signal) and the variance across sample draws
        # (noise), at the current parameters: noise-dominated -> f_min near -1, signal-dominated -> near 0
        f_min, omega = -0.8, max(SIGMOID_SCALE * norm ** 2, 1e-24)          # fallback without enough draws
        draws = [g for g in (evaluate(parameters)[1] for _ in range(6)) if g is not None]
        if len(draws) > 1:
            stacked = torch.stack(draws) / scales
            mean = stacked.mean(0)
            signal = float(mean.pow(2).sum())
            noise = float((stacked - mean).pow(2).sum(1).mean())
            f_min = -0.99 + 0.98 * signal / max(signal + noise, 1e-30)
            omega = max(SIGMOID_SCALE * noise, 1e-24)
        return AdaptiveStochasticGradientDescent(scales, estimate_gain(norm, step, self.A, self.alpha), self.A,
                                                 self.alpha, omega=omega, f_min=f_min, max_step=step)

    def register(self, fixed: torch.Tensor, moving: torch.Tensor, fixed_spacing=1.0, moving_spacing=None,
                 fixed_origin=0.0, moving_origin=None) -> RegistrationResult:
        """Register ``moving`` onto ``fixed``. Spacings and origins are in mm (a number or 3 values, in the
        order of the tensor axes); the moving ones default to the fixed ones."""
        if fixed.ndim != 3 or moving.ndim != 3:
            raise ValueError("fixed and moving must be 3D tensors")
        device = torch.device(self.device) if self.device is not None else fixed.device
        dtype = torch.float32
        fixed, moving = fixed.to(device, dtype), moving.to(device, dtype)
        f_spacing = _vector(fixed_spacing, None, device, dtype) if fixed_spacing is not None else None
        f_origin = _vector(fixed_origin, None, device, dtype)
        m_spacing = _vector(moving_spacing, f_spacing, device, dtype)
        m_origin = _vector(moving_origin, f_origin, device, dtype)

        center = f_origin + f_spacing * (torch.tensor(fixed.shape, device=device, dtype=dtype) - 1) / 2
        transform = TRANSFORMS[self.transform_name](center)
        parameters = transform.identity() if self.initial_parameters is None \
            else torch.as_tensor(self.initial_parameters, dtype=dtype, device=device).clone()
        if parameters.shape != (transform.num_parameters,):
            raise ValueError(f"{self.transform_name} has {transform.num_parameters} parameters")

        generator = torch.Generator(device=device).manual_seed(self.seed)
        fixed_levels = build_pyramid(fixed, f_spacing, f_origin, schedule(self.num_resolutions, fixed.shape))
        moving_levels = build_pyramid(moving, m_spacing, m_origin, schedule(self.num_resolutions, moving.shape))
        losses = []
        for level in range(self.num_resolutions):
            fixed_level, moving_level = fixed_levels[level], moving_levels[level]
            if self.metric_name in ("mattes", "mi"):
                metric = METRICS[self.metric_name](fixed_level.image, moving_level.image, num_bins=self.num_bins)
            else:
                metric = METRICS[self.metric_name](fixed_level.image, moving_level.image)
            count = int(self.num_samples[level])
            extent = (torch.tensor(fixed_level.image.shape, device=device, dtype=dtype) - 1) * fixed_level.spacing
            corners = torch.cartesian_prod(*[torch.stack([torch.zeros_like(e), e]) for e in extent])
            scales = transform.parameter_scales(fixed_level.origin + corners)
            step = float(self.max_step_length if self.max_step_length is not None else 0.25 * fixed_level.spacing.mean())

            def evaluate(theta):
                return self._evaluate(transform, theta, metric, fixed_level, moving_level, count, generator)

            optimizer = self._optimizer(parameters, scales, step, evaluate, generator, level)
            history = []
            for iteration in range(int(self.max_iterations[level])):
                loss, gradient = evaluate(parameters)
                if gradient is None:                     # too little overlap: keep the parameters, resample
                    continue
                parameters = optimizer.step(parameters, gradient)
                history.append(loss)
            losses.append(history)
            if self.verbose:
                tail = history[-20:]
                print(f"resolution {level + 1}/{self.num_resolutions}  shape {tuple(fixed_level.image.shape)}  "
                      f"loss {sum(tail) / max(len(tail), 1):.5f}  t {optimizer.time:.1f}")

        return RegistrationResult(self.transform_name, parameters.detach(), transform.matrix(parameters).detach(),
                                  losses, tuple(fixed.shape), f_spacing, f_origin, m_spacing, m_origin,
                                  self.final_order)


def register(fixed: torch.Tensor, moving: torch.Tensor, fixed_spacing=1.0, moving_spacing=None, fixed_origin=0.0,
             moving_origin=None, **options) -> RegistrationResult:
    """Shortcut for ``Registration(**options).register(fixed, moving, ...)``."""
    return Registration(**options).register(fixed, moving, fixed_spacing, moving_spacing, fixed_origin,
                                            moving_origin)

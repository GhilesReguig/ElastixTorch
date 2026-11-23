"""Adaptive stochastic gradient descent (ASGD, Klein et al. 2009), the default optimiser of Elastix."""
import math

import torch


class AdaptiveStochasticGradientDescent:
    """``theta <- theta - gamma(t) * g``, with the gain ``gamma(t) = a / (t + A) ** alpha``. The time ``t`` only
    advances when successive gradients are not aligned: ``t <- max(0, t + f(-g_k . g_{k-1}))`` with the
    sigmoid ``f`` going from ``f_min < 0`` (consistent gradients: the step stays large, t decreases) to
    ``f_max > 0`` (gradients that oppose each other: noise around the optimum, t increases, the step
    shrinks). The steps are taken on the scaled parameters ``theta * scales``.

    Args:
        scales: physical shift per unit of each parameter (see `Transform.parameter_scales`).
        a, A, alpha: gain parameters (``a`` is set by `estimate_gain`).
        omega: width of the sigmoid, in units of squared (scaled) gradient.
        max_step: largest step (norm in the scaled parameters, i.e. in mm); larger steps are shortened.
    """

    def __init__(self, scales: torch.Tensor, a: float, A: float = 20.0, alpha: float = 1.0, omega: float = 1.0,
                 f_max: float = 1.0, f_min: float = -0.8, max_step: float = float("inf")):
        self.scales, self.a, self.A, self.alpha, self.omega = scales, a, A, alpha, omega
        self.f_max, self.f_min, self.max_step = f_max, f_min, max_step
        self.time, self._previous = 0.0, None

    @property
    def gain(self) -> float:
        return self.a / (self.time + self.A) ** self.alpha

    def _sigmoid(self, x: float) -> float:
        e = math.exp(max(-50.0, min(50.0, -x / self.omega)))
        return self.f_min + (self.f_max - self.f_min) / (1 - (self.f_max / self.f_min) * e)

    def step(self, parameters: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        scaled = gradient / self.scales
        move = self.gain * scaled                      # step in the scaled parameters (mm)
        length = float(move.norm())
        if length > self.max_step:                     # a noisy gradient must not throw the image away
            move = move * (self.max_step / length)
        updated = parameters - move / self.scales
        if self._previous is not None:
            self.time = max(0.0, self.time + self._sigmoid(-float((scaled * self._previous).sum())))
        self._previous = scaled
        return updated


def estimate_gain(gradient_norm: float, max_step_length: float, A: float = 20.0, alpha: float = 1.0) -> float:
    """``a`` such that the first step moves the image by about ``max_step_length`` (mm) for a gradient of the
    typical norm (in the scaled parameters): ``gamma(0) * |g| = max_step_length``."""
    return max_step_length * A ** alpha / max(gradient_norm, 1e-12)

"""Similarity metrics evaluated on sampled points (to be minimised). Each takes the fixed and the moving
intensities at the sampled points and returns a scalar loss, differentiable w.r.t. the moving values."""
import torch

from .interpolation import bspline_derivative_weights, bspline_weights


class MattesMutualInformation:
    """Mattes mutual information (negated), the default metric of Elastix: a joint histogram where the moving
    intensity is spread on 4 bins by a cubic B-spline Parzen window and the fixed one falls in a single bin
    (zero-order window, the Elastix default).

    The bin mapping follows ``itkParzenWindowHistogramImageToImageMetric``: the min/max of the images of the
    level, widened by 1 % on each side; fixed values on bins 0 to ``num_bins - 1`` (rounded), moving values on
    positions 1 to ``num_bins - 2``. The moving values are assumed to lie in the range of the moving image, as
    with linear interpolation (Elastix's exponential limiter for values outside it is therefore not needed).
    """

    def __init__(self, fixed: torch.Tensor, moving: torch.Tensor, num_bins: int = 32):
        if num_bins < 8:
            raise ValueError("num_bins must be at least 8")
        self.num_bins = num_bins
        self.fixed_range = (float(fixed.min()), float(fixed.max()))
        self.moving_range = (float(moving.min()), float(moving.max()))
        # position = (value - low) * scale + offset
        self._fixed_map = self._bin_map(self._widen(self.fixed_range), padding=0)
        self._moving_map = self._bin_map(self._widen(self.moving_range), padding=1)

    @staticmethod
    def _widen(value_range, ratio=0.01):
        low, high = value_range
        return low - ratio * (high - low), high + ratio * (high - low)

    def _bin_map(self, limits, padding):
        low, high = limits
        width = self.num_bins - 2 * padding - 1
        small = 0.001 * (high - low) / (self.num_bins - 2 * padding - 1)
        bin_size = min(max((high - low + 2 * small) / width, 1e-10), 1e10)
        return low - small, 1.0 / bin_size, float(padding)

    def _fixed_bin(self, values):
        low, scale, offset = self._fixed_map
        position = (values - low) * scale + offset
        return torch.floor(position + 0.5).long().clamp(0, self.num_bins - 1)

    def _moving_position(self, values):
        """Continuous bin position of the moving values and its derivative w.r.t. the values."""
        low, scale, offset = self._moving_map
        return ((values - low) * scale + offset).clamp(1, self.num_bins - 2 - 1e-6), scale

    def __call__(self, fixed_values: torch.Tensor, moving_values: torch.Tensor) -> torch.Tensor:
        bins = self.num_bins
        fixed_bin = self._fixed_bin(fixed_values)
        position, _ = self._moving_position(moving_values)
        base = torch.floor(position.detach())
        weights = bspline_weights(position - base, 3)
        moving_bins = (base.long()[:, None] + torch.arange(-1, 3, device=position.device)).clamp(0, bins - 1)
        joint = torch.zeros(bins, bins, dtype=weights.dtype, device=weights.device)
        joint = joint.index_put((fixed_bin[:, None].expand_as(moving_bins), moving_bins), weights, accumulate=True)
        joint = joint / joint.sum().clamp_min(1e-12)
        marginal_fixed, marginal_moving = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
        ratio = joint / (marginal_fixed * marginal_moving).clamp_min(1e-12)
        return -(joint * torch.log(ratio.clamp_min(1e-12))).sum()

    def value_and_gradient(self, fixed_values, moving_values):
        """Loss and its derivative w.r.t. each moving value, without autograd. The histogram sums to the number
        of samples (the B-spline weights sum to 1), so ``dMI/dp`` reduces to ``log(p / (p_f p_m))``."""
        bins = self.num_bins
        position, d_position = self._moving_position(moving_values)
        fixed_bin = self._fixed_bin(fixed_values)
        base = torch.floor(position)
        t = position - base
        # component-major layout ((4, N) tensors): faster than (N, 4) for a few thousand samples
        weights = bspline_weights(t, 3).T.contiguous()
        derivative = bspline_derivative_weights(t).T.contiguous()
        moving_bins = (base.long()[None, :] + torch.arange(-1, 3, device=position.device)[:, None]).clamp(0, bins - 1)
        flat = (fixed_bin[None, :] * bins + moving_bins).reshape(-1)
        joint = torch.zeros(bins * bins, dtype=weights.dtype, device=weights.device)
        joint = joint.index_add(0, flat, weights.reshape(-1)).reshape(bins, bins)
        total = joint.sum().clamp_min(1e-12)
        p = joint / total
        log_ratio = torch.log((p / (p.sum(1, keepdim=True) * p.sum(0, keepdim=True)).clamp_min(1e-12)).clamp_min(1e-12))
        mi = (p * log_ratio).sum()
        d_mi = (torch.take(log_ratio, flat).reshape(4, -1) * derivative).sum(0) * d_position / total
        return -mi, -d_mi


class MeanSquares:
    """Mean squared difference of the intensities (for images of the same modality and intensity scale)."""

    def __init__(self, fixed: torch.Tensor, moving: torch.Tensor, **_):
        self.norm = max(float(fixed.var()), 1e-12)

    def __call__(self, fixed_values, moving_values):
        return ((fixed_values - moving_values) ** 2).mean() / self.norm

    def value_and_gradient(self, fixed_values, moving_values):
        difference = moving_values - fixed_values
        return (difference ** 2).mean() / self.norm, 2 * difference / (len(difference) * self.norm)


class NormalizedCorrelation:
    """Negative normalised cross-correlation (invariant to a linear change of intensities)."""

    def __init__(self, fixed: torch.Tensor, moving: torch.Tensor, **_):
        pass

    def __call__(self, fixed_values, moving_values):
        f = fixed_values - fixed_values.mean()
        m = moving_values - moving_values.mean()
        return -(f * m).sum() / torch.sqrt((f * f).sum() * (m * m).sum() + 1e-12)

    def value_and_gradient(self, fixed_values, moving_values):
        f = fixed_values - fixed_values.mean()
        m = moving_values - moving_values.mean()
        s_fm, s_ff, s_mm = (f * m).sum(), (f * f).sum(), (m * m).sum()
        d = torch.sqrt(s_ff * s_mm + 1e-12)
        return -s_fm / d, -f / d + s_fm * s_ff * m / d ** 3


METRICS = {"mattes": MattesMutualInformation, "mi": MattesMutualInformation, "mse": MeanSquares,
           "ncc": NormalizedCorrelation}

import math

import pytest
import torch

from elastixtorch import (AdaptiveStochasticGradientDescent, MattesMutualInformation, TRANSFORMS, cubic_prefilter,
                              euler_matrix, resample, sample)
from elastixtorch.interpolation import bspline_weights, inside
from elastixtorch.pyramid import build_pyramid, gaussian_smooth, schedule

from conftest import make_phantom, rotation


@pytest.mark.parametrize("order", [1, 3])
def test_weights_sum_to_one_and_are_non_negative(order):
    t = torch.linspace(0, 0.999, 50)
    weights = bspline_weights(t, order)
    assert torch.allclose(weights.sum(1), torch.ones(50), atol=1e-6)
    assert (weights >= 0).all()


def test_cubic_interpolation_passes_through_the_voxels(phantom):
    coefficients = cubic_prefilter(phantom)
    index = torch.stack(torch.meshgrid(*[torch.arange(0, n, 5) for n in phantom.shape], indexing="ij"),
                        dim=-1).reshape(-1, 3)
    values = sample(coefficients, index.float(), order=3)
    torch.testing.assert_close(values, phantom[index[:, 0], index[:, 1], index[:, 2]], atol=1e-4, rtol=1e-4)


def test_linear_interpolation_is_exact_on_a_linear_image():
    x = torch.arange(10.0)
    volume = 2 * x[:, None, None] + 3 * x[None, :, None] - x[None, None, :]
    points = torch.rand(40, 3) * 8.5 + 0.2
    expected = 2 * points[:, 0] + 3 * points[:, 1] - points[:, 2]
    torch.testing.assert_close(sample(volume, points, 1), expected, atol=1e-4, rtol=1e-5)


def test_cubic_interpolation_reproduces_a_linear_image_inside_the_volume():
    x = torch.arange(16.0)
    volume = 2 * x[:, None, None] + 3 * x[None, :, None] - x[None, None, :]
    points = torch.rand(40, 3) * 8 + 4
    expected = 2 * points[:, 0] + 3 * points[:, 1] - points[:, 2]
    torch.testing.assert_close(sample(cubic_prefilter(volume), points, 3), expected, atol=2e-2, rtol=1e-3)


def test_coordinates_receive_a_gradient():
    volume = torch.rand(8, 8, 8)
    points = (torch.rand(5, 3) * 5 + 1).requires_grad_(True)
    sample(volume, points, 1).sum().backward()
    assert points.grad is not None and points.grad.abs().sum() > 0


def test_inside_mask():
    points = torch.tensor([[0.0, 0, 0], [3, 3, 3], [3.1, 0, 0], [-0.1, 0, 0]])
    assert inside(points, (4, 4, 4)).tolist() == [True, True, False, False]


def test_euler_matrix_is_a_rotation_in_the_documented_order():
    angles = torch.tensor([0.3, -0.5, 0.8])
    matrix = euler_matrix(angles)
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(matrix, rotation([math.degrees(a) for a in angles.tolist()]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("name", ["translation", "euler", "similarity", "affine"])
def test_identity_parameters_give_the_identity_matrix(name):
    transform = TRANSFORMS[name](torch.tensor([3.0, -2.0, 5.0]))
    torch.testing.assert_close(transform.matrix(transform.identity()), torch.eye(4), atol=1e-6, rtol=0)


def test_rotation_is_about_the_centre():
    centre = torch.tensor([10.0, 20.0, 30.0])
    transform = TRANSFORMS["euler"](centre)
    parameters = torch.tensor([0.2, 0.1, -0.3, 0, 0, 0])
    torch.testing.assert_close(transform.apply(parameters, centre[None])[0], centre, atol=1e-5, rtol=0)
    parameters = torch.tensor([0.0, 0, 0, 1, 2, 3])
    torch.testing.assert_close(transform.apply(parameters, centre[None])[0], centre + torch.tensor([1.0, 2, 3]))


def test_parameter_scales_are_in_mm_per_unit():
    transform = TRANSFORMS["euler"](torch.zeros(3))
    points = torch.cartesian_prod(*[torch.tensor([-50.0, 50.0])] * 3)
    scales = transform.parameter_scales(points)
    assert torch.allclose(scales[3:], torch.ones(3), atol=1e-3)              # translations: 1 mm per unit
    assert (scales[:3] > 30).all()                                            # rotations: about the radius


def test_pyramid_geometry_and_smoothing():
    volume = make_phantom((40, 36, 44))
    assert schedule(3, volume.shape) == ((4, 4, 4), (2, 2, 2), (1, 1, 1))
    assert schedule(4, (20, 40, 100))[0] == (2, 5, 8)                         # limited to keep 8 voxels per axis
    level = build_pyramid(volume, torch.ones(3), torch.zeros(3), ((2, 2, 2), (1, 1, 1)))[0]
    assert level.image.shape == (20, 18, 22)
    assert torch.allclose(level.spacing, torch.full((3,), 2.0)) and torch.allclose(level.origin, torch.full((3,), 0.5))
    assert float(gaussian_smooth(volume, [1.5] * 3).var()) < float(volume.var())
    assert torch.allclose(gaussian_smooth(torch.ones(9, 9, 9), [1.0] * 3), torch.ones(9, 9, 9), atol=1e-5)


def test_mattes_is_higher_for_aligned_images_and_invariant_to_intensity_mapping(phantom):
    metric = MattesMutualInformation(phantom, phantom, 32)
    values = phantom.reshape(-1)[::7]
    aligned = float(metric(values, values))
    shuffled = float(metric(values, values[torch.randperm(len(values), generator=torch.Generator().manual_seed(0))]))
    inverted = float(MattesMutualInformation(phantom, -phantom, 32)(values, -values))
    assert -aligned > -shuffled + 0.5
    assert inverted == pytest.approx(aligned, abs=0.05)


def test_mattes_gradient_pushes_towards_the_matching_intensity(phantom):
    metric = MattesMutualInformation(phantom, phantom, 32)
    fixed = phantom.reshape(-1)[::11]
    moving = (fixed + 0.05 * torch.randn_like(fixed)).requires_grad_(True)
    metric(fixed, moving).backward()
    assert torch.isfinite(moving.grad).all() and moving.grad.abs().sum() > 0


def test_asgd_time_grows_for_alternating_gradients_and_not_for_steady_ones():
    scales = torch.ones(2)
    steady = AdaptiveStochasticGradientDescent(scales, a=20.0, omega=0.1)
    noisy = AdaptiveStochasticGradientDescent(scales, a=20.0, omega=0.1)
    theta = torch.zeros(2)
    for k in range(30):
        steady.step(theta, torch.tensor([1.0, 0.0]))
        noisy.step(theta, torch.tensor([1.0, 0.0]) * (-1) ** k)
    assert steady.time == 0.0 and noisy.time > 5
    assert noisy.gain < steady.gain


def test_asgd_minimises_a_quadratic():
    scales = torch.tensor([1.0, 4.0])
    theta = torch.tensor([3.0, -2.0])
    optimizer = AdaptiveStochasticGradientDescent(scales, a=20.0 * 0.5, omega=0.05)
    for _ in range(300):
        theta = optimizer.step(theta, theta + 0.01 * torch.randn(2))
    assert theta.abs().max() < 0.2


def test_resample_with_identity_returns_the_image_and_fills_outside(phantom):
    out = resample(phantom, torch.eye(4), torch.ones(3), torch.zeros(3), phantom.shape, torch.ones(3), torch.zeros(3))
    torch.testing.assert_close(out, phantom, atol=1e-4, rtol=1e-4)
    shifted = torch.eye(4)
    shifted[0, 3] = 100.0
    out = resample(phantom, shifted, torch.ones(3), torch.zeros(3), phantom.shape, torch.ones(3), torch.zeros(3),
                   default_value=-1.0)
    assert (out == -1.0).all()


def test_trilinear_interpolation_matches_the_generic_one_and_its_gradient(phantom):
    from elastixtorch.interpolation import sample_linear
    points = torch.rand(500, 3) * (torch.tensor(phantom.shape) - 1.0)
    points[:3] = torch.tensor(phantom.shape) - 1.0                          # exactly on the last voxel
    torch.testing.assert_close(sample_linear(phantom, points), sample(phantom, points, 1), atol=1e-6, rtol=1e-5)
    where = points.clone().requires_grad_(True)
    (expected,) = torch.autograd.grad(sample(phantom, where, 1).sum(), where)
    _, spatial = sample_linear(phantom, points, gradient=True)
    inner = (points > 0.01).all(1) & (points < torch.tensor(phantom.shape) - 1.01).all(1)   # away from kinks at borders
    torch.testing.assert_close(spatial[inner], expected[inner], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", ["mattes", "mse", "ncc"])
def test_analytic_metric_derivative_matches_autograd(phantom, name):
    from elastixtorch import METRICS
    metric = METRICS[name](phantom, phantom * 1.1 + 0.2, num_bins=32) if name == "mattes" else METRICS[name](phantom, phantom)
    generator = torch.Generator().manual_seed(0)
    fixed = phantom.reshape(-1)[torch.randperm(phantom.numel(), generator=generator)[:3000]]
    moving = fixed * 1.1 + 0.2 + 0.05 * torch.randn(3000, generator=generator)
    # inside the range of the moving image, as linear interpolation guarantees
    moving = moving.clamp(float(phantom.min()) * 1.1 + 0.2, float(phantom.max()) * 1.1 + 0.2).requires_grad_(True)
    loss = metric(fixed, moving)
    (expected,) = torch.autograd.grad(loss, moving)
    value, derivative = metric.value_and_gradient(fixed, moving.detach())
    torch.testing.assert_close(value, loss.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(derivative, expected, atol=1e-7, rtol=1e-3)


def test_pyramid_in_cascade_matches_levels_computed_from_the_full_resolution():
    volume = make_phantom((40, 36, 44))
    factors = schedule(3, volume.shape)
    levels = build_pyramid(volume, torch.ones(3), torch.zeros(3), factors)
    for level, f in zip(levels, factors):
        # direct computation: smoothing with sigma = 0.5 * f, then sampling at the centres of the f^3 blocks
        smooth = gaussian_smooth(volume, [0.5 * k if k > 1 else 0.0 for k in f])
        axes = [torch.arange(n // k) * k + (k - 1) / 2 for n, k in zip(volume.shape, f)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        direct = sample(smooth, grid.reshape(-1, 3), 1).reshape(grid.shape[:3])
        assert level.image.shape == direct.shape
        torch.testing.assert_close(level.spacing, torch.tensor(f, dtype=torch.float32))
        torch.testing.assert_close(level.origin, (torch.tensor(f, dtype=torch.float32) - 1) / 2)
        assert float((level.image - direct).abs().max() / direct.abs().max()) < 0.05


def test_mattes_analytic_derivative_matches_autograd_over_the_whole_range(phantom):
    torch.manual_seed(0)
    fixed_values = phantom.flatten()[torch.randperm(phantom.numel())[:3000]]
    # moving values spread over the whole image range, down to its bounds
    low, high = float(phantom.min()), float(phantom.max())
    moving_values = (fixed_values * 0.8 + 0.3 * torch.randn(3000) * (high - low)).clamp(low, high)
    metric = MattesMutualInformation(phantom, phantom)
    values = moving_values.clone().requires_grad_(True)
    (expected,) = torch.autograd.grad(metric(fixed_values, values), values)
    loss, gradient = metric.value_and_gradient(fixed_values, moving_values)
    torch.testing.assert_close(loss, metric(fixed_values, moving_values))
    torch.testing.assert_close(gradient, expected, atol=1e-6, rtol=1e-4)


def test_mattes_bins_follow_elastix(phantom):
    metric = MattesMutualInformation(phantom, phantom)
    low, high = metric.fixed_range
    assert int(metric._fixed_bin(torch.tensor([low]))) == 0 and int(metric._fixed_bin(torch.tensor([high]))) == 31
    position, _ = metric._moving_position(torch.tensor([low, high]))
    assert 1.0 < float(position[0]) < 1.5 and 29.5 < float(position[1]) < 30.0     # padding of 1 bin

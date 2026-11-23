"""Device handling: the CPU is the reference; the CUDA cases only run on a machine with a GPU.

    pytest tests/test_devices.py -v -rs     # on a GPU machine, also runs the [cuda] cases

The random draws of a CUDA generator differ from the CPU ones, so full registrations are compared by their
accuracy (and by the distance between the CPU and CUDA results), not sample by sample.
"""
import pytest
import torch

from elastixtorch import METRICS, cubic_prefilter, register, resample, sample
from elastixtorch.interpolation import sample_linear
from elastixtorch.pyramid import build_pyramid, schedule

from conftest import make_phantom, warp
from test_registration import corner_error, true_matrix

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")),
]
SHAPE = (48, 48, 48)
TRUTH = true_matrix((4.0, -3.0, 5.0), (2.5, -1.5, 2.0))


@pytest.fixture(scope="module")
def pair():
    fixed = make_phantom(SHAPE)
    origin = -(torch.tensor(SHAPE) - 1) / 2
    return fixed, warp(fixed, TRUTH, (1.0,) * 3), origin


@pytest.mark.parametrize("device", DEVICES)
def test_interpolation_matches_the_cpu(device, pair):
    fixed = pair[0]
    points = torch.rand(3000, 3, generator=torch.Generator().manual_seed(0)) * (torch.tensor(SHAPE) - 1.0)
    on = lambda x: x.to(device)
    torch.testing.assert_close(sample_linear(on(fixed), on(points)).cpu(), sample_linear(fixed, points),
                               atol=1e-5, rtol=1e-5)
    _, spatial = sample_linear(on(fixed), on(points), gradient=True)
    _, ref_spatial = sample_linear(fixed, points, gradient=True)
    torch.testing.assert_close(spatial.cpu(), ref_spatial, atol=1e-5, rtol=1e-4)
    coefficients = cubic_prefilter(on(fixed))
    assert coefficients.device.type == device
    torch.testing.assert_close(sample(coefficients, on(points), 3).cpu(), sample(cubic_prefilter(fixed), points, 3),
                               atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["mattes", "mse", "ncc"])
def test_metrics_and_their_derivative_match_the_cpu(device, name, pair):
    fixed, moving, _ = pair
    make = lambda f, m: METRICS[name](f, m, num_bins=32) if name == "mattes" else METRICS[name](f, m)
    f_values, m_values = fixed.reshape(-1)[::17], moving.reshape(-1)[::17]
    loss, derivative = make(fixed.to(device), moving.to(device)).value_and_gradient(f_values.to(device),
                                                                                   m_values.to(device))
    ref_loss, ref_derivative = make(fixed, moving).value_and_gradient(f_values, m_values)
    assert derivative.device.type == device
    torch.testing.assert_close(loss.cpu(), ref_loss, atol=1e-5, rtol=1e-4)        # CUDA index_add: summation order
    torch.testing.assert_close(derivative.cpu(), ref_derivative, atol=1e-7, rtol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_pyramid_matches_the_cpu(device, pair):
    fixed = pair[0]
    factors = schedule(3, SHAPE)
    gpu = build_pyramid(fixed.to(device), torch.ones(3, device=device), torch.zeros(3, device=device), factors)
    cpu = build_pyramid(fixed, torch.ones(3), torch.zeros(3), factors)
    for a, b in zip(gpu, cpu):
        assert a.image.device.type == device
        torch.testing.assert_close(a.image.cpu(), b.image, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transform,metric", [("translation", "mattes"), ("euler", "mattes"), ("euler", "mse"),
                                              ("euler", "ncc"), ("similarity", "mattes"), ("affine", "mattes")])
def test_registration_on_the_device_recovers_the_transform(device, transform, metric, pair):
    fixed, moving, origin = pair
    truth = true_matrix((0, 0, 0), (2.5, -1.5, 2.0)) if transform == "translation" else TRUTH
    if transform == "translation":
        moving = warp(fixed, truth, (1.0,) * 3)
    options = dict(fixed_origin=origin, transform=transform, metric=metric, num_resolutions=3, max_iterations=150)
    result = register(fixed, moving, device=device, **options)
    assert result.matrix.device.type == device and result.parameters.device.type == device
    error = corner_error(result.matrix.cpu(), truth, SHAPE, (1.0,) * 3)
    assert error < (1.0 if transform in ("similarity", "affine") else 0.7)
    reference = register(fixed, moving, device="cpu", **options)
    assert corner_error(result.matrix.cpu(), reference.matrix, SHAPE, (1.0,) * 3) < 0.7


@pytest.mark.parametrize("device", DEVICES)
def test_the_device_of_the_inputs_is_the_default(device, pair):
    fixed, moving, origin = pair
    result = register(fixed.to(device), moving.to(device), fixed_origin=origin, num_resolutions=2,
                      max_iterations=60)
    assert result.matrix.device.type == device                 # the device of the inputs is the default
    assert torch.isfinite(result.parameters).all()
    assert corner_error(result.matrix.cpu(), TRUTH, SHAPE, (1.0,) * 3) < 1.5


@pytest.mark.parametrize("device", DEVICES)
def test_resampling_on_the_device_matches_the_cpu(device, pair):
    fixed, moving, origin = pair
    result = register(fixed, moving, fixed_origin=origin, device=device, num_resolutions=2, max_iterations=60)
    aligned = result.resample(moving)                           # moved to the device of the result
    assert aligned.device.type == device
    expected = resample(moving, result.matrix.cpu(), result.moving_spacing.cpu(), result.moving_origin.cpu(),
                        SHAPE, result.fixed_spacing.cpu(), result.fixed_origin.cpu())
    torch.testing.assert_close(aligned.cpu(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_same_seed_gives_the_same_result_on_the_device(device, pair):
    fixed, moving, origin = pair
    run = lambda seed: register(fixed, moving, fixed_origin=origin, device=device, num_resolutions=2,
                                max_iterations=40, seed=seed).parameters.cpu()
    # identical on the CPU; on CUDA index_add sums in an unspecified order: equal up to rounding
    torch.testing.assert_close(run(3), run(3), rtol=0, atol=0 if device == "cpu" else 1e-3)

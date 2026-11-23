import pytest
import torch

from elastixtorch import Registration, register

from conftest import make_phantom, rotation, warp


def true_matrix(angles_deg, translation, scale=1.0):
    matrix = torch.eye(4)
    matrix[:3, :3] = scale * rotation(angles_deg)
    matrix[:3, 3] = torch.tensor(translation)
    return matrix


def corner_error(estimated, truth, shape, spacing):
    """RMS distance (mm) between the points mapped by both matrices, over a grid in the image."""
    half = (torch.tensor(shape) - 1) / 2 * torch.tensor(spacing)
    points = torch.cartesian_prod(*[torch.tensor([-0.7, 0.0, 0.7]) * h for h in half])
    mapped = lambda m: points @ m[:3, :3].T + m[:3, 3]
    return float((mapped(estimated) - mapped(truth)).pow(2).sum(1).mean().sqrt())


def run(moving_matrix, spacing=(1.0, 1.0, 1.0), shape=(48, 48, 48), **options):
    fixed = make_phantom(shape, spacing)
    moving = warp(fixed, moving_matrix, spacing)
    origin = -(torch.tensor(shape) - 1) / 2 * torch.tensor(spacing)
    options.setdefault("max_iterations", 150)
    result = register(fixed, moving, spacing, fixed_origin=origin, **options)
    return fixed, moving, result


def test_translation_is_recovered():
    truth = true_matrix((0, 0, 0), (3.2, -2.1, 1.7))
    _, _, result = run(truth, transform="translation", num_resolutions=3)
    assert corner_error(result.matrix, truth, (48,) * 3, (1.0,) * 3) < 0.4


def test_rigid_transform_is_recovered_with_mutual_information():
    truth = true_matrix((4.0, -5.0, 6.0), (3.0, -2.0, 1.5))
    _, _, result = run(truth, transform="rigid", num_resolutions=3)
    assert corner_error(result.matrix, truth, (48,) * 3, (1.0,) * 3) < 0.7
    assert result.parameters.shape == (6,) and len(result.losses) == 3


def test_anisotropic_spacing_is_handled_in_millimetres():
    spacing = (2.0, 1.0, 1.0)
    truth = true_matrix((0, 0, 5.0), (4.0, -2.0, 1.0))
    _, _, result = run(truth, spacing=spacing, shape=(32, 48, 48), transform="euler", num_resolutions=3)
    assert corner_error(result.matrix, truth, (32, 48, 48), spacing) < 0.8


@pytest.mark.parametrize("metric", ["mse", "ncc"])
def test_monomodal_metrics_recover_a_rigid_transform(metric):
    truth = true_matrix((3.0, 2.0, -4.0), (2.0, 2.5, -1.0))
    _, _, result = run(truth, transform="euler", metric=metric, num_resolutions=3)
    assert corner_error(result.matrix, truth, (48,) * 3, (1.0,) * 3) < 0.7


def test_mutual_information_handles_an_intensity_change_that_mse_cannot():
    truth = true_matrix((0, 0, 0), (3.0, -2.0, 2.0))
    fixed = make_phantom()
    moving = 5.0 - 2.0 * warp(fixed, truth, (1.0,) * 3)                      # inverted contrast
    origin = -(torch.tensor(fixed.shape) - 1) / 2
    result = register(fixed, moving, fixed_origin=origin, transform="translation", num_resolutions=3,
                      max_iterations=150)
    assert corner_error(result.matrix, truth, fixed.shape, (1.0,) * 3) < 0.6


def test_affine_and_similarity_transforms_are_recovered():
    truth = true_matrix((2.0, -3.0, 3.0), (2.0, -1.5, 1.0), scale=1.05)
    for name in ("similarity", "affine"):
        _, _, result = run(truth, transform=name, num_resolutions=3, max_iterations=200)
        assert corner_error(result.matrix, truth, (48,) * 3, (1.0,) * 3) < 1.0, name
    assert result.parameters.shape == (12,)


def test_resampled_moving_image_matches_the_fixed_image():
    truth = true_matrix((3.0, -4.0, 5.0), (2.0, -2.0, 1.0))
    fixed, moving, result = run(truth, transform="euler", num_resolutions=3)
    aligned = result.resample(moving)
    mask = aligned != 0
    before = float(((fixed - moving) ** 2)[mask].mean())
    after = float(((fixed - aligned) ** 2)[mask].mean())
    assert after < 0.1 * before


def test_identical_images_stay_at_the_identity():
    fixed = make_phantom()
    result = register(fixed, fixed.clone(), transform="euler", num_resolutions=3, max_iterations=100)
    assert corner_error(result.matrix, torch.eye(4), fixed.shape, (1.0,) * 3) < 0.3


def test_results_are_reproducible_with_the_seed_and_depend_on_it():
    fixed = make_phantom()
    moving = warp(fixed, true_matrix((0, 0, 3.0), (2.0, 0, 0)), (1.0,) * 3)
    kwargs = dict(transform="euler", num_resolutions=2, max_iterations=30)
    a = register(fixed, moving, seed=1, **kwargs).parameters
    b = register(fixed, moving, seed=1, **kwargs).parameters
    c = register(fixed, moving, seed=2, **kwargs).parameters
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_initial_parameters_are_used():
    fixed = make_phantom()
    moving = warp(fixed, true_matrix((0, 0, 0), (6.0, 0, 0)), (1.0,) * 3)
    origin = -(torch.tensor(fixed.shape) - 1) / 2
    result = register(fixed, moving, fixed_origin=origin, transform="translation", num_resolutions=1,
                      max_iterations=0, initial_parameters=[6.0, 0, 0])
    assert result.parameters.tolist() == [6.0, 0.0, 0.0]


def test_invalid_arguments():
    with pytest.raises(ValueError, match="transform"):
        Registration(transform="bspline")
    with pytest.raises(ValueError, match="metric"):
        Registration(metric="x")
    with pytest.raises(ValueError, match="one value per resolution"):
        Registration(num_resolutions=3, max_iterations=[1, 2])
    with pytest.raises(ValueError, match="3D"):
        Registration().register(torch.zeros(8, 8), torch.zeros(8, 8))
    with pytest.raises(ValueError, match="parameters"):
        register(make_phantom(), make_phantom(), transform="euler", initial_parameters=[0.0])


def test_non_overlapping_images_raise():
    fixed = make_phantom()
    with pytest.raises(RuntimeError, match="overlap"):
        register(fixed, fixed, transform="translation", num_resolutions=1, initial_parameters=[1000.0, 0, 0])


@pytest.mark.parametrize("transform", ["translation", "euler", "similarity", "affine"])
@pytest.mark.parametrize("metric", ["mattes", "mse", "ncc"])
def test_analytic_gradient_matches_autograd(transform, metric):
    from elastixtorch.interpolation import inside, sample_linear
    from elastixtorch.pyramid import Level
    from elastixtorch import METRICS, TRANSFORMS
    fixed = make_phantom((32, 32, 32))
    moving = warp(fixed, true_matrix((3.0, -2.0, 4.0), (1.5, -1.0, 2.0)), (1.0,) * 3)
    level = lambda image: Level(image, torch.ones(3), torch.zeros(3))
    transform_ = TRANSFORMS[transform](torch.full((3,), 15.5))
    parameters = transform_.identity() + 0.01
    metric_ = METRICS[metric](fixed, moving, num_bins=32) if metric == "mattes" else METRICS[metric](fixed, moving)
    loss_a, grad_a = Registration._evaluate(transform_, parameters, metric_, level(fixed), level(moving), 2000,
                                            torch.Generator().manual_seed(4))
    # reference: backpropagation through the whole evaluation, on the same random positions
    coordinates = torch.rand(2000, 3, generator=torch.Generator().manual_seed(4)) * 31
    theta = parameters.clone().requires_grad_(True)
    mapped = transform_.apply(theta, coordinates)
    valid = inside(mapped, moving.shape)
    loss_b = metric_(sample_linear(fixed, coordinates)[valid], sample_linear(moving, mapped[valid]))
    (grad_b,) = torch.autograd.grad(loss_b, theta)
    assert loss_a == pytest.approx(float(loss_b), rel=1e-5, abs=1e-6)
    torch.testing.assert_close(grad_a, grad_b, rtol=2e-3, atol=1e-6 * float(grad_b.abs().max()))

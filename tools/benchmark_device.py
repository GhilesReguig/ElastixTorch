"""Time one registration on the CPU and on CUDA (synthetic phantom, no data needed), and with itk-elastix
when it is installed (``pip install itk-elastix``; rigid default parameter map, its own default threads).

    python tools/benchmark_device.py                 # 96^3 and 192^3 volumes
    python tools/benchmark_device.py --samples 2048 20000
    python tools/benchmark_device.py --no-elastix

The error is the RMS distance (mm) between the points mapped by the estimated and the true transform, over a
grid inside the volume. The Elastix transform is read back from its deformation field (transformix), which
avoids depending on its parameter conventions.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import make_phantom, warp  # noqa: E402
from test_registration import corner_error, true_matrix  # noqa: E402
from elastixtorch import register  # noqa: E402


def timed(fixed, moving, origin, device, samples):
    options = dict(fixed_origin=origin, device=device, num_samples=samples)
    register(fixed, moving, **{**options, "max_iterations": 3})          # warm-up (CUDA context, kernels)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = register(fixed, moving, **options)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - start, result


def elastix_available():
    try:
        import itk  # noqa: F401
        return hasattr(itk, "elastix_registration_method")
    except ImportError:
        return False


def to_itk(volume, origin):
    """ITK image of a (axis 0, axis 1, axis 2) tensor: ITK's x is the last tensor axis."""
    import itk
    image = itk.image_from_array(volume.numpy().astype("float32"))
    image.SetSpacing([1.0, 1.0, 1.0])
    image.SetOrigin([float(o) for o in origin.flip(0)])
    return image


def elastix_matrix(fixed_image, parameters, shape, origin):
    """4x4 matrix (fixed -> moving, tensor axes) fitted by least squares on the Elastix deformation field."""
    import itk
    with tempfile.TemporaryDirectory() as directory:      # transformix also writes the field to a file
        field = itk.transformix_deformation_field(fixed_image, parameters, output_directory=directory)
        field = itk.array_from_image(field)                                                   # (z, y, x, 3), xyz
    displacement = torch.from_numpy(field.copy()).flip(-1).reshape(-1, 3).double()          # tensor axes
    axes = [torch.arange(n, dtype=torch.float64) for n in shape]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) + origin.double()
    keep = torch.arange(0, len(points), 97)
    homogeneous = torch.cat([points[keep], torch.ones(len(keep), 1, dtype=torch.float64)], 1)
    solution = torch.linalg.lstsq(homogeneous, points[keep] + displacement[keep]).solution     # (4, 3)
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :] = solution.T
    return matrix.float()


def timed_elastix(fixed, moving, origin, samples):
    import itk
    parameter_object = itk.ParameterObject.New()
    parameter_map = parameter_object.GetDefaultParameterMap("rigid")
    parameter_map["NumberOfSpatialSamples"] = [str(samples)]
    parameter_map["WriteResultImage"] = ["false"]
    parameter_object.AddParameterMap(parameter_map)
    fixed_image, moving_image = to_itk(fixed, origin), to_itk(moving, origin)
    start = time.perf_counter()
    _, parameters = itk.elastix_registration_method(fixed_image, moving_image, parameter_object=parameter_object,
                                                     log_to_console=False)
    seconds = time.perf_counter() - start
    return seconds, elastix_matrix(fixed_image, parameters, fixed.shape, origin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[96, 192])
    parser.add_argument("--samples", type=int, nargs="+", default=[2048, 20000])
    parser.add_argument("--no-elastix", action="store_true", help="skip itk-elastix")
    args = parser.parse_args()
    with_elastix = not args.no_elastix and elastix_available()
    if not args.no_elastix and not with_elastix:
        print("itk-elastix is not installed: skipped (pip install itk-elastix)")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    truth = true_matrix((4.0, -3.0, 5.0), (2.5, -1.5, 2.0))
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, devices {devices}")
    for size in args.sizes:
        shape = (size,) * 3
        fixed = make_phantom(shape)
        moving = warp(fixed, truth, (1.0,) * 3)
        origin = -(torch.tensor(shape) - 1) / 2
        for samples in args.samples:
            for device in devices:
                seconds, result = timed(fixed, moving, origin, device, samples)
                error = corner_error(result.matrix.cpu(), truth, shape, (1.0,) * 3)
                print(f"{size}^3  {samples:6d} samples  {device:7s}  {seconds:6.2f} s  error {error:.3f} mm", flush=True)
            if with_elastix:
                seconds, matrix = timed_elastix(fixed, moving, origin, samples)
                error = corner_error(matrix, truth, shape, (1.0,) * 3)
                print(f"{size}^3  {samples:6d} samples  {'elastix':7s}  {seconds:6.2f} s  error {error:.3f} mm", flush=True)


if __name__ == "__main__":
    main()

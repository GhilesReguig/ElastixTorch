# ElastixTorch

Elastix-style [2] 3D image registration (multi-resolution, Mattes mutual information, adaptive stochastic gradient
descent) in native PyTorch, on CPU or CUDA. The only dependency is `torch`.

```python
from elastixtorch import register

# fixed, moving: 3D float tensors; spacings and origins in mm, in the order of the tensor axes
result = register(fixed, moving, fixed_spacing=(1.0, 1.0, 1.0), transform="rigid", metric="mattes")
result.matrix                      # 4x4: physical point of the fixed image -> physical point of the moving image
result.parameters                  # rigid: 3 angles (rad), 3 translations (mm)
aligned = result.resample(moving)  # moving image on the fixed grid, cubic B-spline interpolation
```

`Registration(...)` takes the same options and can be reused. `device="cuda"` (or CUDA input tensors) runs
everything on the GPU; the result then stays on that device.

## Installation

```bash
pip install ElastixTorch
```

For the GPU, install a CUDA build of PyTorch first (see
https://pytorch.org/get-started/locally/).

From the sources:

```bash
pip install git+https://github.com/GhilesReguig/ElastixTorch
# or, to modify the code and run the tests:
git clone https://github.com/GhilesReguig/ElastixTorch
cd ElastixTorch
pip install -e ".[test]"
pytest
```

## How it maps to Elastix

| Elastix | Here |
|---|---|
| `MultiResolutionRegistration`, smoothing image pyramid | `num_resolutions=4`; Gaussian smoothing (sigma = 0.5 x factor voxels), then downsampling by 2^(R-1-r) (smaller factors on an axis that would get fewer than 8 voxels) |
| `EulerTransform`, `TranslationTransform`, `SimilarityTransform`, `AffineTransform` | `transform="euler"` (alias `"rigid"`), `"translation"`, `"similarity"`, `"affine"`; centre of rotation = centre of the fixed image |
| `AdvancedMattesMutualInformation` | `metric="mattes"`: 32 bins, cubic B-spline window on the moving intensity, zero-order window on the fixed one, bin mapping of Elastix's `ParzenWindowHistogramImageToImageMetric` (see below) |
| `AdvancedMeanSquares`, `AdvancedNormalizedCorrelation` | `metric="mse"`, `"ncc"` |
| `RandomCoordinate` sampler, new samples every iteration | 2048 random continuous positions per iteration (`num_samples`) |
| `AdaptiveStochasticGradientDescent` | gain `a / (t + A)^alpha`, time `t` driven by a sigmoid of the product of successive gradients (Klein et al. 2009 [1]) |
| `AutomaticParameterEstimation`, `AutomaticScalesEstimation` | `a` is set from gradients measured around the initial parameters so that the first step is `max_step_length` mm; the sigmoid minimum and width come from the signal and noise of the gradient; parameters are scaled by their physical effect (mm per unit) |
| `BSplineInterpolationOrder 1`, `FinalBSplineInterpolationOrder 3` | linear interpolation during the optimisation; `final_order=3` for the resampling (exact cubic B-spline prefilter) |
| physical coordinates (spacing, origin) | yes; the metric is evaluated in mm |

The gradient of the metric is analytic, as in Elastix: derivative of the metric w.r.t. each moving value x
gradient of the interpolated image x Jacobian of the transform. Autograd is only used for the derivative of
the 3x3 matrix w.r.t. the parameters.

## Differences from Elastix

- **No B-spline (deformable) transform**, no masks, no direction cosines (the axes are the tensor axes), 3D only.
- The pyramid is smoothed **and downsampled** (each level computed from the next finer one), whereas the
  default Elastix pyramid (`FixedSmoothingImagePyramid`) only smooths. The results are the same (see below).
- The ASGD parameter estimation is simpler than Elastix's `AutomaticParameterEstimationOriginal`
  (`elxAdaptiveStochasticGradientDescent.hxx`):
  - `f_min = -0.99 + 0.98 * signal / (signal + noise)` is the Elastix formula, but here the signal is the
    squared norm of the mean of 6 random-sample gradients and the noise their variance, at the current
    parameters, and `omega = 0.1 * noise`. Elastix compares "exact" gradients (100000 grid points) with
    random-sample ones at perturbed parameters, and scales `omega` with terms of the Jacobian of the transform.
  - `a` is set so that the first step, with the mean gradient norm over 8 draws around the current parameters,
    is `max_step_length`. Elastix uses `a = A delta / (sigma1 sqrt(maxJCJ)) * signal / (signal + noise)`.
  - The step is clipped to `max_step_length` (default: a quarter of the mean voxel spacing of the level).
    Without the clip, a single noisy gradient can throw the image out of the field of view.
- Parameters are given as arguments, not as parameter files.

## Accuracy compared with itk-elastix

itk-elastix [3, 4] is the Python package of Elastix.

Colin27 T1w at 2 mm, rigid, Mattes MI. Mean / worst landmark error (mm) after registering 4 known rigid
transforms (2° / 2 mm to 15° / 15 mm, 8 landmarks), 3 seeds each here (Elastix gives the same result at every
run).

| | 2048 samples | 4096 samples |
|---|---|---|
| itk-elastix | 0.013 / 0.016 | 0.018 / 0.021 |
| ElastixTorch | 0.014 / 0.027 | 0.016 / 0.023 |

The remaining differences are of the order of the spread between seeds.

### What was tried

Several variants were compared on the same cases; those that did not help were removed from the code.

| Variant | 2048 samples | 4096 samples | |
|---|---|---|---|
| Mattes histogram with bins 2 to 29 for both images, out-of-range values clipped | 0.016 / 0.033 | 0.017 / 0.023 | removed: never better, same time |
| ... and a fixed sigmoid minimum (`f_min = -0.8`) | 0.028 / 0.038 | 0.020 / 0.031 | removed |
| ... and Elastix's exact ASGD estimation | 0.025 / 0.036 | 0.018 / 0.026 | removed: worse, about 15 % slower |
| Elastix's histogram, exact estimation and smoothing pyramid | 0.022 / 0.034 | 0.018 / 0.026 | removed |

- **The Mattes histogram of Elastix** (`ParzenWindowHistogramImageToImageMetric`) brings the results closest
  to Elastix. Compared with the earlier one, only the bin mapping changes (the normalisation by the number of
  valid samples, the Parzen windows and the derivative are equivalent): the intensity ranges of each pyramid
  level are widened by 1 % on each side; the fixed values fill bins 0 to 31 (rounded), the moving values
  positions 1 to 30 (a border of 1 bin for the cubic window). Elastix also compresses moving values outside the
  image range with an exponential limiter; it is not implemented here, as linearly interpolated values never
  leave the range. Elastix computes the min / max on the images of each level, as here: with a voxel 100 times
  brighter than the tissue, which sets the range at full resolution but is smoothed away at the coarse levels,
  Elastix's metric at the identity is -1.146 / -0.917 / -0.350 at the three coarse levels, against
  -1.148 / -0.861 / -0.342 here with per-level ranges and about -0.001 with the full-resolution range.
- **The decay of the gain.** With a fixed sigmoid minimum (-0.8), the ASGD time `t` only reached 16 to 31 at the
  end of each resolution (Elastix: 83 to 122). The gain stayed at about half its initial value, and the last steps
  (0.034 mm at full resolution) were as large as the final error. Estimating the sigmoid from the signal and
  noise of the gradient brings `t` to 110 to 156 and the last steps to 0.007 mm.
- **Measuring the gradients around the initial parameters** rather than only at them: 0.036 -> 0.028 mm with
  the fixed sigmoid, and fewer bad cases.
- **Elastix's exact ASGD estimation** (Jacobian terms of the transform, "exact" gradients on 100000 grid points,
  perturbed parameters) gives parameters of the same order as Elastix's on the same case (5° / 5 mm: `a` 254 /
  209 / 100 / 67 against 313 / 144 / 74 / 50, `f_min` and `omega` within a factor 1 to 3), but no better
  results: the remaining gap is not explained by the parameter estimation.
- **The pyramid**: with Elastix's smoothing pyramid (no downsampling) the errors are the same to within
  0.002 mm in every run. The coarse levels only bring the transform close to the solution; the result is set by
  the full-resolution level.
- No effect beyond the spread between seeds: more iterations, Sobol or stratified sampling, cubic interpolation
  during the optimisation (it reduces the bias of the full-image metric optimum from 0.029 to 0.009 mm, but is
  slower), and the sigmoid width alone. The trilinear interpolation by one gather of the 8 neighbours and the
  pyramid computed in cascade give the same results as the generic code, faster.

## Speed

`python tools/benchmark_device.py` on a CUDA machine (torch 2.2.2, 10 CPU threads; synthetic phantom, rigid,
Mattes MI, 4 x 250 iterations; Elastix with its default rigid parameter map, the same number of samples and its
default threads; one run per line):

| | CPU | CUDA | itk-elastix |
|---|---|---|---|
| 96^3, 2048 samples | 4.85 s / 0.022 mm | 5.21 s / 0.013 mm | 0.87 s / 0.024 mm |
| 96^3, 20000 samples | 10.64 s / 0.010 mm | 5.23 s / 0.010 mm | 2.56 s / 0.019 mm |
| 192^3, 2048 samples | 5.11 s / 0.024 mm | 5.21 s / 0.012 mm | 1.29 s / 0.032 mm |
| 192^3, 20000 samples | 11.14 s / 0.011 mm | 5.25 s / 0.011 mm | 3.49 s / 0.015 mm |

On the GPU the time stays at about 5.2 s (5.2 ms per iteration) whatever the image size and the number of
samples. This is the GPU's fixed cost per iteration, not computation: each iteration is a few dozen small
operations, each one a kernel launch, and it reads a few scalars back to Python (loss, number of valid samples,
ASGD time), which makes it wait for the GPU. More samples are therefore almost free on a GPU: with 20000
samples, CUDA is twice as fast as the CPU. Elastix stays 1.5 to 6 times faster than CUDA.

## References

[1] S. Klein, J.P.W. Pluim, M. Staring and M.A. Viergever, "Adaptive stochastic gradient descent optimisation
for image registration", International Journal of Computer Vision, vol. 81, no. 3, pp. 227-239, 2009,
https://doi.org/10.1007/s11263-008-0168-y.

[2] S. Klein, M. Staring, K. Murphy, M.A. Viergever and J.P.W. Pluim, "elastix: a toolbox for intensity-based
medical image registration", IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196-205, January 2010,
https://doi.org/10.1109/TMI.2009.2035616.

[3] K. Ntatsis, N. Dekker, V. Valk, T. Birdsong, D. Zukić, S. Klein, M. Staring, M. McCormick,
"itk-elastix: Medical image registration in Python", Proceedings of the 22nd Python in Science
Conference, pp. 101-105, 2023, https://doi.org/10.25080/gerudo-f2bc6f59-00d.

[4] D.P. Shamonin, E.E. Bron, B.P.F. Lelieveldt, M. Smits, S. Klein and M. Staring, "Fast Parallel Image
Registration on CPU and GPU for Diagnostic Classification of Alzheimer's Disease", Frontiers in
Neuroinformatics, vol. 7, no. 50, pp. 1-15, January 2014, https://doi.org/10.3389/fninf.2013.00050.

## License

MIT, see [LICENSE](https://github.com/GhilesReguig/ElastixTorch/blob/main/LICENSE).

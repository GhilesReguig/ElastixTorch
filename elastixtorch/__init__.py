"""Elastix-style image registration in native PyTorch (the only dependency is torch)."""
from .interpolation import cubic_prefilter, sample
from .metrics import METRICS, MattesMutualInformation, MeanSquares, NormalizedCorrelation
from .optimizer import AdaptiveStochasticGradientDescent
from .registration import Registration, RegistrationResult, register
from .resample import resample
from .transforms import TRANSFORMS, euler_matrix

__all__ = ["Registration", "RegistrationResult", "register", "resample", "sample", "cubic_prefilter",
           "METRICS", "MattesMutualInformation", "MeanSquares", "NormalizedCorrelation",
           "AdaptiveStochasticGradientDescent", "TRANSFORMS", "euler_matrix"]
__version__ = "0.1.0"

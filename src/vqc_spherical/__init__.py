"""
vqc_spherical -- variational quantum circuits on pure spherical manifolds.

Import public classes from here; run experiments as modules with
``python -m vqc_spherical.<module>``.
"""

from .AnsatzLayer import AnsatzLayer
from .EmbeddingLayer import EmbeddingLayer
from .VQCModel import VQCModel
from .VQCOptimizer import Trainer

__all__ = ["EmbeddingLayer", "AnsatzLayer", "VQCModel", "Trainer"]

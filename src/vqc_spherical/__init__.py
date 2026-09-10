"""
vqc_spherical -- variational quantum circuits on pure spherical manifolds.

The package is importable two ways, because both are used:

  * as a package, `from vqc_spherical import VQCModel`, which is what the test
    suite and any external consumer does;
  * as a flat directory, `import VQCModel`, which is what the experiment
    scripts do when run from inside this folder.

Every module therefore attempts a relative import first and falls back to a
flat one, rather than committing to a single style and breaking the other.
"""

try:
    from .EmbeddingLayer import EmbeddingLayer
    from .AnsatzLayer import AnsatzLayer
    from .VQCModel import VQCModel
    from .VQCOptimizer import Trainer
except ImportError:  # pragma: no cover - flat sys.path
    from EmbeddingLayer import EmbeddingLayer
    from AnsatzLayer import AnsatzLayer
    from VQCModel import VQCModel
    from VQCOptimizer import Trainer

__all__ = ["EmbeddingLayer", "AnsatzLayer", "VQCModel", "Trainer"]

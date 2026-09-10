"""
Single source of truth for every sweep constant.

PREDICTIONS.md quotes numbers from this file.  If a sweep changes here and the
pre-registration is not updated in its dated appendix, the two have drifted and
the pre-registration is void -- which is the whole reason for keeping them in
one place.
"""

import numpy as np

# ----------------------------------------------------------------------
# Seeds.  Three INDEPENDENT streams: one seed shared between data, weight
# initialisation and the train/test split correlates runs in ways that stay
# invisible until someone asks.
# ----------------------------------------------------------------------
N_SEEDS = 10
DATA_SEED_BASE = 1000
INIT_SEED_BASE = 2000
SPLIT_SEED_BASE = 3000

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
N_SAMPLES = 400
TEST_FRACTION = 0.25
NOISE_STD = 0.05

# Population-scale settings for the analytic ceilings (no training involved)
N_CEILING = 20000
N_DIRECTIONS = 4096
N_PROBE = 3000

# ----------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------
DEPTHS = (2, 4, 6, 10)
DEPTH_DEFAULT = 2
UPLOADS = (1, 2, 3, 4)
SPIN_QUBITS = (1, 2, 3, 4)
FIXED_BUDGET = 8          # U * L held constant, to decouple the ladder from
                          # the parameter count

# ----------------------------------------------------------------------
# Optimisation.  Two learning rates, always: one cannot distinguish "cannot
# express it" from "did not find it".
# ----------------------------------------------------------------------
EPOCHS = 150
STEPSIZES = (0.05, 0.20)
OPTIMIZER = "adam"

# ----------------------------------------------------------------------
# Representations, held against a fixed classifier
# ----------------------------------------------------------------------
REPRESENTATIONS = ("bloch", "cartesian", "sincos", "raw")
FFNN_HIDDEN = 8
FFNN_EPOCHS = 600

# ----------------------------------------------------------------------
# Faithfulness.  Rotation conjugation, not a latitude sweep: rotating a fixed
# dataset preserves every geodesic distance exactly.
# ----------------------------------------------------------------------
POLAR_ANGLES = tuple(np.linspace(0.0, np.pi / 2, 7))
N_RANDOM_ROTATIONS = 12
ROTATION_SEED_BASE = 4000

# Calibrated, not guessed: at separation pi/6 and noise 0.07 the affine ceiling
# of the band task is 0.884 -- above chance and clearly below saturation, so
# there is room for a representation to be worse.  The largest single risk to
# this project is a benchmark every model solves, which measures nothing.
BAND_SEPARATION = np.pi / 6
BAND_NOISE = 0.07
BAND_CEILING = 0.884
SATURATION_LIMIT = 0.97

TRAIN_SIZES = (10, 20, 40, 80, 160)
N_SEEDS_EFFICIENCY = 20

# ----------------------------------------------------------------------
# Numerics
# ----------------------------------------------------------------------
RANK_RTOL = 1e-10         # eigenvalue cutoff, relative to the largest
RESIDUAL_TOL = 1e-10      # "lives entirely in ell <= L"

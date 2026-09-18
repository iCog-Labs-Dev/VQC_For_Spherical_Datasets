"""
The classical controls that appear in EVERY table.

The most important is BestAffine.  It is not trained: it computes, by
exhaustive scan, the best classifier of the form alpha + beta.r -- exactly the
hypothesis class of the single-upload VQC.  Reporting it turns the
dequantisation statement from something a reviewer discovers into something the
work owns: a four-parameter classical model matches the circuit on every
degree-1 task, and on the degree-2 targets both sit at the same computed
ceiling.

LogisticOnCartesian is the softer version of the same check -- same hypothesis
class, but optimising log-loss rather than accuracy, so it lands at or below
BestAffine.

All models expose fit / predict / param_count so runners can loop over them.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from . import Analysis as A, Config as C, Kernels as K, Representations as R


class BestAffine:
    """
    The provable ceiling of any degree-1 model, as a fitted object.

    Not an optimiser: for each of ~4000 directions the optimal threshold is
    found exactly by sorting, and the best pair kept.  No seeds, no learning
    rate, no epochs -- so it cannot be accused of an optimisation failure,
    which is the whole reason it exists.
    """

    name = "BestAffine"

    def __init__(self, n_directions=C.N_DIRECTIONS, balanced=False):
        self.n_directions = n_directions
        self.balanced = balanced
        self.flip_ = False

    def fit(self, theta, phi, y):
        res = A.affine_ceiling(theta, phi, y, self.n_directions, balanced=self.balanced)
        self.direction_ = res["direction"]
        self.threshold_ = res["threshold"]
        self.ceiling_ = res["accuracy"]
        # affine_ceiling maximises over both orientations; recover which one
        if np.mean(self._raw_predict(theta, phi) == np.asarray(y).astype(int)) < 0.5:
            self.flip_ = True
        return self

    def _raw_predict(self, theta, phi):
        return (K.to_cartesian(theta, phi) @ self.direction_ > self.threshold_).astype(int)

    def predict(self, theta, phi):
        p = self._raw_predict(theta, phi)
        return 1 - p if self.flip_ else p

    @staticmethod
    def param_count():
        return 4      # alpha plus three components of beta


class _SklearnOnRepresentation:
    """Shared plumbing: featurise, fit, predict."""

    representation = None
    name = None

    def _make(self):
        raise NotImplementedError

    def fit(self, theta, phi, y):
        self.model_ = self._make()
        self.model_.fit(R.featurize(theta, phi, self.representation),
                        np.asarray(y).astype(int))
        return self

    def predict(self, theta, phi):
        return self.model_.predict(R.featurize(theta, phi, self.representation))


class LogisticOnCartesian(_SklearnOnRepresentation):
    """
    The dequantisation check.  Four parameters, the SAME hypothesis class as the
    single-upload VQC.  Include it in every table.
    """

    representation = "cartesian"
    name = "LogisticOnCartesian"

    def _make(self):
        return LogisticRegression(max_iter=2000, C=1e4)

    @staticmethod
    def param_count():
        return 4


class LogisticOnSinCos(_SklearnOnRepresentation):
    """
    Five parameters on the torus representation.

    Read its results carefully: sin(theta) is one of its features, so a target
    like |z| > 1/sqrt(3) -- which is exactly sin(theta) < sqrt(2/3) -- is
    linearly separable for it.  sin(theta) is not a polynomial in (x, y, z), so
    this representation is NOT degree-limited and can be strictly more
    expressive than the circuit while being less faithful.  That is the honest
    comparison, and it is why the claim is about priors rather than power.
    """

    representation = "sincos"
    name = "LogisticOnSinCos"

    def _make(self):
        return LogisticRegression(max_iter=2000, C=1e4)

    @staticmethod
    def param_count():
        return 5


class HarmonicRegression:
    """
    Linear model on the real spherical harmonics up to degree L, thresholded.

    (L + 1)^2 parameters, expressing exactly the functions of degree <= L: the
    parameter-matched classical twin of the circuit.  A gap against it is about
    the encoding rather than about capacity.
    """

    def __init__(self, ell_max=1, ridge=1e-8):
        self.ell_max = ell_max
        self.ridge = ridge
        self.name = f"HarmonicRegression(L={ell_max})"

    def fit(self, theta, phi, y):
        X = A.harmonic_design(theta, phi, self.ell_max)
        t = 2.0 * np.asarray(y, dtype=float) - 1.0
        G = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(G, X.T @ t)
        return self

    def predict(self, theta, phi):
        return (A.harmonic_design(theta, phi, self.ell_max) @ self.coef_ > 0).astype(int)

    def param_count(self):
        return (self.ell_max + 1) ** 2


class FFNNOnRepresentation(_SklearnOnRepresentation):
    """
    Capacity-matched feed-forward net; the representation is a CONSTRUCTOR
    argument so the classifier is structurally fixed while the encoding varies.
    Identical architecture, optimiser, epoch budget and seed across
    representations -- a gap produced by unequal training budgets is worthless.
    """

    def __init__(self, representation="cartesian", hidden=C.FFNN_HIDDEN,
                 epochs=C.FFNN_EPOCHS, seed=0):
        self.representation = representation
        self.hidden = hidden
        self.epochs = epochs
        self.seed = seed
        self.name = f"FFNN({representation})"

    def _make(self):
        return MLPClassifier(hidden_layer_sizes=(self.hidden,), max_iter=self.epochs,
                             random_state=self.seed, learning_rate_init=0.01,
                             solver="adam", n_iter_no_change=self.epochs)

    def param_count(self):
        d = R.n_features(self.representation)
        return d * self.hidden + self.hidden + self.hidden + 1


class MajorityBaseline:
    """
    Predicts the training-set majority class.  Trivial, and indispensable: the
    quadrupole is 42/58, so "scored 58%" and "predicts one class for
    everything" are otherwise indistinguishable.
    """

    name = "Majority"

    def fit(self, theta, phi, y):
        self.cls_ = int(np.round(np.mean(np.asarray(y))))
        return self

    def predict(self, theta, phi):
        return np.full(len(np.asarray(theta)), self.cls_, dtype=int)

    @staticmethod
    def param_count():
        return 0

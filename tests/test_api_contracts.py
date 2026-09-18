"""Input validation and error-propagation contracts for the public API."""

import numpy as np
import pytest

from vqc_spherical.AnsatzLayer import AnsatzLayer
from vqc_spherical.EmbeddingLayer import EmbeddingLayer
from vqc_spherical.VQCModel import VQCModel
from vqc_spherical.VQCOptimizer import Trainer


@pytest.mark.parametrize("n_qubits", [0, -1])
def test_model_requires_at_least_one_qubit(n_qubits):
    with pytest.raises(ValueError, match="n_qubits"):
        VQCModel(n_qubits)


@pytest.mark.parametrize("n_uploads", [0, -1])
def test_model_requires_at_least_one_upload(n_uploads):
    with pytest.raises(ValueError, match="n_uploads"):
        VQCModel(1, n_uploads=n_uploads)


@pytest.mark.parametrize("dropout_rate", [-0.1, 1.0, 1.5])
def test_model_validates_dropout_probability(dropout_rate):
    with pytest.raises(ValueError, match="dropout_rate"):
        VQCModel(1, dropout_rate=dropout_rate)


def test_embedding_rejects_incomplete_or_excess_coordinate_pairs():
    embedding = EmbeddingLayer("spherical")
    with pytest.raises(ValueError, match="theta/phi pairs"):
        embedding.apply([0.1], range(1))
    with pytest.raises(ValueError, match="2 coordinate pairs"):
        embedding.apply([0.1, 0.2, 0.3, 0.4], range(1))


def test_broadcast_embedding_accepts_exactly_one_coordinate_pair():
    embedding = EmbeddingLayer("broadcast")
    with pytest.raises(ValueError, match="one theta/phi pair"):
        embedding.apply([0.1, 0.2, 0.3, 0.4], range(2))


def test_ansatz_requires_a_positive_layer_count():
    with pytest.raises(ValueError, match="n_layers"):
        AnsatzLayer(n_layers=0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"stepsize": 0}, "stepsize"),
    ],
)
def test_trainer_validates_optimisation_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Trainer(object(), **kwargs)


def test_batched_evaluation_does_not_hide_model_errors():
    class BrokenModel:
        def forward(self, features, weights):
            raise RuntimeError("circuit failed")

    trainer = Trainer(BrokenModel())
    with pytest.raises(RuntimeError, match="circuit failed"):
        trainer._evaluate(np.zeros(1), np.zeros((3, 2)))

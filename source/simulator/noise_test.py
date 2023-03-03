import pytest
from typing import Sequence
import numpy
from openfermion import get_sparse_operator
from random import random
from math import sqrt

from .noise import *


def apply_noise(
        state: numpy.ndarray,
        noise_model: Sequence,
        n_qubits: int = 1,
        ) -> numpy.ndarray:

    return sum([
        noise_model[1][i] *
        get_sparse_operator(x, n_qubits) @
        state @
        get_sparse_operator(x, n_qubits)
        for i, x in enumerate(noise_model[0])])


def random_vector(n_log_qubits):
    vec = []
    norm = 0
    for x in range(2 ** n_log_qubits):
        vec += [2 * random() - 1 + 1.j * (2 * random() - 1)]
        norm += abs(vec[-1]) ** 2
    return numpy.array(vec) / sqrt(norm)


def test_single_qubit_noise():
    for z in range(12):
        vec = random_vector(1)
        dmatrix = numpy.kron(vec, vec.conj()).reshape((2, 2))
        rate = random()
        numpy.testing.assert_allclose(
            apply_noise(
                dmatrix,
                single_qubit_dephasing(rate)
            ),
            (1. - rate) * dmatrix + rate * numpy.diag(abs(vec) ** 2)
        )
        numpy.testing.assert_allclose(
            apply_noise(
                dmatrix,
                single_qubit_depolarizing(rate)
            ),
            (1. - rate) * dmatrix + rate * numpy.diag([.5, .5])
        )
        numpy.testing.assert_allclose(
            apply_noise(
                dmatrix,
                noiseless()
            ),
            dmatrix
        )


def test_noise_checks():
    functions = [
        single_qubit_dephasing,
        single_qubit_depolarizing,
        two_qubit_dephasing,
        two_qubit_depolarizing
    ]
    for x in functions:
        with pytest.raises(NoiseError):
            outputs = x(1.5)
        with pytest.raises(NoiseError):
            outputs = x(-.4)


def test_two_qubit_noise():
    for z in range(12):
        vec_1 = random_vector(1)
        vec_2 = random_vector(1)
        dmatrix = numpy.kron(
            numpy.kron(vec_1, vec_1.conj()).reshape((2, 2)),
            numpy.kron(vec_2, vec_2.conj()).reshape((2, 2)))
        rate = random()
        dmatrix_1 = numpy.kron(
            numpy.kron(vec_1, vec_1.conj()).reshape((2, 2)),
            numpy.eye(2))
        dmatrix_2 = numpy.kron(
            numpy.eye(2),
            numpy.kron(vec_2, vec_2.conj()).reshape((2, 2)))
        numpy.testing.assert_allclose(
            apply_noise(
                dmatrix,
                two_qubit_depolarizing(rate),
                2
            ),
            (
                ((1 - rate) ** 2) * dmatrix +
                ((1 - rate) * rate) * (dmatrix_1 + dmatrix_2) / 2 +
                (rate ** 2) * numpy.eye(4) / 4
            )
        )
        dmatrix_1 = numpy.kron(
            numpy.kron(vec_1, vec_1.conj()).reshape((2, 2)),
            numpy.diag(abs(vec_2) ** 2))
        dmatrix_2 = numpy.kron(
            numpy.diag(abs(vec_1) ** 2),
            numpy.kron(vec_2, vec_2.conj()).reshape((2, 2)))
        numpy.testing.assert_allclose(
            apply_noise(
                dmatrix,
                two_qubit_dephasing(rate),
                2
            ),
            (
                ((1 - rate) ** 2) * dmatrix +
                ((1 - rate) * rate) * (dmatrix_1 + dmatrix_2) +
                (rate ** 2) * numpy.kron(
                    numpy.diag(abs(vec_1) ** 2),
                    numpy.diag(abs(vec_2) ** 2),
                )
            )
        )

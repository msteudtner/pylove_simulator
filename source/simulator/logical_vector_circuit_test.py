"""Test functions for logical vector circuit."""

import pytest
from openfermion import QubitOperator

from .noise import noiseless
from random import choices, random
from math import cos, sin

from .logical_vector_circuit import (
    cnot_sandwich,
    half_noisy_cnot_sandwich,
    noisy_cnot_sandwich,
    make_noisy_rotation,
    noisy_state_prep,
    cnot_noise_operation,
    statistical_noise_operation,
)


cnot = (
    .5 * (QubitOperator(()) - QubitOperator('Z0')) * QubitOperator('X1') +
    .5 * (QubitOperator(()) + QubitOperator('Z0')))


def biased_noise():
    return [[QubitOperator('Y0 Y1'), ], [1., ]]


@pytest.mark.parametrize('op', [
    QubitOperator(''),
    QubitOperator('X0'),
    QubitOperator('Y0'),
    QubitOperator('Z0'),
    QubitOperator('X0 X1'),
    QubitOperator('Y0 X1'),
    QubitOperator('Z0 X1'),
    QubitOperator('X0 Y1'),
    QubitOperator('Y0 Y1'),
    QubitOperator('Z0 Y1'),
    QubitOperator('X0 Z1'),
    QubitOperator('Y0 Z1'),
    QubitOperator('Z0 Z1'),
    QubitOperator('Z0 X1 X2'),
    QubitOperator('X0 Z1 Z2 Y3', 1.12),
    (QubitOperator('X0 X1 Y2 Y3 Z6 X8 Y11', -.355j) +
        QubitOperator(' Y0 Y1', .785)),
])
def test_cnot_sandwiches_without_noise(op):
    homemade_sandwich = cnot * op * cnot
    homemade_sandwich.compress()
    assert homemade_sandwich.isclose(cnot_sandwich(op, 0, 1))
    assert homemade_sandwich.isclose(
        noisy_cnot_sandwich(op, 0, 1, noiseless(), noiseless()))
    assert homemade_sandwich.isclose(
        half_noisy_cnot_sandwich(op, 0, 1, noiseless(), noiseless(), True))
    assert homemade_sandwich.isclose(
        half_noisy_cnot_sandwich(op, 0, 1, noiseless(), noiseless(), False))

    lefty = QubitOperator('Y0 Y1') * cnot * op * cnot
    righty = cnot * op * QubitOperator('Y0 Y1') * cnot
    leftyrighty = (
        QubitOperator('Y0 Y1') *
        cnot *
        op *
        QubitOperator('Y0 Y1') *
        cnot)
    lefty.compress()
    righty.compress()
    leftyrighty.compress()

    assert lefty.isclose(
        half_noisy_cnot_sandwich(op, 0, 1, noiseless(), biased_noise(), True))
    assert righty.isclose(
        half_noisy_cnot_sandwich(
            op,
            0,
            1,
            noiseless(),
            biased_noise(),
            False))
    assert leftyrighty.isclose(
        noisy_cnot_sandwich(op, 0, 1, noiseless(), biased_noise()))


def random_pstring():
    sigma = choices('XYZI', k=20)
    ptuple = ()
    for z in range(20):
        if sigma[z] != 'I':
            ptuple += ((z, sigma[z]),)
    return ptuple


def test_make_noisy_rotation():
    for x in range(20):
        pstring = random_pstring()
        angle = (random() - .5) * 10.

        assert make_noisy_rotation(
            pstring,
            angle,
            noiseless(),
            noiseless()
        )[0].isclose(
            cos(angle) * QubitOperator(()) +
            1.j * sin(angle) * QubitOperator(pstring))


def test_noisy_state_prep():
    for x in range(20):
        for y in noisy_state_prep(
                    random_pstring(),
                    noiseless(),
                    noiseless())[:2]:
            assert y.isclose(QubitOperator(()))


def random_integers(num=2):
    pool = [x for x in range(20)]
    chosen = []
    for x in range(num):
        integer = choices(pool)[0]
        pool.remove(integer)
        chosen += [integer]
    return tuple(chosen)


def test_cnot_noise_operation():
    for x in range(10):
        a, b = random_integers()
        assert cnot_noise_operation(
            a,
            b,
            biased_noise()).isclose(QubitOperator(((a, 'Y'), (b, 'Y'))))


def biased_single_qubit_noise():
    return [[QubitOperator('Y0')], [1.0, ]]


def test_statistical_noise_operation():
    ints = choices([x for x in range(200)], k=20)
    times = choices([x for x in range(50)], k=20)
    for x, y in zip(ints, times):
        assert statistical_noise_operation(
            x,
            biased_single_qubit_noise(),
            y).isclose(QubitOperator(((x, 'Y'))) ** y)

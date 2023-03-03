"""Test functions for logical vector construction."""

from openfermion import QubitOperator
import pytest
import numpy
from random import choices, random

from .logical_vector_construction import (
    SimulatorError,
    check_for_imaginary_coeff,
    check_for_identity,
    fix_single_term,
    fix_single_term_plus,
    reduce_to_logical,
    reduce_to_logical_plus,
    generator_dependence
)


def test_check_for_no_imaginary_coeff_():
    try:
        check_for_imaginary_coeff(
            [
                QubitOperator('X0 Z1 Y2', -.211),
                QubitOperator('Y1 Z4 Z4'),
                QubitOperator('X2 X4 X5', .453),
                QubitOperator('Y7 X9 Y11 X12', -245.239)
            ],
            'Error'
        )
    except SimulatorError:
        pytest.fail("Unexpected SimulatorError.")


def test_check_for_imaginary_coeff():
    with pytest.raises(SimulatorError):
        check_for_imaginary_coeff(
            [
                QubitOperator('X0 Z1 Y2', -.211),
                QubitOperator('Y1 Z4 Z4'),
                QubitOperator('X2 X4 X5', -.045j),
                QubitOperator('Y7 X9 Y11 X12', -245.239)
            ],
            'Error'
        )


def test_check_for_no_identity():
    try:
        check_for_identity(
            [
                QubitOperator('X0 Z1 Y2', -.211),
                QubitOperator('Y1 Z4 Z4'),
                QubitOperator('X2 X4 X5', .913 - .045j),
                QubitOperator('Y7 X9 Y11 X12', -245.239)
            ],
            'Error'
        )
    except SimulatorError:
        pytest.fail("Unexpected SimulatorError.")


def test_check_for_identity():
    with pytest.raises(SimulatorError):
        check_for_identity(
            [
                QubitOperator('X0 Z1 Y2', -.211),
                QubitOperator('Y1 Z4 Z4'),
                QubitOperator(' ', .913 - .045j),
                QubitOperator('Y7 X9 Y11 X12', -245.239)
            ],
            'Error')


def test_fix_single_term():
    assert fix_single_term(
        QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1),
        0,
        'Y',
        'X',
        QubitOperator('Z2 Y4 Y4', -1.)).isclose(
            QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1) *
            QubitOperator('Z2 Y4 Y4', -1.)
        )

    assert fix_single_term(
        QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1),
        2,
        'Y',
        'X',
        QubitOperator('Z2 Y4 Y4', -1.)).isclose(
            QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1)
        )

    assert fix_single_term(
        QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1),
        3,
        'Y',
        'X',
        QubitOperator('Z2 Y4 Y4', -1.)).isclose(
            QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1) *
            QubitOperator('Z2 Y4 Y4', -1.)
        )

    assert fix_single_term_plus(
        QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1),
        0,
        'Y',
        'X',
        QubitOperator('Z2 Y4 Y4', -1.)) == (
            QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1) *
            QubitOperator('Z2 Y4 Y4', -1.),
            1
        )

    assert fix_single_term_plus(
        QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1),
        2,
        'Y',
        'X',
        QubitOperator('Z2 Y4 Y4', -1.)) == (
            QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1),
            0
        )

    assert fix_single_term_plus(
        QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1),
        3,
        'Y',
        'X',
        QubitOperator('Z2 Y4 Y4', -1.)) == (
            QubitOperator('X0 Y1 Z2 Y3 X4', .2j - .1) *
            QubitOperator('Z2 Y4 Y4', -1.),
            1
        )


def random_pstring(n_qubits, offset=0):
    sigma = choices('XYZI', k=n_qubits)
    ptuple = ()
    for z in range(n_qubits):
        if sigma[z] != 'I':
            ptuple += ((z + offset, sigma[z]),)
    return ptuple


def test_reduce_to_logical():
    for x in range(20):
        log_ops = random_pstring(6)
        first_ops = random_pstring(14)
        syndr = numpy.zeros(14, dtype=int)
        coeff = 4 * random()
        phys_ops = QubitOperator(
            first_ops +
            tuple([(y[0] + 14, y[1]) for y in log_ops]),
            coeff)

        red_op = reduce_to_logical(
            phys_ops,
            syndr,
            [QubitOperator(((y, 'Z'), )) for y in range(14)],
            ['rm'] * 14 + [y for y in range(6)],
            [y for y in range(14)],
            'Z' * 14,
            'X' * 14)

        assert list(red_op.terms.keys()) == list(
                QubitOperator(log_ops).terms.keys())
        assert abs(list(red_op.terms.values())[0]) == coeff

        for z in range(14):
            assert QubitOperator(((z, 'Z'), ), (-1.) ** syndr[z]).isclose(
                QubitOperator(first_ops) *
                QubitOperator(((z, 'Z'), )) *
                QubitOperator(first_ops)
            )

        plus_ops_list, plus_syndr_list = reduce_to_logical_plus(
            phys_ops,
            [QubitOperator(((y, 'Z'), ), (-1.) ** syndr[y])
                for y in range(14)],
            ['rm'] * 14 + [y for y in range(6)],
            [y for y in range(14)],
            'Z' * 14,
            'X' * 14)

        for z in range(14):
            QubitOperator(first_ops, (-1.) ** plus_syndr_list[0][z]).isclose(
                QubitOperator(((z, 'Y'), )) *
                QubitOperator(first_ops) *
                QubitOperator(((z, 'Y'), ))
            )

        assert plus_ops_list[0].isclose(red_op)

        dep_ops, dep_syndr = generator_dependence(
            phys_ops,
            [QubitOperator(((y, 'Z'), )) for y in range(20)],
            [y for y in range(20)],
            'Z' * 20,
            'X' * 20)

        numpy.testing.assert_allclose(dep_syndr[:14], plus_syndr_list[0])
        assert abs(list(dep_ops.terms.values())[0]) == coeff

        for z in range(20):
            assert dep_ops.isclose(
                QubitOperator(((z, 'Y'), )) *
                dep_ops *
                QubitOperator(((z, 'Y'), ))
            )

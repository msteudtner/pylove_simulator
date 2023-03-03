import pytest
from .utils import *
import numpy
from math import sqrt
from random import random, choices, shuffle


def random_vector(n_log_qubits):
    vec = []
    norm = 0
    for x in range(2 ** n_log_qubits):
        vec += [2 * random() - 1 + 1.j * (2 * random() - 1)]
        norm += abs(vec[-1]) ** 2
    return numpy.array(vec) / sqrt(norm)


def make_dmatrix(vector):
    return numpy.kron(
        vector,
        vector.conj()).reshape([len(vector), ] * 2)


def test_is_vector():
    vec = random_vector(6)
    assert is_vector(vec) == True
    assert is_vector(make_dmatrix(vec)) == False


def test_fidelity():
    vec_0 = numpy.array(list(random_vector(6)) + [0])
    vec_1 = numpy.array(vec_0) / sqrt(2)
    vec_1[-1] = 1. / sqrt(2)
    assert numpy.isclose(fidelity(vec_0, vec_1), 1 / sqrt(2))
    assert numpy.isclose(fidelity(make_dmatrix(vec_0), vec_1), 1 / sqrt(2))
    assert numpy.isclose(fidelity(vec_0, make_dmatrix(vec_1)), 1 / sqrt(2))
    assert numpy.isclose(
        fidelity(make_dmatrix(vec_0), make_dmatrix(vec_1)), 1 / sqrt(2))


def rando_graph(n_vertices, n_edges):
    for z in range(4):
        edge_dict = {}
        edge_set = set()
        for x in range(n_vertices):
            edge_dict[x] = set()
        a, b = (
            choices([x for x in range(n_vertices)], k=n_edges),
            choices([x for x in range(n_vertices)], k=n_edges)
            )
        for x, y in zip(a, b):
            if x != y:
                edge_dict[x].add(y)
                edge_dict[y].add(x)
                if (x, y) not in edge_set and (y, x) not in edge_set:
                    edge_set.add((x, y))
        return edge_dict, edge_set


def test_expectation_value():
    for z in range(6):
        coeff = 2 * random() - 1.
        edges, _ = rando_graph(20, 40)
        ops = [QubitOperator(((x, 'X'),
            *[(y, 'Z') for y in edges[x]])) for x in range(20)]
        obs = ops[0] * coeff
        state_vec = random_vector(9)
        dmatrix = make_dmatrix(state_vec)
        outputs = [expectation_value(
            obs,
            z,
            ops[: 11],
            ['rm', ] * 11 + [x for x in range(9)],
            [x for x in range(11)],
            'X' * 11,
            'Y' * 11) for z in (
                state_vec,
                dmatrix,
                {(1,) + (0,) * 10: dmatrix})]
        assert numpy.isclose(outputs[0], coeff)
        assert numpy.isclose(outputs[1], coeff)
        assert numpy.isclose(outputs[2], -coeff)


def test_anticom():
    assert anticom(QubitOperator('X0'), QubitOperator('Z0')) == True
    assert anticom(QubitOperator('X0 X1'), QubitOperator('Y0 Y1')) == False


def test_check_mapping():
    A = {
        (0, 2): QubitOperator('Y0 Z1 X2'),
        (2, 4): QubitOperator('Y2 Z3 X4'),
        (4, 6): QubitOperator('Y4 Z5 X6'),
    }
    B = {
        0: QubitOperator('Z0'),
        2: QubitOperator('Z2'),
        4: QubitOperator('Z4'),
    }
    stabs = [
        QubitOperator('Z7'),
        QubitOperator('Z8'),
        QubitOperator('Z9'),
    ]
    all_ops = [
        (A, {**B, **{6: QubitOperator('Y3 Z6')}}, stabs),
        (A, {**B, **{6: QubitOperator('X6')}}, stabs),
        ({**A, **{(2, 3): QubitOperator('X2 X3')}}, B, stabs),
        (A, {**B, **{5: QubitOperator('X4 X5'),}}, stabs),
        ( A, B, stabs + [QubitOperator('Z0')]),
        (A, B, stabs + [QubitOperator('Y0')]),
        (A, B, stabs + [QubitOperator('Y9')]),
    ]
    for x in all_ops:
        print(x)
        with pytest.raises(SimulatorError):
            check_mapping(*x)


def test_remap_modes():
    for z in range(6):
        new_indexes = [x for x in range(20)]
        shuffle(new_indexes)
        new_mode = {x: new_indexes[x] for x in range(20)}
        _, edges = rando_graph(20, 40)
        A = {x: QubitOperator(((x[0], 'X'), (x[1], 'Y'))) for x in edges}
        B = {x: QubitOperator(((x, 'Z'))) for x in range(20)}
        output = remap_modes(new_mode, [A, B])
        assert len(output) == 2
        for x in range(20):
            assert output[1][new_mode[x]] == QubitOperator(((x, 'Z')))
        for x in edges:
            assert ((new_mode[x[0]], new_mode[x[1]]) in output[0] or
                (new_mode[x[1]], new_mode[x[0]]) in output[0])
            if new_mode[x[0]] < new_mode[x[1]]:
                assert output[0][(new_mode[x[0]],
                    new_mode[x[1]])] == QubitOperator(((x[0], 'X'),
                    (x[1], 'Y')))
            else:
                output[0][(new_mode[x[1]],
                    new_mode[x[0]])] == QubitOperator(((x[0], 'X'),
                    (x[1], 'Y')), -1)
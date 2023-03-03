"""Test functions for logical vector simulator."""

import pytest
import numpy
from random import choices, random
from openfermion import QubitOperator, get_sparse_operator
from math import sqrt, e

from .logical_vector_construction import SimulatorError
from .logical_vector_simulator import (
    make_simulation_schedule,
    prepare_stabilizers,
    simulate_state_prep,
    make_initial_state,
    invert_tri_matrix,
    prepare_state_prep,
    propagate_syndromes,
    _logical_state_simulation_shot,
    logical_state_simulation
    )


def test_prepare_stabilizers():
    assert prepare_stabilizers(
        [
            QubitOperator('Z0'),
            QubitOperator('Z1'),
            QubitOperator('Z2'),
            QubitOperator('Z3'),
            QubitOperator('Z4'),
            QubitOperator('Z5'),
        ]) == (
            [
                QubitOperator('Z0'),
                QubitOperator('Z1'),
                QubitOperator('Z2'),
                QubitOperator('Z3'),
                QubitOperator('Z4'),
                QubitOperator('Z5'),
            ],
            [x for x in range(6)],
            'Z' * 6,
            'Y' * 6,
        )


def test_prepare_stabilizer_checks():

    with pytest.raises(SimulatorError):
        prepare_stabilizers(
            [
                QubitOperator('Z0'),
                QubitOperator('Z1'),
                QubitOperator('Z2'),
                QubitOperator('Z2'),
                QubitOperator('Z4'),
                QubitOperator('Z5')
            ])


def random_pstring(n_qubits, offset=0):
    sigma = choices('XYZI', k=n_qubits)
    ptuple = ()
    for z in range(n_qubits):
        if sigma[z] != 'I':
            ptuple += ((z + offset, sigma[z]),)
    return ptuple


def test_make_simulation_schedule():
    rando_pstrings = [random_pstring(20) for x in range(16)]
    meas_schedule, _ = make_simulation_schedule(
        [QubitOperator(x) for x in rando_pstrings],
        20,
        'meas')
    for y in meas_schedule:
        if y[0] == 'meas':
            assert y[1] == rando_pstrings[y[2]]


def biased_single_qubit_noise(sigma):
    return [
        [QubitOperator(((0, sigma))), QubitOperator(())],
        [.5, .5]]


def biased_two_qubit_noise(sigma):
    return [
        [
            QubitOperator(((0, sigma))),
            QubitOperator(((1, sigma))),
            QubitOperator(((0, sigma), (1, sigma))),
            QubitOperator(())],
        [.25, ] * 4]


def test_simulate_state_prep():
    schedule, _ = make_simulation_schedule(
        [QubitOperator(((x, 'Z'))) for x in range(20)],
        20,
        'meas'
    )
    numpy.testing.assert_allclose(
        numpy.zeros((20), dtype=int),
        simulate_state_prep(
            schedule,
            [QubitOperator(((x, 'Z'))) for x in range(20)],
            biased_single_qubit_noise('Z'),
            biased_two_qubit_noise('Z')))


def rando_graph(n_vertices, n_edges):
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


def CZ(a, b):
    return (
        .5 * (QubitOperator(()) + QubitOperator(((a, 'Z')))) +
        .5 * (QubitOperator(()) - QubitOperator(((a, 'Z')))) *
        QubitOperator(((b, 'Z'))))


def test_make_initial_state():
    for z in range(10):
        h_dict, h_set = rando_graph(10, 30)
        state = numpy.ones(1024, dtype=complex)
        for x, y in h_set:
            state = get_sparse_operator(CZ(x, y), 10).dot(state)
        state = state / 32
        stabs = [QubitOperator((
                (x, 'X'),
                *[(y, 'Z') for y in h_dict[x]])) for x in range(10)]
        vec = numpy.zeros(1024, dtype=complex)
        vec[0] = 1.
        vec, syndr = make_initial_state(
            vec,
            stabs,
            numpy.zeros(12, dtype=int))
        for x in stabs:
            numpy.testing.assert_allclose(
                vec,
                get_sparse_operator(x, 10).dot(vec))
        numpy.testing.assert_allclose(syndr, numpy.zeros(2))
        assert pytest.approx(abs(vec @ vec)) == 1.
        assert pytest.approx(abs(vec @ state)) == 1.


def rando_trimatrix(dim):
    elements = choices([1, 0], k=(dim ** 2 - dim) // 2)
    mtx = numpy.diag([1, ] * dim)
    for x in range(dim - 1):
        mtx[x + 1:, x] = elements[: dim - 1 - x]
        elements = elements[dim - 1 - x:]
    return mtx


def test_invert_tri_matrix():
    for z in range(20):
        mtx = rando_trimatrix(20)
        numpy.testing.assert_allclose(
            (invert_tri_matrix(mtx) @ mtx) % 2,  numpy.diag([1, ] * 20))


def test_prepare_state_prep():
    for z in range(10):
        mtx = rando_trimatrix(20)
        edges, _ = rando_graph(20, 50)
        ops = [QubitOperator((
                    (x, 'X'),
                    *[(y, 'Z') for y in edges[x]])) for x in range(20)]
        stabs = ops[: 11]
        logs = ops[11:]
        preps = [QubitOperator(()), ] * 20
        for x in range(20):
            for y in range(20):
                if mtx[x, y] == 1:
                    preps[x] = preps[x] * ops[y]
        outputs = prepare_state_prep(
            preps,
            logs,
            stabs,
            (['rm', ] * 11) + [x for x in range(9)],
            [x for x in range(11)],
            'X' * 11,
            'Y' * 11,
        )
        numpy.testing.assert_allclose(
            outputs[0],
            numpy.array([1., ] + ([0., ] * 511), dtype=complex))
        numpy.testing.assert_allclose(
            (mtx @ outputs[1]) % 2,
            numpy.eye(20)
        )
        red_logs = [QubitOperator(((x, 'X'))) for x in range(9)]
        for x in range(9):
            for y in edges[x + 11]:
                if y > 10:
                    red_logs[x] = red_logs[x] * QubitOperator(((y - 11, 'Z')))
        assert red_logs == outputs[2]
        new_syndr = propagate_syndromes(
            list(numpy.diag([1, ] * 20)),
            preps,
            ops,
            [x for x in range(20)],
            'X' * 20,
            'Y' * 20
        )
        new_syndr = numpy.transpose(numpy.array(new_syndr))
        numpy.testing.assert_allclose(
            (mtx @ new_syndr) % 2,
            numpy.eye(20)
        )


def random_vector(n_log_qubits):
    vec = []
    norm = 0
    for x in range(2 ** n_log_qubits):
        vec += [2 * random() - 1 + 1.j * (2 * random() - 1)]
        norm += abs(vec[-1]) ** 2
    return numpy.array(vec) / sqrt(norm)


def test_logical_state_simulation():
    for z in range(6):
        edges, _ = rando_graph(20, 50)
        ops = [QubitOperator((
                    (x, 'X'),
                    *[(y, 'Z') for y in edges[x]])) for x in range(20)]
        stabs = ops[: 11]
        logs = ops[11:]
        rando_angles = [4 * random() - 2 for x in range(11)]
        rando_syndr = numpy.array(choices([0, 1], k=11))
        rando_state = random_vector(9)
        shot_outputs = _logical_state_simulation_shot(
            stabs,
            20,
            rando_state,
            rando_syndr,
            (['rm', ] * 11) + [x for x in range(9)],
            [x for x in range(11)],
            'X' * 11,
            'Y' * 11,
            [[QubitOperator(()), ], [1., ]],
            [[QubitOperator(()), ], [1., ]],
            [[
                'rot',
                ((x, 'X'), *[(y, 'Z') for y in edges[x]]),
                rando_angles[x]] for x in range(11)]
        )
        numpy.testing.assert_allclose(
            rando_state * numpy.prod([e ** (
                1.j * rando_angles[x] *
                (-1.) ** rando_syndr[x]) for x in range(11)]),
            shot_outputs[0])
        numpy.testing.assert_allclose(
            rando_syndr,
            shot_outputs[1])
        mtx = rando_trimatrix(20)
        preps = [QubitOperator(()), ] * 20
        for x in range(20):
            for y in range(20):
                if mtx[x, y] == 1:
                    preps[x] = preps[x] * ops[y]
        house_numbers = [tuple(choices([0, 1], k=11)), tuple([0, ] * 11)]
        sim_outputs = logical_state_simulation(
            stabs,
            logs,
            20,
            1,
            [QubitOperator(
                ((x, 'X'), *[(y, 'Z') for y in edges[x]]),
                rando_angles[x]) for x in range(11)],
            [[QubitOperator(()), ], [1., ]],
            [[QubitOperator(()), ], [1., ]],
            d_matrix_blocks='custom',
            block_numbers=house_numbers
        )
        vec = numpy.zeros(512)
        vec[0] = 1.
        red_ops = [QubitOperator(((x, 'X'))) for x in range(9)]
        for x in range(9):
            for y in edges[x + 11]:
                if y > 10:
                    red_ops[x] = red_ops[x] * QubitOperator(((y - 11, 'Z')))
        for x in range(9):
            vec = get_sparse_operator(
                QubitOperator(()) +
                red_ops[x],
                9
            ).dot(vec)
        vec = vec * numpy.prod([
            e ** (1.j * rando_angles[x]) for x in range(11)]) / sqrt(512)
        numpy.testing.assert_allclose(
            sim_outputs[2],
            vec
        )
        dmatrix = numpy.kron(
            vec,
            vec.conj()).reshape([512, 512])
        for x in house_numbers:
            assert x in list(sim_outputs[0].keys())
        numpy.testing.assert_allclose(
            dmatrix,
            sim_outputs[0][tuple([0] * 11)]
        )
        assert sim_outputs[1] == 1
        assert sim_outputs[4] == (
            stabs,
            ['rm', ] * 11 + [x for x in range(9)],
            [x for x in range(11)],
            'X' * 11,
            'Y' * 11
        )


def test_simulator_checks():
    bad_stabs = [
        [QubitOperator('Z0'), QubitOperator(()), QubitOperator('Z2')],
        [QubitOperator('Z0'), QubitOperator('Z1'), QubitOperator('X1')],
        [QubitOperator('Z0'), QubitOperator('Z0 Z2'), QubitOperator('Z2')],
        [QubitOperator('Z0'), QubitOperator('Z1'), QubitOperator('Z2', 1j)],
        [QubitOperator('Z0'), QubitOperator('Z1'), QubitOperator('Z9')],
    ]
    stabs = [
        QubitOperator('Z0'), QubitOperator('Z1'), QubitOperator('Z2')
    ]
    logs = [
        QubitOperator('Z3'), QubitOperator('Z4'), QubitOperator('Z5')
    ]
    circ = [QubitOperator('Z0', .4), QubitOperator('Z1', -.4)]
    all_ops = [
        *[(x, logs, [], circ) for x in bad_stabs],
        *[(logs, x, [], circ) for x in bad_stabs],
        (stabs, stabs, [], circ),
        (stabs, logs, [*stabs, *stabs], circ),
        (stabs, [*logs, QubitOperator('Z6')], [], circ),
        (stabs, logs, [], [
            QubitOperator('X3 Z4 Z5 X6', .4),
            QubitOperator('Z4 Z5', -.6)]),
        (stabs, logs, [], [
            QubitOperator('X3 X4', .4),
            QubitOperator('Z4 Z5', -.6j)]),
        (stabs, logs, [], [
            QubitOperator('X1 X4', .4),
            QubitOperator('Z4 Z5', -.6j)]),
        (stabs, logs, [], [
            QubitOperator((), .4),
            QubitOperator('Z4 Z5', -.6j)]),
    ]
    ops_w_wrong_types = [
        (stabs[: -1] + ['Z2', ], logs, [], circ),
        (stabs, logs[: -1] + [(5, 'Z'), ], [], circ),
        (stabs, logs,
            stabs + logs[: -1] + [5.0, ], circ),
        (stabs, logs, [], circ + [{((2, 'Z'), ): .567}])
    ]
    for x in all_ops:
        with pytest.raises(SimulatorError):
            outputs = logical_state_simulation(
                stabilizers=x[0],
                logical_operators=x[1],
                n_phys_qubits=6,
                rounds=1,
                phys_circuit=x[3],
                stat_noise=[[QubitOperator(())], [1., ]],
                gate_noise=[[QubitOperator(())], [1., ]],
                state_prep=x[2])
    for x in ops_w_wrong_types:
        with pytest.raises(TypeError):
            outputs = logical_state_simulation(
                stabilizers=x[0],
                logical_operators=x[1],
                n_phys_qubits=6,
                rounds=1,
                phys_circuit=x[3],
                stat_noise=[[QubitOperator(())], [1., ]],
                gate_noise=[[QubitOperator(())], [1., ]],
                state_prep=x[2])
    with pytest.raises(SimulatorError):
            outputs = logical_state_simulation(
                stabilizers=stabs,
                logical_operators=logs,
                n_phys_qubits=6,
                rounds=1,
                phys_circuit=circ,
                stat_noise=[[QubitOperator(())], [1., ]],
                gate_noise=[[QubitOperator(())], [1., ]],
                d_matrix_blocks='custom',
                block_numbers=[[0, 0, 0, 0]]
                )
    with pytest.raises(SimulatorError):
            outputs = logical_state_simulation(
                stabilizers=stabs,
                logical_operators=logs,
                n_phys_qubits=6,
                rounds=1,
                phys_circuit=circ,
                stat_noise=[[QubitOperator(())], [1., ]],
                gate_noise=[[QubitOperator(())], [1., ]],
                d_matrix_blocks='custom',
                block_numbers=[[0.01, 0, 0]]
                )



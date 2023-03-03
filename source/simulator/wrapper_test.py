from .noise import single_qubit_depolarizing
from .wrapper import postselect, pylove_simulation, tr
from .noise import single_qubit_depolarizing, two_qubit_depolarizing
from .logical_vector_construction import SimulatorError
from openfermion import QubitOperator
import numpy
import pytest
from random import choices, random


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


def test_tr():
    h_dict, _ = rando_graph(16, 40)
    signs = choices([-1, 1], k=16)
    g_x = [QubitOperator((
        (x, 'X'),
        *[(y, 'Z') for y in h_dict[x]])) for x in range(16)]
    g_y = [QubitOperator((
        (x, 'Y'),
        *[(y, 'Z') for y in h_dict[x]])) for x in range(16)]
    extra_string = [1,] + choices([0, 1], k=9)
    op = sum(g_x[10:])
    circuit=[
            QubitOperator('Z10 Z15', random()),
            QubitOperator('Z11 Z15', -random()),
            QubitOperator('Z12 Z15', random()),
            QubitOperator('Z13 Z15', -random()),
            QubitOperator('Z14 Z15', random()),]
    tstate_1_custom = pylove_simulation(
        stabilizers=[signs[x] * g_x[x] for x in range(10)],
        logical_operators=[signs[x] * g_y[x] for x in range(10, 16)],
        quantum_circuit=circuit,
        shots=25,
        mode='custom',
        block_numbers=[
            [0,] * 10,
            extra_string])

    assert  numpy.trace(
        tstate_1_custom.state[tuple(extra_string)]) == pytest.approx(0.)
    assert tr(tstate_1_custom, tstate_1_custom.ideal()) == pytest.approx(1.)
    assert tr(tstate_1_custom, op) == pytest.approx(tr(tstate_1_custom.ideal(), op))
    assert tr(
        tstate_1_custom,
        g_x[4]) == pytest.approx(signs[4])

    tstate_2_all = pylove_simulation(
        stabilizers=[signs[x] * g_x[x] for x in range(10)],
        logical_operators=[signs[x] * g_y[x] for x in range(10, 16)],
        quantum_circuit=circuit,
        shots=100,
        mode='all',
        wire_noise=single_qubit_depolarizing(.005),
        gate_noise=two_qubit_depolarizing(.005))

    assert tr(
        tstate_2_all.ideal(),
        tstate_1_custom.ideal()) == pytest.approx(1.)
    fid = tr(tstate_2_all.ideal(), tstate_2_all)
    expct2 = tr(tstate_2_all, op)
    expct1 = tr(tstate_2_all.ideal(), op)
    tstate_2_all.state[tuple([0,] * 10)] = (
        tstate_2_all.state[tuple([0,] * 10)] +
        tstate_1_custom.state[tuple([0,] * 10)])
    assert tr(
        tstate_2_all,
        tstate_1_custom.ideal()) == pytest.approx(1 + fid)
    assert tr(tstate_2_all, op) == pytest.approx(expct1 + expct2)


def test_postselect():
    h_dict, _ = rando_graph(8, 20)
    signs = choices([-1, 1], k=8)
    g_x = [QubitOperator((
        (x, 'X'),
        *[(y, 'Z') for y in h_dict[x]])) for x in range(8)]
    g_y = [QubitOperator((
        (x, 'Y'),
        *[(y, 'Z') for y in h_dict[x]])) for x in range(8)]
    circuit=[
            QubitOperator('Z4', random()),
            QubitOperator('Z5 Z4', -random()),
            QubitOperator('Z6 Z5 Z4 ', random()),
            QubitOperator('Z7 Z6 Z5', -random()),
            QubitOperator('Z7', random()),]
    
    tstate = pylove_simulation(
        stabilizers=[signs[x] * g_x[x] for x in range(4)],
        logical_operators=[signs[x] * g_y[x] for x in range(4, 8)],
        quantum_circuit=circuit,
        shots=200,
        mode='all',
        wire_noise=single_qubit_depolarizing(.005),
        gate_noise=two_qubit_depolarizing(.005))

    assert postselect(
        tstate,
        mode='code').n_entries == pytest.approx(
            200 * numpy.trace(tstate.state[(0, 0, 0, 0)]))

    
def test_errors():
    h_dict, _ = rando_graph(8, 20)
    signs = choices([-1, 1], k=8)
    g_x = [QubitOperator((
        (x, 'X'),
        *[(y, 'Z') for y in h_dict[x]])) for x in range(8)]
    g_y = [QubitOperator((
        (x, 'Y'),
        *[(y, 'Z') for y in h_dict[x]])) for x in range(8)]
    circuit=[
            QubitOperator('Z4', random()),
            QubitOperator('Z5 Z4', -random()),
            QubitOperator('Z6 Z5 Z4 ', random()),
            QubitOperator('Z7 Z6 Z5', -random()),
            QubitOperator('Z7', random()),]
    dmatrix = pylove_simulation(
            stabilizers=[signs[x] * g_x[x] for x in range(4)],
            logical_operators=[signs[x] * g_y[x] for x in range(4, 8)],
            quantum_circuit=circuit,
            shots=1,
            mode='all')

    assert tr(dmatrix) == pytest.approx(1.)
    with pytest.raises(SimulatorError):
        output = tr(dmatrix, dmatrix)
    with pytest.raises(SimulatorError):
        output = postselect(dmatrix, 'test')
    with pytest.raises(SimulatorError):
        output = tr(
            g_x[0],
            g_x[1],)
    with pytest.raises(SimulatorError):
        output = postselect(
            state=dmatrix,
            mode='custom',
            block_numbers=[(0, 0, 0, 0, 0)])
    with pytest.raises(SimulatorError):
        output = postselect(
            state=dmatrix,
            mode='custom',
            block_numbers=[(0, 0, 0, 0.5)])
    with pytest.raises(SimulatorError):
        output = postselect(
            postselect(dmatrix, 'code'),
            mode='code')
    with pytest.raises(SimulatorError):
        output = postselect(
            dmatrix.ideal(),
            mode='code')
from .logical_vector_construction import (
    SimulatorError,
    reduce_to_logical_plus)
from openfermion import (
    get_sparse_operator,
    QubitOperator)
from openfermion.config import EQ_TOLERANCE
from math import sqrt
from typing import List, Sequence, Union
import numpy
from operator import xor
from scipy.linalg import sqrtm


def is_vector(state: numpy.ndarray) -> bool:
    """
    Checks whether an array is a vector or a dense matrix.

    Args:

        state (numpy.ndarray):  Input array.

    Returns:

        (bool):

            True; the input is a state vector or
            False; the input is a density matrix.
    """
    if len(numpy.shape(state)) == 1:
        return True

    else:
        return False


def fidelity(
        state1: numpy.ndarray,
        state2: numpy.ndarray,
        ) -> float:
    """
    Returns the fidelity (as in Nielsen & Chuang) of two quantum states.

    Args:

        state1 (numpy.ndarray):
            
            Input state as state vector or density matrix.

        state2 (numpy.ndarray):
            
            Input state as state vector or density matrix, independent of
            the format of <state1>.

    Returns:

        (float):
            
            Fidelity of the input states.
    """
    check_states = (is_vector(state1), is_vector(state2))

    # the states are both vectors
    if check_states == (True, True):
        return abs(state1.dot(state2.conj()))

    # state 1 is a vector and state 2 a density matrix
    elif check_states == (True, False):
        return sqrt(abs(state1.conj() @ state2 @ state1))

    # state 1 is a density matrix and state 2 a vector
    elif check_states == (False, True):
        return sqrt(abs(state2.conj() @ state1 @ state2))

    # both states are density matrices
    else:
        return (
            abs(
                numpy.trace(
                    sqrtm(
                        sqrtm(state1) @
                        state2 @
                        sqrtm(state1)))))


def expectation_value(
        operator: QubitOperator,
        state: Union[numpy.ndarray, dict],
        stabilizer_list: Sequence,
        qubit_order: Sequence,
        fixed_positions: Sequence,
        fixed_ops: str,
        other_ops: str
        ) -> complex:
    """
    Computes the expectation value of a physical-level symbolic operator
    from a (logical-level) state from the simulator.

    Args:

        operator (QubitOperator):
            
            Symbolic operator to compute the expectation value of. The operator
            is defined on the physical level.

        state (numpy.ndarray, dict):
            
            State with respect to which the expectation value is computed.
            Supported are density matrix outputs of all modes of
            <d_matrix_blocks> of the simulator and logical-level
            state vectors.

        qubit_order (list):

            List containing integers and flags 'rm'. This list indicates which
            qubits are removed after the correction, and how the rest of the
            qubits are relabelled. After the corrections by <fix_single_term>,
            the <n>-th qubit is relabelled as <qubit_order[n]>, or removed if 
            <qubit_order[n]> = 'rm'.

        fixed_positions (list):

            List of integers, indicating which on which positions
            <fix_single_term> checks for corrections. The encounter
            of <fixed_ops[n]> or <other_ops[n]> on position
            <fixed_positions[n]> triggers the multiplication of the
            <n>-th stabilizer generator including its proper syndrome sign.

        fixed_ops (str):    

            String of characters 'X', 'Y' and 'Z'  indicating one of the
            possible Pauli types against which <fix_single_term> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

        other_ops (str):

            String of characters  'X', 'Y' and 'Z' indicating the other type
            of Pauli operators against which <fix_single_term> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

    Returns:

        (complex):
            
            Expectation value of <operator> with respect to <state>.

    Raises:

        (SimulatorError):

            Custom error for when the input format is not recognized.
    """

    n_phys_qubits = len(qubit_order)
    n_log_qubits = n_phys_qubits - len(fixed_positions)

    # <custom> mode -- we have a dictionary containing the
    #  density matrices
    if (type(state) == dict and
        numpy.shape(list(state.values())[0]) == tuple(
            [2 ** n_log_qubits] * 2)):

        expect = 0.
        for x in operator:
            op, syndr = reduce_to_logical_plus(
                x,
                stabilizer_list,
                qubit_order,
                fixed_positions,
                fixed_ops,
                other_ops)
            op, syndr = get_sparse_operator(op[0], n_log_qubits), syndr[0]

            for pattern, mtx in state.items():
                expect += numpy.trace(op.dot(mtx)) * (-1) ** (syndr @ pattern)

        return expect

    # <code> mode
    #  we only have a single density matrix
    elif (
            type(state) == numpy.ndarray and
            numpy.shape(state) == tuple([2 ** n_log_qubits] * 2)):

        expect = 0.
        for x in operator:
            expect += numpy.trace(
                get_sparse_operator(
                    reduce_to_logical_plus(
                        x,
                        stabilizer_list,
                        qubit_order,
                        fixed_positions,
                        fixed_ops,
                        other_ops)[0][0],
                    n_log_qubits).dot(state))

        return expect

    # <all> mode
    #  we have an array A shaped such that
    #  A[tuple(<some syndrome pattern>)] returns a density
    #  matrix
    elif (
            type(state) == numpy.ndarray and
            numpy.shape(state) == tuple(
                ([2] * (n_phys_qubits - n_log_qubits)) +
                ([2 ** n_log_qubits] * 2))):

        expect = 0.
        for x in operator:
            op, syndr = reduce_to_logical_plus(
                x,
                stabilizer_list,
                qubit_order,
                fixed_positions,
                fixed_ops,
                other_ops)
            op, syndr = get_sparse_operator(op[0], n_log_qubits), syndr[0]

            for y in range(2 ** (n_phys_qubits - n_log_qubits)):
                pattern = tuple(int(z) for z in bin(y)[2:].zfill(
                        n_phys_qubits -
                        n_log_qubits))

                expect += (
                    numpy.trace(op.dot(state[pattern])) *
                    (-1) ** (syndr @ pattern))

        return expect

    # <state> is a vector
    elif (
            type(state) == numpy.ndarray and
            numpy.shape(state) == (2 ** n_log_qubits,)):

        expect = 0.
        for x in operator:
            expect += state.conj() @ get_sparse_operator(
                reduce_to_logical_plus(
                    x,
                    stabilizer_list,
                    qubit_order,
                    fixed_positions,
                    fixed_ops,
                    other_ops)[0][0],
                n_log_qubits).dot(state)

        return expect
    # or some error message if the right format wasnt there
    else:
        raise SimulatorError("Input state has unknown format/shape.")


def anticom(a, b, thresh=EQ_TOLERANCE):
    """
    Check if two symbolic operators anticommute.

    Args:

        a (QubitOperator):
            
            Symbolic expression of type QubitOperator; a signed Pauli string.

        b (QubitOperator):
            
            Symbolic expression of type QubitOperator; a signed Pauli string.

        thresh (float):
            
            Threshold value for an imaginary coefficient to be deemed
            non-real. Set to Openfermion's <EQ_TOLERANCE> by default.

    Returns:

        (bool):
            
            True if <a> and <b> anticommute, otherwise False.
    """
    return abs(numpy.imag(list((a * b).terms.values())[0])) >= thresh


def check_mapping(
        A_op: dict,
        B_op: dict,
        stabs: list):
    """
    Test (anti-) commutation relations between vertex operators,
    edge operators, and stabilizer generators. Raises an error if
    one of these relations is not right.

    Args:
        A_op (dict):
            
            Dictionary with integer tuples as keys and QubitOperator-typed
            symbolic expressions signifying edge operators as values. 
    
        B_op (dict):

            Dictionary with integers as keys and QubitOperator-typed symbolic
            expressions signifying edge operators as values. 

        stabs (list):
        
            List of symbolic expressions of type QubitOperator corresponding
            to stabilizer generators.

    Returns:

        (None):

            Just interrupts the program with exceptions.

    Raises:

        (SimulatorError):

            Custom error if
                - edge operators that (do not) share a vertex (anti-)commute;
                - edge operators (anti-)commute with operators on a
                  (non-)adjacent vertex;
                - stabilizers anticommute with vertex or edge operators;
                - stabilizers anticommute with each other.

    """
    a_ok = True
    msg = ''
    other_A_op = {}
    # go through all edge operators
    for edge, edge_op in A_op.items():
        # check (anti-)commutation relations with vertex operators
        for vertex, vertex_op in B_op.items():
            if vertex in edge:
                if not anticom(vertex_op, edge_op):
                    a_ok = False
                    msg += (
                        'Edge operator along ' +
                        str(edge) +
                        ' commutes with operator on vertex ' +
                        str(vertex) + '. '
                        )
            else:
                if anticom(vertex_op, edge_op):
                    a_ok = False
                    msg += (
                        'Edge operator along ' +
                        str(edge) +
                        ' anticommutes with operator on vertex ' +
                        str(vertex) + '. '
                        )
        # check (anti-)commutation relations with other edge operators
        for other_edge, other_edge_op in other_A_op.items():
            if xor(edge[0] in other_edge, edge[1] in other_edge):
                if not anticom(edge_op, other_edge_op):
                    a_ok = False
                    msg += (
                        'Edge operator along ' +
                        str(edge) +
                        ' commutes with edge operator along ' +
                        str(other_edge) + '. '
                        )
            else:
                if anticom(edge_op, other_edge_op):
                    a_ok = False
                    msg += (
                        'Edge operator along ' +
                        str(edge) +
                        ' anticommutes with edge operator along ' +
                        str(other_edge) + '. '
                        )

        # update list of other edge operators
        other_A_op.update({edge: edge_op})

    # go through all he vertex operators in the same manner
    other_B_op = {}
    for i, x in B_op.items():
        for j, y in other_B_op.items():
            if anticom(x, y):
                a_ok = False
                msg += (
                    'Vertex operator on ' + str(i) + ' anticommutes with ' +
                    'vertex operator on ' + str(j) + '. ')
        other_B_op.update({i: x})
    
    other_stabs = {}
    # go through all the stabilizer generators
    for i, x in enumerate(stabs):
        # check commutation with edge operators
        for edge, edge_op in A_op.items():
            if anticom(edge_op, x):
                a_ok = False
                msg += (
                    'Stabilizer generator ' +
                    str(i) +
                    ' anticommutes with edge operator along ' +
                    str(edge) + '.'
                )
        # go through all vertex operators
        for vertex, vertex_op in B_op.items():
            if anticom(vertex_op, x):
                a_ok = False
                msg += (
                    'Stabilizer generator ' +
                    str(i) +
                    ' anticommutes with operator on vertex ' +
                    str(vertex) + '. '
                )
        # go through previous stabilizers
        for j, y in other_stabs.items():
            if anticom(x, y):
                a_ok = False
                msg += (
                    'Stabilizer nr. ' + str(i) + ' anticommutes with ' +
                    'stabilizer nr. ' + str(j) + '. ')
        other_stabs.update({i: x})
    if not a_ok:
        raise SimulatorError(msg)
    return None


def remap_modes(table: dict, ops: Union[list, dict]):
    """
    Re-allocate the fermionic mode labels in dictionaries filled with
    vertex and/or edge operators.

    Args:

        table (dict):

            Dictionary mapping old mode indices to new ones. The structure is
            {<old_mode_number (int)> : <new_mode_number (int)>}.

        ops (list, dict):
            
            List of dictionaries or sole dictionary containing edge and/or
            vertex operators. The conventions are
            {(<int>, <int>): <QubitOperator>} for edge operators and  
            {<int> : <QubitOperator>} for vertex operators. Edge operators
            will be ordered, such that the edges are directed from the 
            vertex with the smaller index to the vertex with the larger index,
            and edge operators receive a minus sign accordingly.
    Returns:

        (tuple):

            Tuple of new operator dictionaries, or new dictionary with
            labels exchanged as specified in <table>.

    """
    if type(ops) == dict:
        ops = (ops,)
    new_ops = ()
    for x in ops:
        new_dict = {}

        for y in list(x.keys()):
            if type(y) == int:
                new_dict.update({table[y]: x[y]})

            elif type(y) == tuple:

                # make sure the new tuples are ordered
                if table[y[0]] < table[y[1]]:
                    new_dict.update({(table[y[0]], table[y[1]]): x[y]})

                else:
                    new_dict.update({(table[y[1]], table[y[0]]): -x[y]})

        new_ops += (new_dict,)
    return new_ops

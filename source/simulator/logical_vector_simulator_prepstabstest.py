"""Logical vector simulator."""
from typing import List, Sequence, Union, Optional, Tuple

import numpy
from openfermion.ops import QubitOperator
from openfermion import get_sparse_operator

from .logical_vector_circuit import (
    make_noisy_rotation, statistical_noise_operation)
from .logical_vector_construction import (
    check_no_identity_stabilizer, check_no_imaginary_stabilizer,
    fix_single_term, reduce_to_logical)

# dictionary with Pauli matrices
PAULI = {'X': numpy.array([[0, 1], [1, 0]]),
         'Y': numpy.array([[0, -1j], [1j, 0]]),
         'Z': numpy.array([[1, 0], [0, -1]])}


def prepare_stabilizers(
        stabilizers: Union[QubitOperator, List]) -> Tuple[Sequence, str]:
    r"""
    Prepare stabilizers for simulator.

    Takes a number of stabilizers and processes them by
        - creating the list of fixed positions
        - creating two strings of Pauli operators that, when encountered at
        a fixed position a physical operator, demand its multiplication with
        the corresponding stabilizer
        - bringing the stabilizers into a form that will not demand the
        multiplication with stabilizers already considered.

    Args:
        stabilizers (QubitOperator or list):    stabilizers as a list of
                                                QubitOperators, or a sum
                                                of them
    Returns:
        (list): list of stabilizers in a more convenient from, designed
                to minimize the number of multiplications in subsequent
                tapering procedures
        (list): list of fixed positions for subsequent tapering procedures
        (str):  string of Pauli operators 'X'/'Y'/'Z' fixed for subsequent
                tapering procedures
        (str):  second list of Pauli operators 'X'/'Y'/'Z', that prompt a
                multiplication with a stabilizer in subsequent tapering
                procedures

    """
    # some type checking

    if not isinstance(stabilizers, (QubitOperator, list,
                                    tuple, numpy.ndarray)):
        raise TypeError('Input stabilizers must be QubitOperator or list.')

    stabilizer_list = list(stabilizers)

    # checking that the stabilizers are all nontrivial and only
    #     have real coefficients
    check_no_identity_stabilizer(stabilizer_list,
                                 msg='Trivial stabilizer (identity).')
    check_no_imaginary_stabilizer(stabilizer_list,
                                  msg='Stabilizer with complex coefficient.')
    final_stabs = []
    fixed_positions = []
    fixed_ops = ''
    other_ops = ''

    # We need the index of the stabilizer to connect it to the fixed qubit.

    # start with a dummy loop that selects always the first item of a
    # list we are going to shrink throughout the loop
    for _ in range(len(stabilizer_list)):
        selected_stab = list(stabilizer_list[0].terms)[0]
        final_stabs += [stabilizer_list[0]]
        # Find first position non-fixed position with non-trivial Pauli.
        for qubit_pauli in selected_stab:
            if qubit_pauli[0] not in fixed_positions:
                fixed_positions += [qubit_pauli[0]]
                fixed_ops += qubit_pauli[1]
                break

        # type checking the Pauli, if it is X or Z, the other_op is Y,
        #    and if it is Y, the other_op is X
        if fixed_ops[-1] in 'XZ':
            other_ops += 'Y'
        else:
            other_ops += 'X'

        # now start fixing the remaining stabilizers in the list by
        #    first making a new list, to which all the updated stabilizers
        #    are added.
        updated_stabilizers = []
        for update_stab in stabilizer_list[1:]:
            updated_stabilizers += [
                fix_single_term(update_stab,
                                fixed_positions[-1],
                                fixed_ops[-1],
                                other_ops[-1],
                                stabilizer_list[0])
            ]

        # the updated stabilizers now become the new list from the top
        #     of which the next stabilizer is taken in the next iteration
        #    of the loop
        stabilizer_list = updated_stabilizers[:]

        # always check if by multiplication of the stabilizers have
        #    generated complex coefficients or identities, and draw
        #    conclusions about the set of generators
        check_no_identity_stabilizer(stabilizer_list,
                                     msg='Linearly dependent stabilizers.')
        check_no_imaginary_stabilizer(stabilizer_list,
                                      msg='Stabilizers anticommute.')

    return final_stabs, fixed_positions, fixed_ops, other_ops


def prepare_initial_state(
        more_stabs: Union[List, QubitOperator]) -> numpy.ndarray:
    """
    Prepare initial state.

    Args:
        more_stabs (list or QubitOperator): sum or list of symbolic operators
                                            stabilizing the initial state.
                                            These stabilizers need to
                                            constrain the state completely.
    Returns:
        (numpy.ndarray): vector of the initial state
    """
    # quick check if the new stabilizers commute and are linearly independent
    check_no_identity_stabilizer(more_stabs,
                                 msg='Stabilizers are linearly dependent.')
    check_no_imaginary_stabilizer(more_stabs,
                                  msg='Stabilizers anticommute.')

    # use prep_stabilizers() to fix the form of the new stabs
    (fxd_stabs, fxd_pos,
     fxd_ops, _) = prepare_stabilizers(list(more_stabs))

    prod = 1
    n_qubits = len(fxd_pos)

    # now go through all the qubits and fix the states ...
    # CHANGE HERE: only fix the ones in fxd_pos
    for x in range(n_qubits):
    #for x in fxd_pos:
        
        # .. into the |0> in case the qubit operator is fixed to X or Y
        #if x in fxd_pos:
        if fxd_ops[fxd_pos.index(x)] in 'XY':
            prod = numpy.kron(prod, numpy.array([1, 0], dtype=complex))

        # .. into |+> in case the quit operator is fixed to Z
        else:
            prod = (numpy.kron(prod, numpy.array([1., 1.], dtype=complex)) /
                    numpy.sqrt(2))

    # END CHANGE: only fix the ones in fxd_pos

    # now multiply the basis states |00+0++0> with the stabilizer group
    #     operator (1 + S_1)(1 + S_2) .. (1 + S_7) / sqrt(2 ** 7)
    for x in fxd_stabs:
        prod = (get_sparse_operator(QubitOperator(()) + x, n_qubits) @
                prod) / numpy.sqrt(2)

    return prod


def _logical_state_simulation_shot(
        stabilizer_list: Sequence,
        n_phys_qubits: int,
        state_vec: Union[List, numpy.ndarray],
        qubit_order: Sequence,
        fixed_positions: Sequence,
        fixed_ops: str,
        other_ops: str,
        stat_noise: List,
        gate_noise: List,
        schedule: Sequence) -> Tuple[numpy.ndarray, List]:
    r"""
    Simulate sigle-shot of logical state.

    Auxiliary routine. Takes the scheduled events (rotations and idle),
    and
        - statistically places Kraus operators into them
        - calculates their symbolic operators
        - transforms them into logical operators and syndromes
        - turns those operators into dense arrays
        - calculates the state vector.

    Outputs the state vector and the syndrome pattern.

    Args:
        stabilizer_list (list): list of stabilizer generators (QubitOperator)
        n_phys_qubits (int):    number of physical qubits
        state_vec (list or ndarray):    initial state vector at which the
                                        logical  quantum circuit should act
        qubit_order (list): list containing integers and flags 'rm'.
                            This list indicates the reordering of the qubits.
                            When the operators are brought into a fixed form,
                            the n-th qubit is indexed with qubit_order[n], or
                            in case qubit_order[n] = 'rm', removed.
        fixed_positions (list): list of integers, indicating which positions
                                are fixed by the tapering procedure.
        fixed_ops (str):    string of Pauli operators 'X'/'Y'/'Z' on the
                            encounter of which at the fixed position, the
                            operator is multiplied with the corresponding
                            stabilizer
        other_ops (str):    same as fixed_ops
        stat_noise (list):  list holding two sub-lists of noise parameters.
                            It has the shape [[#1], [#2]]. [#1] is a
                            list of Pauli operators (QubitOperators) with
                            index 0 and the identity. The list describes
                            Kraus operators the of Pauli noise. [#2] is a list
                            of the operators' statistical weights.
        gate_noise (list):  list holding two sub-lists of noise parameters.
                            It has the shape [[#1], [#2]]. [#1] is a
                            list of Kraus operators (QubitOperators) including
                            the identity, where the qubit index 0 denotes the
                            control and the index 1 denotes the target qubit.
                            [#2] is a list of the Kraus operators' statistical
                            weight.
        schedule (list):    list containing events in the quantum circuit.
                            It consists of sub-list of two types. Type one
                            has the form ['idl', <int>, <int>], signifying a
                            qubit is idle for a certain time. Its entries are
                            the qubit index and idle period in numbers of time
                            steps. The second type of entry is of the form
                            ['rot', <str>, <float>], signifying a Pauli string
                            rotation. Its entries are the Pauli string as a
                            string of characters such that it could be input
                            into QubitOperator(.), and the rotation angle.
        Returns:
            (numpy.ndarray):    one-dimensional array containing the resulting
                                logical state vector.
                                The array entries correspond to the
                                coefficients of the computational state
                                configurations
                                [|000>, |001>, |010>, |011>, |100>, ...],
                                where the first bit in the list corresponds to
                                the logical qubit indexed 0.
            (list): one-dimensional list with binary entries 0/1,
                    signifying the syndromes

    """
    # syndromes are later passed to every function by reference
    syndromes = [0] * len(stabilizer_list)

    for x in schedule:

        # put idle noise if the schedule tells us we are in a waiting time
        if x[0] == 'idl':
            phys_op = statistical_noise_operation(x[1], stat_noise, x[2])

        # this case is when the schedule shows a rotation
        else:
            phys_op = make_noisy_rotation(*x[1:], stat_noise, gate_noise)[0]

        # whatever physical operator we have at this moment is turned
        # logical and multiplied with the current state vector.
        # While state_vec is updated explicitly, the syndromes
        # are passed by reference to reduce_to_logical(), which
        # updates them.
        state_vec = (get_sparse_operator(reduce_to_logical(
                                         phys_op,
                                         syndromes,
                                         stabilizer_list,
                                         qubit_order,
                                         fixed_positions,
                                         fixed_ops,
                                         other_ops),
                     n_phys_qubits - len(stabilizer_list)).toarray() @
                     state_vec)

    return state_vec, syndromes


def logical_state_simulation(
    stabs: Union[Sequence, QubitOperator],
    n_phys_qubits: int,
    rounds: int,
    stat_noise: List,
    gate_noise: List,
    phys_circuit: Sequence,
    state_prep: Sequence,
    Hamiltonian: QubitOperator = QubitOperator(' ')) -> Tuple[
        numpy.ndarray, QubitOperator, numpy.ndarray, Tuple]:
    r"""
    Simulate logical state.

    Simulates a physical quantum circuit under Pauli noise.
    In detail, the algorithm uses the system's stabilizer conditions to
    reconstruct the total density matrix by separating the simulation of
    the logical subspace from the evolution of its syndrome pattern.
    To that end, it
        - processes the list of stabilizers for a unique characterization
          of the logical subspaces and syndrome patterns
        - constructs the circuit from the list of rotations and applies it
          without noise to a given initial state
        - reduces a given Hamiltonian to its logical form
        - runs multiple versions of the circuit in which noise is
          statistically inserted.

    Args:
        stabs (QubitOperator or list):  stabilizer generators as a list of
                                        symbolic operators (QubitOperator),
                                        or a sum of them
        n_phys_qubits (int):    number of physical qubits
        rounds (int):   number of shots that the density matrix is
                        reconstructed with
        stat_noise (list):  list holding two sub-lists of noise parameters.
                            It has the shape [[#1], [#2]]. [#1] is a
                            list of Pauli operators (QubitOperators) with
                            index 0 and the identity. The list describes
                            Kraus operators the of Pauli noise. [#2] is a list
                            of the operators' statistical weights.
        gate_noise (list):  list holding two sub-lists of noise parameters.
                            It has the shape [[#1], [#2]]. [#1] is a
                            list of Kraus operators (QubitOperators) including
                            the identity, where the qubit index 0 denotes the
                            control and the index 1 denotes the target qubit.
                            [#2] is a list of the Kraus operators' statistical
                            weight.
        phys_circuit (list):    schedule of the circuit in its physical
                                representation. This list holds Pauli string
                                rotations and their angles. Its elements can
                                either have the form ('X0 Y1', .36),
                                (.36, 'X0 Y1') or QubitOperator('X0 Y1', .36).
        state_prep (list):  list of Pauli strings that act as additional
                            stabilizers for the creation of the initial state.
                            They are given in their physical representation as
                            symbolic operators (QubitOperator).
        Hamiltonian (QubitOperator):    Hamiltonian that is the objective
                                        function of the simulation. This
                                        parameter is optional, its default
                                        is the identity.

    Returns:
        (numpy.ndarray):    reconstructed density matrix. The shape of this
                            array has two parts ( *A, *B), where A contains a
                            number of bits A =(2, 2, 2, ... , 2). This part of
                            the array stores syndrome patterns associated with
                            the sub-matrices of the block-diagonal density
                            matrix. The shape of the sub-matrices is found in
                            B = (2**n, 2**n), where n is the number of logical
                            qubits.
                            This entire array 'mtx' is constructed such that
                            a tuple of bits b = (0,1,1,0,1), can be
                            used to prompt the sub-block associated with the
                            syndrome pattern '01101' by calling mtx[b].
        (QubitOperator):    logical representation of the Hamiltonian
        (numpy.ndarray):    initial state in dense form
        (tuple):            collection of technical data. The elements of
                            this tuple are <depth>: the algorithmic depth
                            of the simulation, <upd_stabs>: the list of
                            stabilizer generators in ther fixed form,
                            <fixed_positions>: list of integers signifying
                            which qubits have been fixed, <fixed_ops>: a
                            string indicating to which Pauli operator the
                            stabilizers have been fixed.

    """
    # Fix the stabilizers
    (upd_stabs, fixed_positions,
     fixed_ops, other_ops) = prepare_stabilizers(stabs)
    # Prepare the qubit_order list such that you could put it into
    #    reduce_to_logical()
    n_log_qubits = n_phys_qubits - len(fixed_positions)
    qubit_order = list(range(n_log_qubits))
    removed_positions = list(fixed_positions)
    removed_positions.sort()
    for x in removed_positions:
        qubit_order.insert(x, 'rm')

    log_state_prep_ops = []
    # turn the operators that define the initial state into their logical
    #    representation

    for x in state_prep:
        log_state_prep_ops += [reduce_to_logical(x,
                                                 [0] * len(stabs),
                                                 upd_stabs,
                                                 qubit_order,
                                                 fixed_positions,
                                                 fixed_ops,
                                                 other_ops)]
    # builidng the initial logical state

    initial_state = prepare_initial_state(log_state_prep_ops)

    # cry for help if the initial state has a different dimension
    if len(initial_state) != 2 ** n_log_qubits:
        raise StabilizerError('System over- or underconstrained.' +
                              ' logical state dimension expected to be ' +
                              str(2**n_log_qubits) +
                              ', but is ' +
                              str(len(initial_state)))

    # start preparing the qubit schedule, that tells at what time
    #    step a qubit was operated on last
    qubit_schedule = [0] * n_phys_qubits
    # queue of logical operations in the noiseless case
    log_circuit = []
    queue = []  # collection of idle noise and rotation gates
    # queue is the input 'schedule' in _logical_state_simulation_shot()

    # go through the rotations of phys_circuit to create a schedule
    #    allowing for noise. This is only gonna happen once in the simulation,
    #    setting a schedule for the shots.
    for x in phys_circuit:
        pauli_string = ''
        angle = 0

    # parse for 3 possible notations of the input Pauli string

        # as QubitOperator: QubitOperator('X0 Y1', .36)
        if type(x) is QubitOperator:
            angle = list(x.terms.values())[0]
            for y in list(x.terms)[0]:
                pauli_string += y[1] + str(y[0]) + ' '

        # as tuple or list: ('X0 Y1', .36)
        elif type(x[0]) is str:
            pauli_string = x[0]
            angle = x[1]

        # as tuple or list: (.36, 'X0 Y1')
        elif type(x[1]) is str:
            pauli_string = x[1]
            angle = x[0]

        # determine which qubits are used in the rotation
        busy_qubits = [int(y[1:]) for y in pauli_string.split()]

        # determine how much time has to pass until every qubit
        #   in the rotation is freed up
        max_time = qubit_schedule[busy_qubits[0]]

        for z in busy_qubits[1:]:
            max_time = max(max_time, qubit_schedule[z])

        # fill the schedule of every busy qubit with noise until
        #    the rotation commences
        for z in busy_qubits:
            if qubit_schedule[z] != max_time:
                queue += [['idl', z, max_time - qubit_schedule[z]]]
                qubit_schedule[z] = max_time

        queue += [['rot', pauli_string, angle]]

        # running a rotation circuit with noise turned down, to
        #    obtain the rotation operator and the depth in one go
        op, depth = make_noisy_rotation(*queue[-1][1:],
                                        [[QubitOperator(())], [1, ]],
                                        [[QubitOperator(())], [1, ]])

        # turn the rotation operator into its logical representation
        new_op = reduce_to_logical(op,
                                   [0] * len(stabs),
                                   upd_stabs,
                                   qubit_order,
                                   fixed_positions,
                                   fixed_ops,
                                   other_ops)

        # update the qubit timer, algorithmic depth and logical circuit
        for z in busy_qubits:
            qubit_schedule[z] += depth

        log_circuit += [new_op]

    depth = max(qubit_schedule)

    # applying the noiseless quantum circuit to the initial state
    ideal_state = numpy.array(initial_state)  # copy the initial state

    # make all rotations in the noiseless schedule dense and transform
    #    the state vector with it
    for x in log_circuit:
        ideal_state = (get_sparse_operator(x, n_log_qubits).toarray() @
                       ideal_state)

    # The array 'mtx' reconstructs the density matrix. It has two parts:
    #    1)    a part that holds the syndrome patterns as bit string vectors
    #    2)     a part that holds the density marix of the logical subspace
    #        associated with the syndrome pattern.
    # When the first part is converted to a tuple eg. b = (0,1,1,0,1)
    # then mtx[b] outputs the denisty matrix block of the (0,1,1,0,1)

    mtx = numpy.zeros([2] * len(fixed_positions) +
                      2 * [2 ** n_log_qubits], dtype=complex)

    # begin of the main sequence
    for x in range(rounds):

        # one of the shots
        # initial_state is always passed as a copy
        vec, syndr = _logical_state_simulation_shot(
            upd_stabs,
            n_phys_qubits,
            numpy.array(initial_state),
            qubit_order,
            fixed_positions,
            fixed_ops,
            other_ops,
            stat_noise,
            gate_noise,
            queue)

        # add the results of the shot to the density matrix
        mtx[tuple(syndr)] += numpy.kron(vec,
                                        vec.conj()).reshape([2 **
                                                             n_log_qubits] * 2)

    # output the results
    return (mtx / rounds,
            reduce_to_logical(Hamiltonian,
                              [0] * len(upd_stabs),
                              upd_stabs,
                              qubit_order,
                              fixed_positions,
                              fixed_ops,
                              other_ops),
            ideal_state,
            (depth,
             upd_stabs,
             qubit_order,
             fixed_positions,
             fixed_ops))


# example
"""
logical_state_simulation(
    [QubitOperator('Z4 Z5  ',-1), QubitOperator('Z6 Z7 Z8')],
    9,
    100,
    [[QubitOperator(()), QubitOperator(()), ],[1, 1]],
    [[QubitOperator(()), QubitOperator(()), ],[1, 1]],
    (
     ('Z0', .147),
     ('Z0 Z1', .147),
     ('Z0 Z1 Z2', .147),
     ('Z0 Z1 Z2 Z3', .147),
     ('Z0 Z1 Z2 Z3 Z4', .147),
     ('Z0', .147),
     ('Z0 Z1', .147),
     ('Z0 Z1 Z2', .147),
     ('Z0 Z1 Z2 Z3', .147),
     ('Z0 Z1 Z2 Z3 Z4', .147),
     ('Z0', .147),
     ('Z0 Z1', .147),
     ('Z0 Z1 Z2', .147),
     ('Z0 Z1 Z2 Z3', .147),
     ('Z0 Z1 Z2 Z3 Z4', .147),
     ),
    [QubitOperator('Z0'),
    QubitOperator('Z1'),
    QubitOperator('Z2'),
    QubitOperator('Z3'),
    QubitOperator('Z4'),
    QubitOperator('Z6'),
    QubitOperator('Z7')]
    )

print(time.time() - start, 's')
"""

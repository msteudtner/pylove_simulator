"""Logical vector simulator."""
from typing import List, Sequence, Union, Optional, Tuple
import numpy
from openfermion import (
    QubitOperator,
    get_sparse_operator,
    count_qubits)
from openfermion.config import EQ_TOLERANCE
from .logical_vector_circuit import (
    make_noisy_rotation,
    noisy_state_prep,
    statistical_noise_operation)
from .logical_vector_construction import (
    SimulatorError,
    check_for_identity,
    check_for_imaginary_coeff,
    fix_single_term,
    reduce_to_logical,
    reduce_to_logical_plus,
    anticom_syndr,
    generator_dependence)
from math import sin, cos, sqrt, ceil
from .noise import noiseless
from multiprocessing import cpu_count
import pypeln


def prepare_stabilizers(
        stabilizers: Union[QubitOperator, List]
        ) -> Tuple[Sequence, str]:
    r"""
    Prepares input parameters for <reduce_to_logical> from a list of
    stabilizer generators by (i) allocating the qubits to be removed,
    (ii) assigning the Pauli operators that <fix_single_term> checks
    against and (iii) updating the list of generators with
    <fix_single_term> themselves, allowing for a linear runtime of
    <reduce_to_logical>.

    Args:

        stabilizers (QubitOperator or list):

            List of symbolic expressions of type QubitOperator
            (or a QubitOperator-typed sum of signed Pauli strings)
            signifying stabilizer generators.

    Returns:

        stabilizer_list (list):

            List of signed Pauli strings of type QubitOperator; the updated
            list of stabilizer generators, obtained in an iterative procedure
            where the zeroth stabilizer is not updated at all and the
            <n>-th stabilizer is updated by running <fix_single_term>
            <n> times with the preceding stabilizers in this list.

        fixed_positions (list):

            List of integers, indicating on which positions
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

            String of characters 'X', 'Y' and 'Z' indicating the other type
            of Pauli operators against which <fix_single_term> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

    Raises:

            (SimulatorError):

                Custom exception when items in <stabilizers> are linearly
                dependent.

    """
    stabilizer_list = list(stabilizers)
    fixed_positions = []
    fixed_ops = ''
    other_ops = ''

    # We need the index of the stabilizer to connect it to the fixed qubit.

    # start with a dummy loop that selects always the first item of a
    # list we are going to shrink throughout the loop
    for x in range(len(stabilizer_list)):
        selected_stab = list(stabilizer_list[x].terms)[0]

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

        # now fix the remaining stabilizers in the list
        for y in range(x + 1, len(stabilizer_list)):
            stabilizer_list[y] = fix_single_term(
                stabilizer_list[y],
                fixed_positions[-1],
                fixed_ops[-1],
                other_ops[-1],
                stabilizer_list[x])

        # always check if by multiplication of the stabilizers have
        #   generated identities, because then they are linearly dependent

        check_for_identity(
            stabilizer_list,
            'Input Pauli strings are linearly dependent.'
            )

    return stabilizer_list, fixed_positions, fixed_ops, other_ops


def make_simulation_schedule(
        ops: Sequence,
        n_phys_qubits: int,
        mode: str) -> Tuple[list, int]:
    """
    Makes a schedule of physical processes simulated during the
    time evolution / ansatz or state preparation circuit. A list is
    created whose items denote events such idle periods on quantum wires,
    Pauli string rotations or projective measurement subcircuits.
    The schedule essentially takes a list of rotations or measurements
    (depending on the value of <mode>) and computes how their respective
    subroutines would be embraced by idle periods in a quantum circuit.

        Args:

            ops (list):

                List of symbolic expressions of type QubitOperator that are
                interpreted in one of two ways depending on the value of
                <mode>.

                If <mode> == 'rot', the items are of the form
                QubitOperator(<pstring>: str, <angle>: float) signifying
                Pauli string rotations exp(1j <angle> <pstring>).

                If <mode> == 'meas', the items are of the form
                QubitOperator(<pstring>: str, +1/-1) signifying projective
                measurements of the Pauli strings +/- <pstring>.

            n_phys_qubits (int):

                Number of physical qubits in the entire quantum circuit.

            mode (str):

                One of two parameters 'rot' or 'meas' determining if the
                schedule is made for a state preparation or time
                evolution / ansatz circuit.

        Returns:

            (list):

                List containing two types of elements:

                (i) Lists of the form ['idl', <n>: int, <m>: int],
                indicating that the <n>-th qubit is idle for
                <m> time steps.

                (ii) Lists of the form
                ['rot', <pstring>: tuple, <angle>: float] in case
                <mode> == 'rot', describing the rotation of the Pauli string
                QubitOperator(<pstring>) about the angle <angle>. Or lists
                ['meas', <pstring>: tuple, <n>: int] describing a
                measurement circuit of the <n>-th Pauli string in <ops>,
                QubitOperator(<pstring>), in case <mode> == 'meas'.

            (int):

                Total depth of the scheduled quantum circuit.
    """
    # make a list that keeps track of the time index of each qubit
    qubit_time_index = [0] * n_phys_qubits

    # schedule of simulated processes
    schedule = []

    if ops:
        for i, x in enumerate(ops):
            busy_qubits = []

            # input is of the form QubitOperator('X0 Y1', -1.2)
            for y in list(x.terms)[0]:
                busy_qubits += [y[0]]

            # determine how much time has to pass until every qubit
            #   in the rotation is freed up
            max_time = qubit_time_index[busy_qubits[0]]

            for z in busy_qubits[1:]:
                max_time = max(max_time, qubit_time_index[z])

            # fill the schedule of every busy qubit with noise until
            #  they are at the same time index
            for z in busy_qubits:
                if qubit_time_index[z] != max_time:
                    schedule += [['idl', z, max_time - qubit_time_index[z]]]
                    qubit_time_index[z] = max_time

            # make the circuit: distinguish if it is a state preparation
            #  or rotation schedule
            if mode == 'meas':
                schedule += [['meas', list(x.terms)[0], i]]

            else:
                schedule += [['rot', *list(x.terms.items())[0]]]

            # running a rotation circuit with noise turned down, to
            #    obtain the rotation operator and the depth in one go
            _, depth = make_noisy_rotation(
                list(x.terms)[0],
                0,
                noiseless(),
                noiseless())

            # update the qubit timers
            for z in busy_qubits:
                qubit_time_index[z] += depth

        depth = max(qubit_time_index)

        # fill the schedule up with idle noise to the end
        for z in busy_qubits:
            if qubit_time_index[z] != depth:
                schedule += [['idl', z, depth - qubit_time_index[z]]]
                qubit_time_index[z] = depth

        return schedule, depth
    else:
        return [], 0


def simulate_state_prep(
        schedule: list,
        prep_stab_ops: list,
        stat_noise: Sequence,
        gate_noise: Sequence) -> numpy.ndarray:
    r"""
    Simulating the initial state from the beginning of the state preparation
    until the end. Assuming that the initial state is a fully-constrained
    stabilizer state, meaning it is an <n>-qubit state with <n> stabilizer
    generators, errors are recorded as <n>-bit syndromes. While measurements
    constrain the system,  we do not care in which specific subspace they
    project, as syndromes can be accounted for in the ansatz / time evolution
    circuit. Projective measurements are therefore modelled as bit resets.
    Starting from an all-zero bit string, the state preparation is simulated
    by bit flips in between bit resets.

    Args:

        schedule (list):

            List containing events of the state preparation circuit including:

            (i) Lists of the form ['idl', <n>: int, <m>: int], indicating
            that the <n>-th qubit is idle for <m> time steps.

            (ii) Lists of the form ['meas', <pstring>: tuple, <n>: int]
            describing a measurement circuit of the <n>-th Pauli string
            in <prep_stab_ops>, QubitOperator(<pstring>).

        prep_stab_ops (list):

            List of symbolic expressions of type QubitOperator signifying
            an extended list of stabilizers fully constraining the quantum
            state. The number of generators is equal to the number of physical
            qubits.

        stat_noise (list):

            List characterizing the single-qubit noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli operators (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli operators have the index 0 as a placeholder
            for the proper label of the qubit the noise acts on.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli operators.

        gate_noise (list):

            List characterizing the gate noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli strings (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli strings itself are on two qubits, where
            the integers 0 and 1 function as placeholders for the labels of
            control and target qubit, respectively.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli string.

    Returns:

        (numpy.ndarray):

            Array of integers 0 and 1 corresponding to syndromes of the
            extended stabilizer list <prep_stab_ops>.
    """
    syndr = numpy.zeros(len(prep_stab_ops), dtype=int)
    for x in schedule:

        # place statistical noise on the idle qubits and check if
        #  it anticommutes with the preparation stabilizers
        if x[0] == 'idl':
            op = statistical_noise_operation(x[1], stat_noise, x[2])

            if op != QubitOperator(()):
                syndr += anticom_syndr(op, prep_stab_ops)
                syndr = syndr % 2

        # if we are in a measurement circuit, get the error operators before
        #  and after the measurement from the routine, compute first the
        #  syndromes from the errors prior to the projection, then reset the
        #  measured stabilizer and go on with the error after the projection

        elif x[0] == 'meas':
            left_op, right_op, _ = noisy_state_prep(
                x[1],
                stat_noise,
                gate_noise)

            if right_op != QubitOperator(()):
                syndr += anticom_syndr(
                    right_op,
                    prep_stab_ops)
            syndr[x[2]] = 0

            if left_op != QubitOperator(()):
                syndr += anticom_syndr(
                    left_op,
                    prep_stab_ops)

            syndr = syndr % 2

    return syndr


def make_initial_state(
        rudimentary_state: numpy.ndarray,
        logical_operators: list,
        syndromes: numpy.ndarray) -> numpy.ndarray:
    r"""
    From a syndrome pattern and list of logical operators this routine returns
    (a snapshot of) a noisy initial state: a dense vector representing the
    logical state and a syndrome pattern for the rest of the qubits.

    Args:

        rudimentary_state (numpy.ndarray):

            Rudimentary quantum state |R> that can be projected into a
            stabilizer state of the signed Pauli string <S>,

                            (1 + S) |R> / sqrt(2),

            for all <S> in <logical_operators>.

        logical_operators (list):

            List of symbolic expressions of type QubitOperator signifying the
            logical operators extending the stabilizer list, in their logical
            representation.

        syndromes (numpy.ndarray):

            Array of integers 1 and 0 signifying syndromes of the entire list
            of extended stabilizers, the last of which
            are <logical_operators>.

    Returns:

        (numpy.ndarray):

            Logical state vector.

        (numpy.ndarray):

            Array of integers 0 and 1 signifying syndromes of the original
            stabilizers; the first entries of <syndromes> not containing
            syndromes for logical operators.
    """
    n_log_qubits = len(logical_operators)
    n_rem_qubits = len(syndromes) - n_log_qubits
    op = numpy.array(rudimentary_state)

    for x in range(n_log_qubits):
        op += ((-1)**syndromes[x + n_rem_qubits]) * get_sparse_operator(
            logical_operators[x],
            n_log_qubits).dot(op)
        op = op / sqrt(2)
    return op, syndromes[:n_rem_qubits]


def invert_tri_matrix(mtx: numpy.ndarray) -> numpy.ndarray:
    r"""
    Inverts a lower triangular binary matrix.

    Args:

        mtx (numpy.ndarray):

            Two-dimensional array of integers 0 and 1. A lower triangular
            matrix with binary inputs to be inverted.

    Returns:

        (numpy.ndarray):

            Two-dimensional array of integers 0 and 1. Inverse of <mtx>.
    """

    dim = numpy.shape(mtx)[0]
    inv_mtx = numpy.eye(dim, dtype=int)

    for x in range(dim):
        for y in range(x + 1, dim):
            inv_mtx[y, x] = (mtx[y, x:y] @ inv_mtx[x:y, x]) % 2

    return inv_mtx


def prepare_state_prep(
    state_prep_ops: list,
    log_ops: list,
    stabs: list,
    qubit_order: list,
    fixed_positions: list,
    fixed_ops: str,
    other_ops: str
        ) -> Tuple[numpy.ndarray, List, List]:

    r"""
    Given the extended list of stabilizer generators, this function computes
    ingredients for an initial logical state (inputs of <make_initial_state>)
    and a stabilizer propagation matrix. The Pauli strings measured during the
    state preparation are generally different from the stabilizers updated by
    <prepare_stabilizers>, and the logical operators fixed by applications of
    <fix_single_term>. The syndrome propagation matrix is used to compute
    syndromes of the latter from syndromes in the state preparation procedure.
    We compute the matrix M that relates the state preparation syndromes
    s to the syndromes of the fixed system t such that t = M s:

    A new set of Pauli strings is obtained by updating the state preparation
    operators using <prepare_stabilizers>, and relate their syndromes s' to
    s using the routine <generator_dependence> to compute a matrix A such
    that s = A s'. Since A is triangular, it can be readily inverted.
    The routine <generator_dependence> is used again to obtain a matrix B
    such that t = B s'. We find

                            t = B s' = B A⁻¹ s,

    and so we compute the syndrome propagation matrix by M = B A⁻¹.

    Args:

        state_prep_ops (list):

            List of symbolic expressions of type QubitOperator, signifying
            the Pauli strings measured during the state preparation routine.

        log_ops (list):

            List of symbolic expressions of type QubitOperator, signifying the
            logical operators stabilizing the initial state, given in their
            physical representation.

        stabs (list):

            List of signed Pauli strings of type QubitOperator; the updated
            list of stabilizer generators, obtained in an iterative procedure
            where the zeroth stabilizer is not updated at all and the
            <n>-th stabilizer is updated by running <fix_single_term>
            <n> times with the preceding stabilizers in this list.

        qubit_order (list):

            List containing integers and flags 'rm'. This list indicates which
            qubits are removed after the correction, and how the rest of the
            qubits are relabeled. After the corrections by <fix_single_term>,
            the <n>-th qubit is relabeled as <qubit_order[n]>, or removed if
            <qubit_order[n]> = 'rm'.

        fixed_positions (list):

            List of integers, indicating on which positions
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

            String of characters 'X', 'Y' and 'Z' indicating the other type
            of Pauli operators against which <fix_single_term> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

    Returns:

            (numpy.ndarray):

                Rudimentary quantum state |R> that can be projected into a
                stabilizer state of the signed Pauli string <S>,

                            (1 + S) |R> / sqrt(2),

                for all <S> being logical representations of the Pauli strings
                in <log_ops> with respect to the updated stabilizers <stabs>.

            (numpy.ndarray):

                Quadratic array of integers 0 and 1; the syndrome
                propagation matrix from <state_prep_ops> to the system fixed
                by <stabs> and <log_ops>.

            (list):

                List of QubitOperator-typed symbolic expressions corresponding
                to the logical representation of the operators in <log_ops>.

    """

    n_phys_qubits = len(state_prep_ops)
    n_log_qubits = n_phys_qubits - len(fixed_ops)

    # we need to obtain a taperable set of generators from the
    #  state preparation operators (which are untaperable generators)
    (
        tp_stabs,
        tp_fixed_positions,
        tp_fixed_ops,
        tp_other_ops) = prepare_stabilizers(state_prep_ops)

    # create the syndrome propagation matrix,
    #  turning syndromes of the taperable generators
    #  into syndromes of the untaperable generators
    tp_mtx = numpy.eye(n_phys_qubits)

    for x in range(n_phys_qubits):
        _, tp_mtx[x, :] = generator_dependence(
            state_prep_ops[x],
            tp_stabs,
            tp_fixed_positions,
            tp_fixed_ops,
            tp_other_ops)

    # fix the logical operators with the stabilizers
    for x in range(n_log_qubits):
        for y in range(n_phys_qubits - n_log_qubits):

            log_ops[x] = fix_single_term(
                log_ops[x],
                fixed_positions[y],
                fixed_ops[y],
                other_ops[y],
                stabs[y])

    # extend the system by the fixed logical operators
    add_fixed_positions = []
    add_fixed_ops = ''
    add_other_ops = ''

    for x in range(n_log_qubits):
        selected_pstring = list(log_ops[x].terms)[0]

        # Find first position non-fixed position with non-trivial Pauli.
        for qubit_pauli in selected_pstring:
            if (qubit_pauli[0] not in add_fixed_positions and
                    qubit_pauli[0] not in fixed_positions):

                add_fixed_positions += [qubit_pauli[0]]
                add_fixed_ops += qubit_pauli[1]
                break

        # type checking the Pauli, if it is X or Z, the other_op is Y,
        #    and if it is Y, the other_op is X
        if add_fixed_ops[-1] in 'XZ':
            add_other_ops += 'Y'
        else:
            add_other_ops += 'X'

        # now start fixing the remaining operators in the list
        #  to the new operator
        for y in range(x + 1, n_log_qubits):

            log_ops[y] = fix_single_term(
                log_ops[y],
                add_fixed_positions[-1],
                add_fixed_ops[-1],
                add_other_ops[-1],
                log_ops[x])

    # compute the syndrome propagation matrix between the taperable
    #  measurement operators and extended (fixed) stabilizers
    prop_mtx = numpy.eye(n_phys_qubits, dtype=int)
    ext_stabs = stabs + log_ops
    check_ops = [0] * n_phys_qubits  # dummy list to check that state prep
    # ops and extended stabilizers span the same stabilizer state

    for x in range(n_phys_qubits):
        check_ops[x], prop_mtx[x, :] = generator_dependence(
            ext_stabs[x],
            tp_stabs,
            tp_fixed_positions,
            tp_fixed_ops,
            tp_other_ops)

    # reduce the logical operators down to the logical level
    redux_ops, _ = reduce_to_logical_plus(
        log_ops,
        stabs,
        qubit_order,
        fixed_positions,
        fixed_ops,
        other_ops)

    # compute the fixed positions of the logical operators
    #  on the logical level
    redux_fixed_positions = [qubit_order[x] for x in add_fixed_positions]

    # create the state |R>, such that (1 + S_1) ... (1 + S_n) |R>
    #  is stabilized by {S_1, ... , S_n}, which are the reduced
    #  logical operators.
    prod = 1
    for x in range(n_log_qubits):

        if add_fixed_ops[redux_fixed_positions.index(x)] in 'XY':
            prod = numpy.kron(prod, numpy.array([1, 0], dtype=complex))

        else:
            prod = (numpy.kron(prod, numpy.array([1., 1.], dtype=complex)) /
                    numpy.sqrt(2))

    # here's |R> and the matrix that propagates syndromes from the untaperable
    #  measurement operators to the fixed and taperable extended stabilizers
    return prod, (prop_mtx @ invert_tri_matrix(tp_mtx)) % 2, redux_ops


def propagate_syndromes(
        original_syndr: Sequence,
        original_stabs: Sequence,
        new_stabs: Sequence,
        fixed_positions: Sequence,
        fixed_ops: str,
        other_ops: str) -> list:
    r"""
    Given a collection of syndrome patterns, compute their equivalents with
    respect to a different set of stabilizer generators.

    Args:

        old_syndr (list):

            List of sequences filled with integers 0 and 1. The sequences in
            the list are syndromes with respect to <original_stabs>, which
            need to be transformed into a list of syndromes corresponding to
            <new_stabs>.

        original_stabs (list):

            List of symbolic expressions of type QubitOperator; the stabilizer
            generators to be replaced with <new_stabs>.

        new_stabs (list):

            List of signed Pauli strings of type QubitOperator; the updated
            list of stabilizer generators, obtained in an iterative procedure
            where the zeroth stabilizer is not updated at all and the
            <n>-th stabilizer is updated by running <fix_single_term>
            <n> times with the preceding stabilizers in this list.

        fixed_positions (list):

            List of integers, indicating on which positions
            <fix_single_term> checks for corrections. The encounter
            of <fixed_ops[n]> or <other_ops[n]> on position
            <fixed_positions[n]> triggers the multiplication of the
            n-th stabilizer generator including its proper syndrome sign.

        fixed_ops (str):

            String of characters 'X', 'Y' and 'Z'  indicating one of the
            possible Pauli types against which <fix_single_term> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

        other_ops (str):

            String of characters 'X', 'Y' and 'Z' indicating the other type
            of Pauli operators against which <fix_single_term> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

    Returns:

        (list):

            List of tuples with integer entries 0 and 1, where the <n>-th
            tuple corresponds to the transformed version of the syndrome
            pattern <old_syndr[n]>.
    """

    tri_mtx = numpy.zeros([len(original_stabs)] * 2)

    # Obtain the inverse syndrome propagation matrix through the
    #   tapering procedure.
    for i, x in enumerate(original_stabs):
        _, tri_mtx[i, :] = generator_dependence(
            x,
            new_stabs,
            fixed_positions,
            fixed_ops,
            other_ops)

    # invert the matrix and compute the new syndromes by matrix multiplication
    inv_mtx = invert_tri_matrix(tri_mtx)
    return [tuple((inv_mtx @ x) % 2) for x in original_syndr]


def _logical_state_simulation_shot(
        stabilizer_list: Sequence,
        n_phys_qubits: int,
        state_vec: numpy.ndarray,
        syndromes: numpy.ndarray,
        qubit_order: Sequence,
        fixed_positions: Sequence,
        fixed_ops: str,
        other_ops: str,
        stat_noise: Sequence,
        gate_noise: Sequence,
        schedule: Sequence) -> Tuple[numpy.ndarray, list]:
    r"""
    Single shot of the logical state simulation.
    This routine takes the results from the noisy state preparation and
    goes through the schedule of the ansatz / time evolution circuit,
    inserting noise operators at random and computing the effect of
    the noisy subcircuits on the logical state and a syndrome pattern.
    When the circuit is processed, the state vector and syndrome pattern
    are returned.

    Args:

        stabilizer_list (list):

            List of signed Pauli strings of type QubitOperator; the updated
            list of stabilizer generators, obtained in an iterative procedure
            where the zeroth stabilizer is not updated at all and the
            <n>-th stabilizer is updated by running <fix_single_term>
            <n> times with the preceding stabilizers in this list.

        n_phys_qubits (int):

            Number of physical qubits in the system.

        state_vec (numpy.ndarray):

            Logical state vector after the state preparation circuit as a
            one-dimensional array of complex floats. The array entries
            correspond to the coefficients of the computational state
            configurations

                [|000>, |001>, |010>, |011>, |100>, ...].

        syndromes (numpy.ndarray):

            One-dimensional array of integers 0 and 1 indicating a syndrome
            pattern with respect to the stabilizers <stabilizer_list>, where
            <syndromes[n] == 1> indicates that <stabilizer_list[n]> is flipped
            in the stabilizer state.

        qubit_order (list):

            List containing integers and flags 'rm'. This list indicates which
            qubits are removed after the correction, and how the rest of the
            qubits are relabeled. After the corrections by <fix_single_term>,
            the <n>-th qubit is relabeled as <qubit_order[n]>, or removed if
            <qubit_order[n]> = 'rm'.

        fixed_positions (list):

            List of integers, indicating on which positions
            <fix_single_term> checks for corrections. The encounter
            of <fixed_ops[n]> or <other_ops[n]> on position
            <fixed_positions[n]> triggers the multiplication of the
            n-th stabilizer generator including its proper syndrome sign.

        fixed_ops (str):

            String of characters 'X', 'Y' and 'Z'  indicating one of the
            possible Pauli types against which <fix_single_term> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

        other_ops (str):

            String of characters 'X', 'Y' and 'Z' indicating the other type
            of Pauli operators against which <fix_single_term> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

        stat_noise (list):

            List characterizing the single-qubit noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli operators (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli operators have the index 0 as a placeholder
            for the proper label of the qubit the noise acts on.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli operators.

        gate_noise (list):

            List characterizing the gate noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli strings (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli strings itself are on two qubits, where
            the integers 0 and 1 function as placeholders for the labels of
            control and target qubit, respectively.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli string.

        schedule (list):

            List of circuit events containing two types of elements:

            (i) Lists of the form ['idl', <n>: int, <m>: int],
            indicating that the <n>-th qubit is idle for
            <m> time steps.

            (ii) Lists of the form  ['rot', <pstring>: tuple, <angle>: float],
            describing the rotation of the Pauli string
            QubitOperator(<pstring>) about the angle <angle>.

    Returns:

        (numpy.ndarray):

            One-dimensional array of complex floats; the logical state vector
            <state_vec> after the circuit.

        (numpy.ndarray):

            One-dimensional array of integers 0 and 1; the syndrome
            pattern <syndromes> after the circuit.
    """
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

        if phys_op != QubitOperator(()) and phys_op != QubitOperator((), -1):
            state_vec = get_sparse_operator(
                reduce_to_logical(
                    phys_op,
                    syndromes,
                    stabilizer_list,
                    qubit_order,
                    fixed_positions,
                    fixed_ops,
                    other_ops),
                n_phys_qubits - len(stabilizer_list)
            ).dot(state_vec)

    return state_vec, syndromes


def logical_state_simulation(
    stabilizers: Union[Sequence, QubitOperator],
    logical_operators: List,
    n_phys_qubits: int,
    rounds: int,
    phys_circuit: Sequence,
    stat_noise: list,
    gate_noise: list,
    state_prep: list = [],
    d_matrix_blocks: str = 'code',
    block_numbers: Union[list, tuple] = (),
    queue_size: int = 100,
    num_processes: int = 0) -> Tuple[
                                        Union[numpy.ndarray, dict],
                                        QubitOperator,
                                        numpy.ndarray,
                                        tuple,
                                        int]:
    r"""
    Simulates a physical quantum circuit acting on quantum state with
    stabilizers under Pauli noise. In this simulator, noise operators are
    statistically placed in and around circuit subroutines, and
    symbolically reduced to their logical representation: Pauli strings on
    the computational state space. Logical state vectors and
    syndrome patterns constitute shots with which the density matrix is
    reconstructed. The density matrix itself is block diagonal with
    respect to stabilizer states of different syndrome patterns, including the
    code space. The simulation can save any number of these blocks
    corresponding to different syndrome numbers. Perfect postselection for
    instance would only require to keep the code space block and the rest can
    be discarded. The states in every block are logical states of the
    stabilizer state corresponding to their syndrome pattern.

    The simulated circuit has two parts, a time evolution / ansatz circuit
    featuring Pauli string rotation subcircuits and a state preparation
    routine featuring subcircuits for projective measurements. Besides the
    stabilizer generators that constrain the system at every point in the
    circuit, the system is additionally constrained by a number of logical
    operators (signed Pauli strings) after the state preparation circuit.
    This extended list of stabilizer generators constrains the state
    completely, meaning there are as many extended stabilizer generators
    as there are physical qubits and the computational subspace has zero
    degrees of freedom. It is possible within this simulator to give the
    state preparation circuit an entirely new pattern of projective
    measurements, as long as they span the same eigenspace as the original
    stabilizers and logical operators.

    The computation of the simulation results is parallelized where possible,
    and the returns include technical data for further analysis.

    Args:

        stabilizers (QubitOperator or list):

            List of symbolic expressions of type QubitOperator
            (or a QubitOperator-typed sum of signed Pauli strings)
            signifying stabilizer generators.

        logical_operators (list):

            List of symbolic expressions of type QubitOperator, signifying the
            logical operators stabilizing the initial state, given in their
            physical representation.

        n_phys_qubits (int):

            Number of physical qubits in the circuit.

        rounds (int):

            Number of shots for the reconstruction of the density matrix.
            The state preparation and time evolution / ansatz circuit is
            evaluated at every shot.

        phys_circuit (list, tuple or other sequences):

            Sequences of QubitOperator-typed symbolic expressions of the form

                    QubitOperator(<pstring>: str, <angle>: float)

            signifying rotation subcircuits of the Pauli strings <pstring>
            about the angle <angle> in the time evolution / ansatz circuit.
            The subcircuits are placed into the circuit in the order of this
            sequence.

        stat_noise (list):

            List characterizing the single-qubit noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli operators (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli operators have the index 0 as a placeholder
            for the proper label of the qubit the noise acts on.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli operators.

        gate_noise (list):

            List characterizing the gate noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli strings (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli strings itself are on two qubits, where
            the integers 0 and 1 function as placeholders for the labels of
            control and target qubit, respectively.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli string.

        state_prep (list) [optional]:

            List of symbolic expressions of type QubitOperator. This
            list contains Pauli strings that are measured in the state
            preparation routine, rather than the Pauli strings
            <stabilizers> and <logical_operators>. If the list is left
            empty, <stabilizers> and <logical_operators> are used. The
            list is empty by default.

        d_matrix_blocks (str) [optional]:

            String of characters, expected to be one of the key words 'code',
            'all' or 'custom' setting the mode of operations. Set to 'code' by
            default. This parameter decides which blocks of the density matrix
            to keep, and determines the output format.

            - 'all' keeps blocks of all syndromes, and returns the
              reconstructed density matrix as a high-dimensional array that
              would return a particular block by taking its syndrome pattern
              as a parameter, see below.

            - 'code' only keeps the code block which is returned as a
              two-dimensional array.

            - 'custom' keeps only the blocks with syndrome numbers specified
              in <block_numbers>, and the reconstructed density matrix is
              returned as a dictionary, with the transformed syndromes as
              keys. These syndromes are generally different from the ones in
              <block_numbers> due to a necessary rearrangement of the
              stabilizer generators, that can be retraced with the other
              outputs of this routine.

        block_numbers (list or tuple) [optional]:

            Sequence of tuples containing integers 0 and 1. A collection of
            syndrome patterns saved in the reconstructed density matrix when
            <d_matrix_blocks> is set to 'custom'. Empty by default.
            This sequence has no function if <d_matrix_blocks> is set to
            anything else.

        queue_size (int) [optional]:

            Maximum number of state vectors and syndrome patterns stored in a
            queue filled by parallel processing waiting to be popped by the
            Numpy part of the simulation updating the density matrix. The
            Numpy part of the simulator can empty the queue several times and
            <rounds> does not need to be an integer multiple of <queue_size>.
            Set to 100 by default.

        num_processes (int) [optional]:

            Number of workers in the process pool parallelizing the shots
            right up to the numpy part of the simulation. If <num_processes>
            is set to zero the number of processes set by a CPU count.
            Zero by default.

    Returns:

        mtx (numpy.ndarray, dict):

            Reconstructed density matrix.

            If <d_matrix_blocks> = 'all', <mtx> is an array with
            <numpy.shape(mtx) = ( *A, *B)>, where <A = (2, 2, 2, ... , 2)> and
            <B = (2**n, 2**n)> with <n> being the number of logical qubits.
            The idea is that this matrix prompts a density matrix block
            associated with a syndrome pattern, a tuple <syndr>, by using it
            as an input:

                                        mtx[syndr].

            If <d_matrix_blocks> = 'code', <mtx> is a two-dimensional array,
            the block associated with the code space.

            If <d_matrix_blocks> = 'custom', <mtx> is of type dictionary with
            syndrome patterns as keys. The syndrome patterns are a version of
            the syndromes specified in <block_numbers>, but updated to fit the
            set of stabilizer generators <upd_stabs> used to fix the logical
            representation.

        block_entries (int):

            Absolute number of shots that went into the density matrix.
            If <d_matrix_blocks> is set to 'all', this number is equal to
            <rounds>.

        ideal_state (numpy.ndarray):

            One-dimensional state encoding the ideal logical state of the code
            space stemming from a noiseless evaluation of the circuit.

        depth (int):

            Total depth of the quantum circuit.

        (tuple):

                Collection of technical data; a tuple
                (<stabilizer_list>, <qubit_order>, <fixed_positions>,
                <fixed_ops>, <other_ops>), where:

                 <stabilizer_list>: list

                 List of signed Pauli strings of type QubitOperator; the
                 updated list of stabilizer generators, obtained in an
                 iterative procedure where the zeroth stabilizer is not
                 updated at all and the <n>-th stabilizer is updated by
                 running <fix_single_term> <n> times with the preceding
                 stabilizers in this list.

                 <qubit_order>: list

                 List containing integers and flags 'rm'. This list indicates
                 which qubits are removed after the correction, and how the
                 rest of the qubits are relabeled. After the corrections by
                 <fix_single_term>, the <n>-th qubit is relabeled as
                 <qubit_order[n]>, or removed if <qubit_order[n]> = 'rm'.

                 <fixed_positions>: list

                 List of integers, indicating on which positions
                 <fix_single_term> checks for corrections. The encounter
                 of <fixed_ops[n]> or <other_ops[n]> on position
                 <fixed_positions[n]> triggers the multiplication of the
                 n-th stabilizer generator including its proper syndrome sign.

                 <fixed_ops>: str
                
                 String of characters 'X', 'Y' and 'Z'  indicating one of the
                 possible Pauli types against which <fix_single_term> checks
                 on positions specified in <fixed_positions>. The <n>-th
                 character corresponds to the Pauli type that is checked
                 against on position <fixed_positions[n]>.

                 <other_ops>: str

                 String of characters 'X', 'Y' and 'Z' indicating the other
                 type of Pauli operators against which <fix_single_term>
                 checks on positions specified in <fixed_positions>.
                 The <n>-th character corresponds to the Pauli type that is
                 checked against on position <fixed_positions[n]>.

    Raises:

        (SimulatorError):

            Custom error triggered by several checks of items in
            <stabilizers>, <logical_operators> and <state_prep>, including
            checks making sure all symbolic expressions are signed nontrivial
            Pauli strings that do not anticommute or contradict each other,
            as well as mismatches in the number of qubits and stabilizer
            conditions.

        (TypeError):

            Error triggered when items in <stabilizers>, <logical_operators>
            or <state_prep> are not of type QubitOperator.
    """
    # Run some checks: types, dimensions, coefficients, anticommutations
    if len(stabilizers) + len(logical_operators) != n_phys_qubits:
        raise SimulatorError(
            'Number of stabilizers (' +
            str(len(stabilizers)) +
            ')  + number logical operators (' +
            str(len(logical_operators)) +
            ') is not equal to the number physical qubits (' +
            str(n_phys_qubits) + ')')

    for i, x in enumerate(stabilizers):
        if type(x) != QubitOperator:
            raise TypeError(
                'Stabilizer ' +
                str(i) +
                ' is not of type QubitOperator.'
                )

        if count_qubits(x) > n_phys_qubits:
            raise SimulatorError(
                'Stabilizer ' +
                str(i) +
                ' has more qubits than specified.')

        coeff = list(x.terms.values())[0]
        if abs(coeff - 1) > EQ_TOLERANCE and abs(coeff + 1) > EQ_TOLERANCE:
            raise SimulatorError(
                'Stabilizer ' +
                str(i) +
                ' has coefficient that is neither +1 nor -1.')

        check_for_identity(
            x,
            'Stabilizer ' +
            str(i) +
            ' is trivial.')

        check_for_imaginary_coeff(
            [y * x for y in list(stabilizers)[: i]],
            'Stabilizers anticommute.')

    # Fix the stabilizers
    try:

        (upd_stabs, fixed_positions,
            fixed_ops, other_ops) = prepare_stabilizers(stabilizers)

    except SimulatorError:
        raise SimulatorError('Stabilizers are linearly dependent.')

    for i, x in enumerate(logical_operators):
        if type(x) != QubitOperator:
            raise TypeError(
                'Logical operator ' +
                str(i) +
                ' is not of type QubitOperator.'
                )

        if count_qubits(x) > n_phys_qubits:
            raise SimulatorError(
                'Logical operator ' +
                str(i) +
                ' has more qubits than specified.')

        coeff = list(x.terms.values())[0]
        if abs(coeff - 1) > EQ_TOLERANCE and abs(coeff + 1) > EQ_TOLERANCE:
            raise SimulatorError(
                'Input for logical operator ' +
                str(i) +
                ' has coefficient that is neither +1 nor -1.')

        syndr = anticom_syndr(x, stabilizers)
        if False in (syndr == 0):
            raise SimulatorError(
                'Input for logical operator ' +
                str(i) +
                ' anticommutes with some stabilizers.')

        check_for_identity(
            x,
            'Logical operator ' +
            str(i) +
            ' is trivial.')

        check_for_imaginary_coeff(
            [y * x for y in list(logical_operators)[: i]],
            'Logical operators anticommute.')

    try:
        _ = prepare_stabilizers(list(stabilizers) + list(logical_operators))

    except SimulatorError:
        raise SimulatorError(
            'Extended list of stabilizers (code stabilizers plus logical' +
            '  operators) has linearly dependent elements.')

    for i, x in enumerate(phys_circuit):

        if type(x) != QubitOperator:
            raise TypeError(
                'Rotation ' +
                str(i) +
                ' is not of type QubitOperator.'
                )

        if count_qubits(x) > n_phys_qubits:
            raise SimulatorError(
                'Rotation circuit ' +
                str(i) +
                ' acts on more qubits than specified.')

        if abs(list(x.terms.values())[0].imag) > EQ_TOLERANCE:
            raise SimulatorError(
                'Imaginary angle in rotation ' +
                str(i))

        syndr = anticom_syndr(x, stabilizers)
        if False in (syndr == 0):
            raise SimulatorError(
                'Rotation ' + str(i) +
                ' is not a logical operator.')

        check_for_identity(
            x,
            'Rotation ' +
            str(i) +
            ' is trivial.')

    # if the state preparation list is empty, just take the logical operators,
    #  otherwise check the qubit numbers
    if state_prep == []:
        state_prep = list(stabilizers) + list(logical_operators)
        for i, x in enumerate(state_prep):
            check_for_imaginary_coeff(
                [y * x for y in list(state_prep)[: i]],
                'Stabilizers anticommute with logical operators.')
        try:
            _ = prepare_stabilizers(list(state_prep))

        except SimulatorError:
            raise SimulatorError(
                'Logical operators are linearly dependent on stabilizers.')

    else:
        if len(state_prep) != n_phys_qubits:
            raise SimulatorError(
                'Number of state prep operators (' +
                str(len(state_prep)) +
                ') is not equal to the number of physical qubits (' +
                str(n_phys_qubits) + ').')

        for i, x in enumerate(state_prep):
            if type(x) != QubitOperator:
                raise TypeError(
                    'State prep operator ' +
                    str(i) +
                    ' is not of type QubitOperator.'
                    )

            coeff = list(x.terms.values())[0]
            if (
                    abs(coeff - 1) > EQ_TOLERANCE and
                    abs(coeff + 1) > EQ_TOLERANCE):
                raise SimulatorError(
                        'State prep operator ' +
                        str(i) +
                        ' has coefficient that is neither +1 nor -1.')

            if count_qubits(x) > n_phys_qubits:
                raise SimulatorError(
                    'State prep operator ' +
                    str(i) +
                    ' has more qubits than specified.')

            check_for_identity(
                x,
                'State prep operator ' +
                str(i) +
                ' is trivial.')

            syndr = anticom_syndr(
                x,
                list(stabilizers) +
                list(logical_operators))

            if False in (syndr == 0):
                raise SimulatorError(
                    'State prep operator ' +
                    str(i) +
                    ' anticommutes with some stabilizers / logical operators.')

            check_for_imaginary_coeff(
                [y * x for y in list(logical_operators)[: i]],
                'State prep operators anticommute.')

        try:
            _ = prepare_stabilizers(list(state_prep))

        except SimulatorError:
            raise SimulatorError(
                'State prep operators are linearly dependent.')

    # Prepare the qubit_order list such that you could put it into
    #   reduce_to_logical()
    n_log_qubits = n_phys_qubits - len(fixed_positions)
    qubit_order = list(range(n_log_qubits))
    removed_positions = list(fixed_positions)
    removed_positions.sort()

    for x in removed_positions:
        qubit_order.insert(x, 'rm')

    # make schedules for the circuit and the state preparation,
    #  so one knows where to put noise operations in between the
    #  fixed gadgets
    init_schedule, init_depth = make_simulation_schedule(
                                state_prep,
                                n_phys_qubits, 'meas')
    circ_schedule, circ_depth = make_simulation_schedule(
                                phys_circuit,
                                n_phys_qubits, 'rot')

    depth = init_depth + circ_depth

    # the syndromes of the measured Pauli strings needs to be handed
    #  down to the syndromes properly. Obtain a rudimentary state, that
    #  can be projected into the correct logical state, a matrix for
    #  syndrome propagation and the reduced logical operators.
    (init_state_rudiment,
        syndr_prop_mtx,
        log_ops_redux) = prepare_state_prep(
        state_prep,
        list(logical_operators),  # making sure inputs are not overwritten
        upd_stabs,
        qubit_order,
        fixed_positions,
        fixed_ops,
        other_ops)

    # obtain the noiseless result
    # first, shortcut the  the noiseless state preparation
    ideal_state, _ = make_initial_state(
        init_state_rudiment,
        log_ops_redux,
        numpy.zeros(n_phys_qubits, dtype=int))

    # then, make the logical Pauli rotations
    for x in phys_circuit:
        angle = numpy.real(list(x.terms.values())[0])
        ideal_state = (cos(angle) * ideal_state +
                       1.j * sin(angle) *
                       get_sparse_operator(
                        reduce_to_logical(
                            QubitOperator(list(x.terms.keys())[0]),
                            numpy.zeros(
                                n_phys_qubits - n_log_qubits,
                                dtype=int),
                            upd_stabs,
                            qubit_order,
                            fixed_positions,
                            fixed_ops,
                            other_ops), n_log_qubits).dot(ideal_state))

    # The array <mtx> reconstructs the density matrix.
    # If d_matrix_blocks = 'all', it has two parts:
    #    1)    a part that holds the syndrome patterns as bit string vectors
    #    2)    a part that holds the density matrix of the logical subspace
    #        associated with the syndrome pattern.
    # When the first part is converted to a tuple eg. b = (0,1,1,0,1)
    # then mtx[b] outputs the density matrix block of the (0,1,1,0,1)
    # If d_matrix_blocks = 'code', it is an array encoding the code space.
    # If d_matrix_blocks = 'custom', mtx is a dictionary
    # with the syndrome patters of the propagated block numbers as keys,
    # and the density matrices as values. The propagated block numbers are
    # correspond to the syndromes in <block_numbers>,
    # for the updated stabilizers.
    if d_matrix_blocks == 'all':
        mtx = numpy.zeros([2] * len(fixed_positions) +
                          2 * [2 ** n_log_qubits], dtype=complex)
    elif d_matrix_blocks == 'code':
        mtx = numpy.zeros(2 * [2 ** n_log_qubits], dtype=complex)

    else:
        block_numbers = list({tuple(x) for x in block_numbers})
        if False in [len(x) == (n_phys_qubits - n_log_qubits) for x in block_numbers]:
            raise SimulatorError('Length mismatch between state syndrome' +
                ' patterns and input block number(s).')
        if False in [set(x) in ({0,}, {1,}, {0, 1}) for x in block_numbers]:
            raise SimulatorError('Block number inputs must be 0 or 1.')
        upd_block_numbers = propagate_syndromes(
            block_numbers,
            list(stabilizers),
            upd_stabs,
            fixed_positions,
            fixed_ops,
            other_ops)

        mtx = dict()
        for x in upd_block_numbers:
            mtx[x] = numpy.zeros(2 * [2 ** n_log_qubits], dtype=complex)

    #  block_entries measures how many times we bin an element into a block
    block_entries = 0
    if num_processes == 0:
        num_processes = cpu_count()

    # begin of the main sequence
    #  In each iteration, a number of workers calculates a chunk of shots.
    #  The number of iterations is determined by the rounded up fraction of
    #  rounds and queue_size.
    for x in range(ceil(rounds / queue_size)):

        # the last iteration might be shorter than the others if the
        #  fraction is not integer. Take that into account by computing
        #  the actual size of the processing queue.
        actual_qsize = queue_size
        if x == ceil(rounds / queue_size) - 1 and rounds % queue_size != 0:
            actual_qsize = rounds % queue_size
        results = list(
            pypeln.process.map(
                lambda x: _logical_state_simulation_shot(
                    list(upd_stabs),
                    int(n_phys_qubits),
                    *make_initial_state(
                        numpy.array(init_state_rudiment),
                        list(log_ops_redux),
                        syndr_prop_mtx @ simulate_state_prep(
                            list(init_schedule),
                            list(state_prep),
                            list(stat_noise),
                            list(gate_noise)) % 2),
                    list(qubit_order),
                    list(fixed_positions),
                    str(fixed_ops),
                    str(other_ops),
                    list(stat_noise),
                    list(gate_noise),
                    list(circ_schedule)),
                range(actual_qsize),
                workers=num_processes))

        for vec, syndr in list(results):
            # Add the results of the shots to the density matrix.
            # If we have a limited number of blocks, discard the results
            #  that cannot be binned.
            if d_matrix_blocks == 'code' and sum(syndr) == 0:
                mtx += numpy.kron(
                    vec,
                    vec.conj()).reshape([2 ** n_log_qubits] * 2)
                block_entries += 1

            elif d_matrix_blocks == 'all':
                mtx[tuple(syndr)] += numpy.kron(vec, vec.conj()).reshape(
                    [2 ** n_log_qubits] * 2)
                block_entries += 1

            elif (d_matrix_blocks == 'custom' and
                    tuple(syndr) in upd_block_numbers):
                mtx[tuple(syndr)] += numpy.kron(vec, vec.conj()).reshape(
                    [2 ** n_log_qubits] * 2)
                block_entries += 1

    # output the results
    if d_matrix_blocks in ('code', 'all') and block_entries > 0:
        mtx = mtx / block_entries

    elif d_matrix_blocks == 'custom':
        for x in list(mtx.keys()):
            if block_entries > 0:
                mtx[x] = mtx[x] / block_entries

    return (mtx,
            block_entries,
            ideal_state,
            depth,
            (upd_stabs,
             qubit_order,
             fixed_positions,
             fixed_ops,
             other_ops))

from openfermion import QubitOperator

from .logical_vector_construction import SimulatorError
from .logical_vector_simulator import logical_state_simulation
from .noise import noiseless
from .utils import expectation_value, fidelity, anticom
from typing import Sequence, Union
from copy import deepcopy
import numpy

class pylove_state:
    """
    Attributes:
        state:

            Returns the reconstructed density matrix.

            If `self.mode == 'all'`, the output is an array of the shape
            `( *A, *B)`, where <A = (2, 2, 2, ... , 2)> and
            `B = (2 ** n, 2 ** n)` with `n = self.n_log_qubits` being the
            number of logical qubits. The idea is that this matrix prompts a
            density matrix block associated with a syndrome pattern, a tuple
            `syndr`, by using it as an input:
            ``
                                    self.state[syndr].
            ``
            If `self.mode == 'code'`, the output is a two-dimensional array:
            the density matrix block associated with the code space.

            If `self.mode == 'custom'`, the output is of type dictionary
            with syndrome patterns as keys. The syndrome patterns are a
            version of the syndromes specified in the input `block_numbers`,
            but updated to fit the set of stabilizer generators in
            `self.stabs`.

            If `self.mode == 'vector'`, the output is `numpy.ndarray` vector.

        mode:

            Outputs a string indicating the format of `self.state`.
            Possible configurations are `'all'`, `'code'`, `'custom'`
            or `'vector'`.

        ideal_state:

            The logical state if the circuit was noiseless,
            stored as a vector.

        n_physical_qubits:

            Number of physical qubits in the state.

        n_log_qubits:

            Number of logical qubits; the size of the computational subspace.
        
        circuit_depth:

            Depth of the quantum circuit (including the state preparation)
            that created the state.
        
        stabs:

            List of signed Pauli strings of type QubitOperator; the
            updated list of stabilizer generators.

        n_entries:

            Number of shots that were binned to reconstruct the density
            matrix. The difference between the total number of shots and
            `self.n_entries` has been discarded during the simulation.

        qubit_order:

            List of integers and flags 'rm', defining the relationship between
            physical and logical qubits. If `self.qubit_order[m] == n`, where  
            `n` is an integer, then the `m`th physical qubit becomes the `n`th
            logical qubit. If `self.qubit_order[m] == 'rm'`, then the qubit is
            removed.

        fixed_positions:

            List of integers, indicating which physical qubit is fixed by
            which stabilizer. The `n`-th stabilizer `self.stabs[n]` fixes
            the `fixed_positions[n]`-th physical qubit.
        
        fixed_ops:

            String of characters `'X'`, `'Y'` and `'Z'` indicating the
            type of Pauli operators fixed by each stabilizer.
            The `n`-th stabilizer `self.stabs[n]` fixes
            the `fixed_positions[n]`-th physical qubit
            to the operator `fixed_ops[n]`.

        other_ops:

            String of characters `'X'`, `'Y'` and `'Z'` indicating the
            the type of Pauli operators fixed by each stabilizer, in addition
            to `fixed_ops`. To remove the `n`th qubit from a physical
            operator, it must act trivially on the `n`th qubit. That is, it
            must act as either the identity or `P`, where
            `(P, self.fixed_ops[n], self.other_ops[n])`is a permutation
            of `('X', 'Y', 'Z')`.
    """

    def __init__(
            self,
            state: Union[numpy.ndarray, dict],
            mode: str,
            n_entries: int,
            ideal_state: numpy.ndarray,
            n_phys_qubits: int,
            stabs: list,
            qubit_order: list,
            fixed_positions: list,
            fixed_ops: str,
            other_ops: str,
            circuit_depth: int):

        self.n_entries = n_entries
        self.ideal_state = ideal_state
        self.circuit_depth = circuit_depth
        self.mode = mode
        self.n_phys_qubits = n_phys_qubits
        self.n_log_qubits = n_phys_qubits - len(fixed_ops)
        self.stabs = stabs
        self.qubit_order = qubit_order
        self.fixed_positions = fixed_positions
        self.fixed_ops = fixed_ops
        self.other_ops = other_ops
        self.state = state


    def ideal(self):
        """
        Creates a `pylove_state` instance of the current instance's ideal
        version; the state returned by the noiseless version of the
        quantum circuit. 
        """
        return pylove_state(
            self.ideal_state,
            'vector',
            1,
            [],
            self.n_phys_qubits,
            self.stabs,
            self.qubit_order,
            self.fixed_positions,
            self.fixed_ops,
            self.other_ops,
            self.circuit_depth)
 

def pylove_simulation(
        stabilizers: Union[Sequence, QubitOperator],
        logical_operators: Sequence,
        quantum_circuit: Sequence,
        shots: int,
        wire_noise: list = noiseless(),
        gate_noise: list = noiseless(),
        state_prep_circuit: Sequence = [],
        mode: str = 'code',
        block_numbers: Sequence = (),
        queue_size: int = 100,
        num_processes: int = 0):
    """
    Args:

        stabilizers (QubitOperator or list):

            List of symbolic expressions of type `QubitOperator`
            (or a `QubitOperator`-typed sum of signed Pauli strings)
            signifying stabilizer generators.

        logical_operators (list):

            List of symbolic expressions of type `QubitOperator`, signifying
            the logical operators stabilizing the initial state, given in
            their physical representation.

        quantum_circuit (list, tuple or other sequences):

            Sequences of QubitOperator-typed symbolic expressions of the form
            ``
                    QubitOperator(pstring: str, angle: float)
            ``
            signifying rotation subcircuits of the Pauli strings `pstring`
            about the angle `angle` in the time evolution/ansatz circuit.
            The subcircuits are placed into the circuit in the order of this
            sequence.

        shots (int):

            Number of shots for the reconstruction of the density matrix.
            The state preparation and time evolution/ansatz circuit is
            evaluated at every shot.

        wire_noise (list):

            List characterizing the single-qubit noise model. It has the form
            `[A, B]`, holding two sublists `A` and `B`:

            `A` is a list of Pauli operators (`QubitOperators`) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli operators have the index `0` as a placeholder
            for the proper label of the qubit the noise acts on.

            `B` is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli operators.

        gate_noise (list):

            List characterizing the gate noise model. It has the form
            `[A, B]`, holding two sublists `A` and `B`:

            `A` is a list of Pauli strings (`QubitOperators`) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli strings itself are on two qubits, where
            the integers `0` and `1` function as placeholders for the labels of
            control and target qubit, respectively.

            `B` is a list holding `float` numbers corresponding to the
            statistical weights of their respective Pauli string.

        state_prep_circuit (list) [optional]:

            List of symbolic expressions of type `QubitOperator` outlining
            the schedule for the state preparation circuit. This
            list contains the signed Pauli strings in the order the
            measurement sub-circuits are placed into the circuit before the
            ansatz/time evolution circuit `quantum_circuit` is run.
            If `state_prep_circuit` is left empty, the schedule is made from
            appending the inputs for the parameters `stabilizers` and
            `logical_operators`. The list is empty by default.

        mode (str) [optional]:

            String of characters, expected to be one of the key words
            `'code'`, `'all'` or `'custom'` setting the mode of operation.
            Set to `'code'` by default. This parameter decides which blocks of
            the density matrix to keep, and determines the output format.

            - `mode='all'` keeps blocks of all syndromes, and returns the
              reconstructed density matrix as a high-dimensional array that
              would return a particular block by taking its syndrome pattern
              as a parameter, see below.

            - `mode='code'` only keeps the code block which is returned as a
              two-dimensional array.

            - `mode='custom'` keeps only the blocks with syndrome numbers
               specified in `block_numbers`, and the reconstructed density
               matrix is returned as a dictionary, with the transformed
               syndromes as keys. These syndromes are generally different from
               the ones in `block_numbers` due to a necessary rearrangement
               of the stabilizer generators, that can be retraced with the
               other outputs of this routine.

        block_numbers (list or tuple) [optional]:

            Sequence of tuples containing integers `0` and `1`. A collection
            of syndrome patterns saved in the reconstructed density matrix
            when `mode` is set to `'custom'`. Empty by default.
            This `block_numbers` is ignored if `mode` is set to
            anything else.

        queue_size (int) [optional]:

            Maximum number of state vectors and syndrome patterns stored in a
            queue filled by parallel processing waiting to be popped by the
            Numpy part of the simulation updating the density matrix. The
            Numpy part of the simulator can empty the queue several times and
            `shots` does not need to be an integer multiple of `queue_size`.
            Set to `100` by default.

        num_processes (int) [optional]:

            Number of workers in the process pool parallelizing the shots
            right up to the numpy part of the simulation. If `num_processes`
            is set to zero the number of processes set by a CPU count.
            Zero by default.
    
    Returns:

        (pylove_state):

            Density matrix object resulting the simulation. 

    """
    n_phys_qubits = len(list(stabilizers)) + len(list(logical_operators))

    result = logical_state_simulation(
        stabilizers=stabilizers,
        logical_operators=logical_operators,
        n_phys_qubits=n_phys_qubits,
        rounds=shots,
        phys_circuit=quantum_circuit,
        stat_noise=wire_noise,
        gate_noise=gate_noise,
        state_prep=state_prep_circuit,
        d_matrix_blocks=mode,
        block_numbers=block_numbers,
        queue_size=queue_size,
        num_processes=num_processes)
        
    return pylove_state(
        state=result[0],
        n_entries=result[1],
        ideal_state=result[2],
        circuit_depth=result[3],
        mode=mode,
        n_phys_qubits=n_phys_qubits,
        stabs=result[4][0],
        qubit_order=result[4][1],
        fixed_positions=result[4][2],
        fixed_ops=result[4][3],
        other_ops=result[4][4])
 

def tr(
        arg1: Union[QubitOperator, pylove_state],
        arg2: Union[QubitOperator, pylove_state] = QubitOperator(())
        ) -> float:
    """
    Trace of a density matrix with an operator (outputting an expectation
    value) or an ideal state (outputting a fidelity).

    Args:

        arg1 (pylove_state or QubitOperator):

            Quantum state or physical-level symbolic operator.
        
        arg2 (pylove_state or QubitOperator) [optional]:

            Quantum state or physical-level symbolic operator. 
            Note that `arg1` and `arg2` cannot both be mixed
            states.
    
    Returns:

        (float):

                Fidelity or expectation value, depending on input
                formats.
    """

    if type(arg1) is pylove_state:
        (state, other) = (arg1, arg2)

    elif type(arg2) is pylove_state:
        (state, other) = (arg2, arg1)

    else:
        raise SimulatorError('Trace function requires' +
        ' a quantum state to be present.')

    if type(other) is pylove_state:

        if not all([getattr(state, x) == getattr(other, x) for x in (
            'stabs',
            'qubit_order',
            'fixed_positions',
            'fixed_ops',
            'other_ops')]):

            raise SimulatorError('Cannot compare two states with' +
                'mismatching stabilizer generators.')

        elif (state.mode, other.mode) == ('vector', 'vector'):
            return abs(state.state.conj() @ other.state) ** 2

        elif (state.mode, other.mode) == ('code', 'vector'):
            return numpy.real(other.state.conj() @ state.state @ other.state)

        elif (state.mode, other.mode) == ('vector', 'code'):
            return numpy.real(state.state.conj() @ other.state @ state.state)

        elif (state.mode, other.mode) == ('all', 'vector'):
            return numpy.real(
                other.state.conj() @
                state.state[
                    tuple([0] * (state.n_phys_qubits - state.n_log_qubits))] @
                other.state)  
        elif (state.mode, other.mode) == ('vector', 'all'):
            return numpy.real(
                state.state.conj() @
                other.state[
                    tuple([0] * (state.n_phys_qubits - state.n_log_qubits))] @
                state.state)
        elif (state.mode, other.mode) == ('custom', 'vector'):
            if tuple(
                [0,] *
                (state.n_phys_qubits -
                    state.n_log_qubits)) in state.state.keys():
                return numpy.real(
                    other.state.conj() @
                    state.state[tuple(
                        [0] * (state.n_phys_qubits - state.n_log_qubits))] @
                    other.state)
            else:
                return 0
        elif (state.mode, other.mode) == ('vector', 'custom'):
            if tuple(
                [0,] *
                (state.n_phys_qubits -
                state.n_log_qubits)) in other.state.keys():
                return numpy.real(
                    state.state.conj() @
                    other.state[tuple(
                        [0] * (state.n_phys_qubits - state.n_log_qubits))] @
                    state.state)
            else:
                return 0
        else:
            raise SimulatorError('Trace computations of pylove_states with' +
            ' these modes are currently not supported.')

    elif type(other) is QubitOperator:
        # we only keep logical operators, because the expectation value of
        # the others is gonna vanish anyway
        op = QubitOperator()
        for x in other:
            pstring = QubitOperator(list(x.terms.keys())[0])
            if not True in [anticom(pstring, y) for y in state.stabs]:
                op += x

        return expectation_value(
            operator=op,
            state=state.state,
            stabilizer_list=state.stabs,
            qubit_order=state.qubit_order,
            fixed_positions=state.fixed_positions,
            fixed_ops=state.fixed_ops,
            other_ops=state.other_ops)

    else:
        raise SimulatorError(
            'Argument of type ' +
            str(type(other)) +
            ' is invalid for the trace function.')
    
def postselect(
        state: pylove_state,
        mode: str,
        block_numbers: Sequence = []):
    """
    Restricting a state to a specified set of density matrix sub-blocks,
    equivalent to a post-selection experiment.

    Args:

        state (pylove_state):

            Input state, where `mode` is not already `code`.
        
        mode (str):

            New `mode` for the output density matrix, characterizing the
            restriction to a certain set of density matrix sub-blocks.
            Expects keywords `'code'` or `'custom'`, where 

            - `mode='code'` only keeps the code block.
            - `mode='custom'` keeps only the blocks with syndrome numbers
               specified in `block_numbers`.
        
        block_numbers (list or tuple):

            Sequence of tuples containing integers `0` and `1`. A collection
            of syndrome patterns saved in the reconstructed density matrix
            when `mode` is set to `'custom'`. Empty by default.
            This `block_numbers` is ignored if `mode` is set to
            anything else.
    
    Returns:

        (pylove_state):

            Restricted state, normalized to unit trace.
            Its `mode` matches the function argument,
            it's `n_entries` is updated to reflect the true
            content of the remaining blocks.

    """
    
    if state.mode == 'code':
        raise SimulatorError('Input state is already just the code space.')
    if state.mode == 'vector':
        raise SimulatorError('Input state is already a vector.')
    if mode == 'code':
        block_numbers = [
            tuple([0] * (state.n_phys_qubits - state.n_log_qubits))]
    else:
        block_numbers = list({tuple(x) for x in block_numbers})
        qubit_diff = state.n_phys_qubits - state.n_log_qubits
        if False in [len(x) == qubit_diff for x in block_numbers]:
            raise SimulatorError('Length mismatch between state syndrome' +
                ' patterns and input block number(s).')
        if False in [set(x) in ({0,}, {1,}, {0, 1}) for x in block_numbers]:
            raise SimulatorError('Block number inputs must be 0 or 1.')
    if state.mode == 'custom':
            check = [x in state.state.keys() for x in block_numbers]
            if False in check:
                raise SimulatorError(
                    'Input state does not contain the block of syndrome ' +
                    str(block_numbers[check.index(False)]) + '.')
    if mode == 'code':
        rescale = numpy.real(numpy.trace(state.state[block_numbers[0]]))
        return pylove_state(
            state=numpy.array(state.state[block_numbers[0]]) / rescale,
            n_entries=round(state.n_entries * rescale),
            ideal_state=state.ideal_state,
            circuit_depth=state.circuit_depth,
            mode='code',
            n_phys_qubits=state.n_phys_qubits,
            stabs=state.stabs,
            qubit_order=state.qubit_order,
            fixed_positions=state.fixed_positions,
            fixed_ops=state.fixed_ops,
            other_ops=state.other_ops)
    elif mode == 'custom':
        rescale = 0
        for x in block_numbers:
            rescale += numpy.real(numpy.trace(state.state[x]))
        return pylove_state(
            state=numpy.array(state.state[block_numbers[0]]) / rescale,
            n_entries=round(state.n_entries * rescale),
            ideal_state=state.ideal_state,
            circuit_depth=state.circuit_depth,
            mode='custom',
            n_phys_qubits=state.n_phys_qubits,
            stabs=state.stabs,
            qubit_order=state.qubit_order,
            fixed_positions=state.fixed_positions,
            fixed_ops=state.fixed_ops,
            other_ops=state.other_ops)
    else:
        raise SimulatorError("Mode must be set to 'code' or 'custom'.")
        











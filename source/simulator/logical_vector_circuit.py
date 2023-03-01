"""Functions to generate logical vector simulator circuits."""
from typing import List, Sequence, Tuple, Union
import numpy
from openfermion import QubitOperator
from math import sin, cos
from random import choices
from functools import reduce
from operator import mul


def cnot_sandwich(
        arg: QubitOperator,
        control: int,
        target: int) -> QubitOperator:
    r"""
    Computes the sandwich of a symbolic operator between
    two identical CNOT gates.

    Args:

        arg (QubitOperator):

            Symbolic operator that is plugged in between the CNOT gates.

        control (int):

            Label of the two CNOT gates' control qubit.

        target (int):

            Label of the qubit the two CNOT ates are targeting.

    Returns:

        (QubitOperator):

            Result of CNOT <arg> CNOT in symbolic form.
    """
    sandwich = QubitOperator()
    for x in arg:
        pstring = list(x.terms.keys())[0]

        if ((control, 'X') in pstring or
                (control, 'Y') in pstring):

            x = QubitOperator(((target, 'X'))) * x

        if ((target, 'Z') in pstring or
                (target, 'Y') in pstring):

            x = x * QubitOperator(((control, 'Z')))

        sandwich += x
    return sandwich


def cnot_noise_operation(
        control: int,
        target: int,
        gate_noise: List) -> QubitOperator:
    r"""
    Statistically picks noise operator to be inserted after a CNOT gate
    as gate noise.

    Args:

        control (int):

            Index of the CNOT gate's control qubit.

        target (int):

            Index of the qubit the CNOT gate targets.

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

        (QubitOperator):

            Symbolic two-qubit operator that can be multiplied
            to the CNOT gate. It has the proper labels of
            target and control qubit.

    """

    op = choices(gate_noise[0], weights=gate_noise[1])[0]

    # separate the case of the Kraus operator being the identity
    if op == QubitOperator(()):
        return op

    else:
        new_op = ''

        # give the Kraus operators the proper qubit indices
        for x in list(op.terms)[0]:
            if x[0] == 0:
                new_op += x[1] + str(control) + ' '
            if x[0] == 1:
                new_op += x[1] + str(target) + ' '

        return QubitOperator(new_op)


def noisy_cnot_sandwich(
        arg: QubitOperator,
        control: int,
        target: int,
        stat_noise: List,
        gate_noise: List) -> QubitOperator:
    r"""
    Computes the result of a symbolic operator sandwiched between two
    noisy CNOT gates with identical control and target qubits.

    Args:

        arg (QubitOperator):

            Symbolic operator that is plugged in between the CNOT gates.

        control (int):

            Index of the CNOT gates' control qubit.

        target (int):

            Index of the qubit the CNOT gates target.

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

        (QubitOperator):

            Symbolic expression of CNOT <arg> CNOT, with noisy
            CNOT gates.
    """

    return (
                statistical_noise_operation(control, stat_noise, 1) *
                statistical_noise_operation(target, stat_noise, 1) *
                cnot_noise_operation(control, target, gate_noise) *
                cnot_sandwich(
                    (
                        arg *
                        statistical_noise_operation(control, stat_noise, 1) *
                        statistical_noise_operation(target, stat_noise, 1) *
                        cnot_noise_operation(control, target, gate_noise)
                    ),
                    control,
                    target))


def half_noisy_cnot_sandwich(
        arg: QubitOperator,
        control: int,
        target: int,
        stat_noise: List,
        gate_noise: List,
        left: bool = True) -> QubitOperator:
    r"""
    Compute the symbolic representation of a QubitOperator
    sandwiched by two CNOT gates with identical control and target
    qubits, but where one of the gates is noisy.

    Args:

        arg (QubitOperator):

            Symbolic operator that is plugged in between the CNOT gates.

        control (int):

            Index of the CNOT gates' control qubit.

        target (int):

            Index of the qubit the CNOT gates target.

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

        left (bool):

            Decides whether the noise is on the CNOT gate multiplied to <arg>
            on the left-hand side (which would mean it is on the right-hand
            side of <arg> in a quantum circuit), or not, in which case the
            other CNOT gate is noisy. True by default.

    Returns:

        (QubitOperator):

        Symbolic representation of CNOT <arg> CNOT, where one of the CNOT
        gates is noisy.
    """
    if left:
        return (
                    statistical_noise_operation(control, stat_noise, 1) *
                    statistical_noise_operation(target, stat_noise, 1) *
                    cnot_noise_operation(control, target, gate_noise) *
                    cnot_sandwich(arg, control, target))
    else:
        return cnot_sandwich(
                        (
                            arg *
                            statistical_noise_operation(
                                control,
                                stat_noise,
                                1) *
                            statistical_noise_operation(
                                target,
                                stat_noise,
                                1) *
                            cnot_noise_operation(control, target, gate_noise)
                        ),
                        control,
                        target)


def statistical_noise_operation(
        qubit_index: int,
        stat_noise: List,
        steps: int) -> QubitOperator:
    r"""
    Compute product of noise operators acting on a single qubit after a number
    of time steps, where the noise operators are drawn from the noise model
    according to their statistical weights at every step.

    Args:

        qubit_index (int):

            Label of the noisy qubit.

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

        steps (int):

            Number of time steps, at each of which a noise operator is drawn
            from the model.

    Returns:

        (QubitOperator):

            Symbolic expression of the noise operator acting on the specified
            qubit.
    """
    # I used that for convenience in some routine
    if steps < 1:
        return QubitOperator(())

    else:
        op = reduce(mul, choices(
            stat_noise[0], weights=stat_noise[1], k=steps))

        # separate the case where the operator is the identity
        if list(op.terms) == [()]:
            return op

        else:
            return QubitOperator(list(op.terms)[0][0][1] + str(qubit_index))


def make_noisy_rotation(
        pauli_tuple: tuple,
        angle: float,
        stat_noise: List,
        gate_noise: List) -> Tuple[QubitOperator, int]:
    r"""
    Computes a symbolic expression for a noisy version of a Pauli string
    rotation subcircuit such as the one depicted below.

    Args:

        
            0 _____________
            1 __|_______|__ } arms
            2 ___|_____|___ }
            3 ____|_R_|____ ..... torso
            4 ___|_____|___
            5 __|_______|__ }
            6 _|_________|__} legs


        pauli_tuple (tuple):

            Pauli string featured in the physical rotation. The Pauli string
            is given as a tuple, following Openfermion's data structure for
            the class QubitOperator, such that QubitOperator(<pauli_string>)
            would yield its symbolic expression. For the sketch above,
            <pauli_string> is ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Z'),
            (4, 'Z'), (5, 'Z'), (6, 'Z')).

        angle (float):

            Angle of the Pauli string rotation. The rotation where this angle
            is used is in the center of the circuit, depicted by 'R' in
            the sketch above.

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

        (QubitOperator):

            Symbolic operator obtained by a noisy rotation circuit such
            as the one sketched above.

        (int):

            Algorithmic depth of this subcircuit.
    """

    # splitting the Pauli strings into three lists: arms, legs and torso
    #    which for an odd number of Paulis means to make the Pauli in the
    #    middle the torso and the ones around it arms and legs
    # For an even number of Paulis, the arms are by definition longer.
    # One can see what happens in the picture: the rotation gate is on the
    #    torso qubit, and the parities of the arms and legs are copied on it
    #    with CNot gates

    height = len(pauli_tuple) // 2
    torso = pauli_tuple[height]
    arms = list(pauli_tuple[:height])
    legs = list(pauli_tuple[height + 1:])

    # reverse the arms because the CNots go into an other direction than
    #    in the legs
    arms.reverse()
    # tst_arms.reverse()
    angle = numpy.real(angle)
    # we start with the middle, the Z rotation is placed first
    op = (QubitOperator((), cos(angle)) +
          QubitOperator(((torso[0], 'Z')), sin(angle) * 1.j))

    # always put noise after every operator or idle period
    op = statistical_noise_operation(torso[0], stat_noise, 1) * op

    depth = 1  # circuit depth, it's a good tool

    # unless legs and arms are empty, put the torso as first term, so
    #    the CNot gates can start from the torso
    if arms:
        arms = [torso] + arms

    if legs:
        legs = [torso] + legs

    # the legs may be one segment shorter than the arms
    # Now put the CNot gates and noise in between the arms and legs
    #    like in the sketch above
    for x in range(len(legs) - 1):

        # sandwich the terms we already have with the new CNots and
        #    put a full line of idle noise on the qubit the gate is
        #    controlled on
        # we start with the arms
        op = noisy_cnot_sandwich(
                (
                    statistical_noise_operation(
                            arms[x + 1][0],
                            stat_noise,
                            depth) *
                    op
                ),
                arms[x + 1][0],
                arms[x][0],
                stat_noise,
                gate_noise)

        # always compress the expression to get rid of terms with
        #    coefficients zero

        # update the depth
        depth += 2

        # do the same with the legs
        op = noisy_cnot_sandwich(
            (
                statistical_noise_operation(
                        legs[x + 1][0],
                        stat_noise,
                        depth) *
                op
            ),
            legs[x + 1][0],
            legs[x][0],
            stat_noise,
            gate_noise)

    # correct for when the arms are longer than the legs
    if len(arms) > len(legs):

        op = noisy_cnot_sandwich(
              (
                  statistical_noise_operation(arms[-1][0], stat_noise, depth) *
                  op
              ),
              arms[-1][0],
              arms[-2][0],
              stat_noise,
              gate_noise)

        depth += 2

    # now put noise outside legs
    legs.reverse()
    arms.reverse()

    for x, y in enumerate(legs):

        op = (statistical_noise_operation(y[0], stat_noise, x - 1) *
              op *
              statistical_noise_operation(y[0], stat_noise, x - 1))

        # since the legs's cnots start from the inside,
        # we need one extra noise gate to even it up in case the arms and
        # legs are the same size (see sketch)

    if len(arms) > len(legs):

        for x, y in enumerate(arms[:-1]):
            op = (statistical_noise_operation(y[0], stat_noise, x - 1) *
                  op *
                  statistical_noise_operation(y[0], stat_noise, x - 1))

    else:

        for x, y in enumerate(arms[:-1]):
            op = (statistical_noise_operation(y[0], stat_noise, x) *
                  op *
                  statistical_noise_operation(y[0], stat_noise, x))

        if arms:
            op = (statistical_noise_operation(arms[0][0], stat_noise, 1) *
                  op *
                  statistical_noise_operation(arms[0][0], stat_noise, 1))

            depth += 2

    # now prepare the basis changes
    # dont do anything if the rotation is diagonal
    pstring = [x[1] for x in pauli_tuple]
    if 'X' in pstring or 'Y' in pstring:
        depth += 2

        # if the Pauli string acts on the corresponding qubit with X,
        #    put Y rotations on the outside, a practical Hadamard sandwich

        for x in pauli_tuple:
            if x[1] == 'X':
                op = (statistical_noise_operation(x[0], stat_noise, 1) *
                      (QubitOperator((), .5) +
                       QubitOperator(((x[0], 'Y')), -.5j)) *
                      op *
                      statistical_noise_operation(x[0], stat_noise, 1) *
                      (QubitOperator(()) +
                       QubitOperator(((x[0], 'Y')), 1j)))

                op.compress()

            # sandwich with X rotations to make the basis change to Y
            elif x[1] == 'Y':
                op = (statistical_noise_operation(x[0], stat_noise, 1) *
                      (QubitOperator((), .5) +
                       QubitOperator(((x[0], 'X')), .5j)) *
                      op *
                      statistical_noise_operation(x[0], stat_noise, 1) *
                      (QubitOperator(()) +
                       QubitOperator(((x[0], 'X')), -1j)))

                op.compress()

            # for an action on a qubit with Z, just be idle and put noise
            elif x[1] == 'Z':
                op = (statistical_noise_operation(x[0], stat_noise, 1) *
                      op *
                      statistical_noise_operation(x[0], stat_noise, 1))

    return op, depth


def noisy_state_prep(
        pauli_tuple: tuple,
        stat_noise: List,
        gate_noise: List) -> Tuple[QubitOperator, int]:
    r"""
    Computes a symbolic expressions for two half-noisy versions of a state
    preparation subcircuit measuring a given Pauli string. Two versions of
    this circuit are computed, one where its left-hand side with respect
    to the center is noisy, and one where the noise is on its right-hand side.

    Args:


        
                    * * * * * *    * * * * *
            0 ______*_______  *    * ______*_______
            1 __|___*___|___  *    * __|___*___|___ } arms
            2 ___|__*__|____  *    * ___|__*__|____ }
            3 ____|_*_|_____  *    * ____|_*_|_____ ..... torso
            4 ___|__*__|____  *    * ___|__*__|____
            5 __|___*___|___  *    * __|___*___|___ }
            6 _|____*____|__  *    * _|____*____|__ } legs
                    *         *    *       *
                    * * * * * *    * * * * *
                     noiseless     noiseless



        pauli_tuple (tuple):

            Pauli string measured in the circuit. The Pauli string is given as
            a tuple, following Openfermion's data structure for the class
            QubitOperator, such that QubitOperator(<pauli_tuple>) would yield
            its symbolic expression. In the sketch above, <pauli_tuple> is
            ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Z'), (4, 'Z'), (5, 'Z'),
             (6, 'Z')).

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

        (QubitOperator):

            Symbolic expression for the left-noisy circuit.

        (QubitOperator):

            Symbolic expression of the right-noisy circuit.

        (int):

            Algorithmic depth of one version of the circuit.

    """

    # splitting the Pauli strings into three lists: arms, legs and torso
    #    which for an odd number of Paulis means to make the Pauli in the
    #    middle the torso and the ones around it arms and legs
    # For an even number of Paulis, the arms are by definition longer.
    # One can see what happens in the picture: the rotation gate is on the
    #    torso qubit, and the parities of the arms and legs are copied on it
    #    with CNot gates

    height = len(pauli_tuple) // 2
    torso = pauli_tuple[height]
    arms = list(pauli_tuple[:height])
    legs = list(pauli_tuple[height + 1:])

    # reverse the arms because the CNots go into an other direction than
    #    in the legs
    arms.reverse()

    depth = 1  # circuit depth, it's a good tool

    lop = statistical_noise_operation(torso[0], stat_noise, 1)
    rop = QubitOperator(())

    # unless legs and arms are empty, put the torso as first term, so
    #    the CNot gates can start from the torso
    if arms:
        arms = [torso] + arms

    if legs:
        legs = [torso] + legs

    # the legs may be one segment shorter than the arms
    # Now put the CNot gates and noise in between the arms and legs
    #    like in the sketch above
    for x in range(len(legs) - 1):

        # sandwich the terms we already have with the new CNots and
        #    put a full line of idle noise on the qubit the gate is
        #    controlled on
        # we start with the arms
        lop = half_noisy_cnot_sandwich(
               (
                    statistical_noise_operation(
                        arms[x + 1][0],
                        stat_noise,
                        (depth + 1) // 2) *
                    lop
                ),
               arms[x + 1][0],
               arms[x][0],
               stat_noise,
               gate_noise,
               True)

        rop = half_noisy_cnot_sandwich(
            (
               rop *
               statistical_noise_operation(
                    arms[x + 1][0],
                    stat_noise,
                    (depth - 1) // 2)
            ),
            arms[x + 1][0],
            arms[x][0],
            stat_noise,
            gate_noise,
            False)

        # always compress the expression to get rid of terms with
        #    coefficients zero

        # update the depth
        depth += 2

        # do the same with the legs
        lop = half_noisy_cnot_sandwich(
            (
               statistical_noise_operation(
                    legs[x + 1][0],
                    stat_noise,
                    (depth + 1) // 2) *
               lop
            ),
            legs[x + 1][0],
            legs[x][0],
            stat_noise,
            gate_noise,
            True)
        rop = half_noisy_cnot_sandwich(
            (
               statistical_noise_operation(
                    legs[x + 1][0],
                    stat_noise,
                    (depth - 1) // 2) *
               rop
            ),
            legs[x + 1][0],
            legs[x][0],
            stat_noise,
            gate_noise,
            False)

    # correct for when the arms are longer than the legs
    if len(arms) > len(legs):

        lop = half_noisy_cnot_sandwich(
            (
               statistical_noise_operation(
                    arms[-1][0],
                    stat_noise,
                    (depth + 1) // 2) *
               lop
            ),
            arms[-1][0],
            arms[-2][0],
            stat_noise,
            gate_noise,
            True)

        rop = half_noisy_cnot_sandwich(
            (
               statistical_noise_operation(
                    arms[-1][0],
                    stat_noise,
                    ((depth - 1) // 2)) *
               rop
            ),
            arms[-1][0],
            arms[-2][0],
            stat_noise,
            gate_noise,
            False)

        depth += 2

    # now put noise outside legs
    legs.reverse()
    arms.reverse()

    for x, y in enumerate(legs):

        lop = statistical_noise_operation(y[0], stat_noise, x - 1) * lop
        rop = rop * statistical_noise_operation(y[0], stat_noise, x - 1)

        # since the legs's cnots start from the inside,
        # we need one extra noise gate to even it up in case the arms and
        # legs are the same size (see sketch)

    if len(arms) > len(legs):

        for x, y in enumerate(arms[:-1]):
            lop = statistical_noise_operation(y[0], stat_noise, x - 1) * lop
            rop = rop * statistical_noise_operation(y[0], stat_noise, x - 1)

    else:

        for x, y in enumerate(arms[:-1]):
            lop = statistical_noise_operation(y[0], stat_noise, x) * lop
            rop = rop * statistical_noise_operation(y[0], stat_noise, x)

        if arms:
            lop = statistical_noise_operation(arms[0][0], stat_noise, 1) * lop
            rop = rop * statistical_noise_operation(arms[0][0], stat_noise, 1)

            depth += 2

    # now prepare the basis changes
    # dont do anything if the rotation is diagonal
    pstring = [x[1] for x in pauli_tuple]
    if 'X' in pstring or 'Y' in pstring:
        depth += 2

        # if the Pauli string acts on the corresponding qubit with X,
        #    put Y rotations on the outside, a practical Hadamard sandwich

        for x in pauli_tuple:
            if x[1] == 'X':
                lop = (statistical_noise_operation(x[0], stat_noise, 1) *
                       (QubitOperator((), .5) +
                        QubitOperator(((x[0], 'Y')), -.5j)) *
                       lop *
                       (QubitOperator(()) +
                        QubitOperator(((x[0], 'Y')), 1j)))

                rop = ((QubitOperator((), .5) +
                        QubitOperator(((x[0], 'Y')), -.5j)) *
                       rop *
                       statistical_noise_operation(x[0], stat_noise, 1) *
                       (QubitOperator(()) +
                        QubitOperator(((x[0], 'Y')), 1j)))

                # ltst[int(x[1:])] = '|Y' + ltst[int(x[1:])] + '+Y'

                lop.compress()
                rop.compress()

            # sandwich with X rotations to make the basis change to Y
            if x[1] == 'Y':
                lop = (statistical_noise_operation(x[0], stat_noise, 1) *
                       (QubitOperator((), .5) +
                        QubitOperator(((x[0], 'X')), .5j)) *
                       lop *
                       (QubitOperator(()) +
                        QubitOperator(((x[0], 'X')), -1j)))
                rop = ((QubitOperator((), .5) +
                        QubitOperator(((x[0], 'X')), .5j)) *
                       rop *
                       statistical_noise_operation(x[0], stat_noise, 1) *
                       (QubitOperator(()) +
                        QubitOperator(((x[0], 'X')), -1j)))

                # ltst[int(x[1:])] = '|X' + ltst[int(x[1:])] + '+X'

                lop.compress()
                rop.compress()

            # for an action on a qubit with Z, just be idle and put noise
            if x[1] == 'Z':
                lop = statistical_noise_operation(x[0], stat_noise, 1) * lop
                rop = rop * statistical_noise_operation(x[0], stat_noise, 1)

    return lop, rop, depth

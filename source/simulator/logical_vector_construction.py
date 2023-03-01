"""Functions to build logical state."""
from typing import List, Sequence, Union, Optional, Tuple
from copy import deepcopy
import numpy
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE


class SimulatorError(Exception):
    r"""Simulator error class."""

    def __init__(self, message):
        """
        Throw custom errors connected to the simulator.

        Args:

            message (str):

                Custom error message.
        """
        Exception.__init__(self, message)


def check_for_imaginary_coeff(
        op_list: List,
        msg: str,
        thresh: float = EQ_TOLERANCE) -> None:
    """
    Checks a list of Pauli strings and raises an
    error if a complex number is found in any of the coefficients.

    Args:

        stabilizer_list (list):

            List of symbolic expressions of type QubitOperator. The
            coefficient of each list item will be checked.

        msg (str):

            Custom error message displayed upon detection
            of a nonzero imaginary part.

        thresh (float):

            Threshold above the value of which an imaginary part will be
            deemed as nonzero. Set to OpenFermion's EQ_TOLERANCE by default.

    Returns:

        (None)

    Raises:

        (SimulatorError):

            Error message in case an imaginary coefficient is detected.
    """
    for op in op_list:
        if abs(numpy.imag(list(op.terms.values())[0])) >= thresh:
            raise SimulatorError(msg)


def check_for_identity(
        op_list: List,
        msg: str) -> None:
    """
    This function checks a list of Pauli strings and raises an error
    if one is proportional to the identity.

    Args:

        op_list (list):

            List of symbolic expressions of type QubitOperator. Every item
            of the list is going to be checked.

        msg (str):

            Custom error message that is output upon the detection of
            the identity.

    Returns:

        (None)

    Raises:

        (SimulatorError):

            Error message if a Pauli string is proportional to the identity.
    """
    for op in op_list:
        if list(op.terms.keys())[0] == ():
            raise SimulatorError(msg)


def fix_single_term(
        term: QubitOperator,
        position: int,
        fixed_op: str,
        other_op: str,
        stabilizer: QubitOperator) -> QubitOperator:
    """
    This function determines whether a given weighted Pauli string needs
    to be corrected by multiplication with a given stabilizer. If the string's
    Pauli operator on a specified position has one of two specified types, the
    correction is applied.

    Args:

        term (QubitOperator):

            Weighted Pauli string; a single Pauli string with any coefficient,
            that is going to be checked for correction.

        position (int):

            Position on which the Pauli string is checked for a certain type
            of Pauli operator; label of the qubit on which this operator acts.

        fixed_op (str):

            One type of Pauli operator ('X', 'Y' or 'Z') that will trigger a
            correction of <term> by multiplication with <stabilizer>, if found
            acting on the specified qubit labeled <position>.

        other_op (str):

            The other type of Pauli operator ('X', 'Y' or 'Z') that will
            trigger a correction of <term> by multiplication with
            <stabilizer>, if found acting on the specified qubit labeled
            <position>.

        stabilizer (QubitOperator):

            Stabilizer that is multiplied to <term> in case of a correction.

    Returns:

        (QubitOperator):

            Weighted Pauli string <term>, possibly corrected.

    """

    pauli_tuple = list(term.terms.keys())[0]
    if (position, fixed_op) in pauli_tuple or (
            position, other_op) in pauli_tuple:

        return term * stabilizer

    else:
        return term


def reduce_to_logical(
        operator: QubitOperator,
        syndromes: numpy.ndarray,
        stabilizer_list: Sequence,
        qubit_order: Sequence,
        fixed_positions: Sequence,
        fixed_ops: str,
        other_ops: str) -> QubitOperator:
    r"""
    Reduces a symbolical operator to its logical representation according to
    a provided system of stabilizers. This process gradually corrects every
    Pauli string in the expression and removes qubits tapering off stabilizer
    conditions. For the corrections to be effective, the stabilizer system is
    provided in a form that fits <fix_single_term>. A pattern of syndromes is
    provided, that is updated with a single Pauli string from the expression.
    The assumption is that all of the expressions Pauli strings would cause
    the same syndrome pattern.

    Args:

        operator (QubitOperator):

            Symbolic expression of the operator that is to be reduced to
            its logical representation.

        syndromes (numpy.ndarray):

            Array of integers 0/1, indicating the list of syndromes.
            The tapering of stabilizers is done with respect to the
            stabilizer generators in <stabilizer_list> multiplied with +1/-1
            according to the entries of this array, after it has been altered
            by syndromes caused by the expression <operator>.

        stabilizer_list (list):

            List of weighted Pauli strings of the type QubitOperator
            signifying stabilizer generators of the system's code space,
            regardless of which syndrome space we might be in before starting
            the tapering procedure.

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

            String of characters  'X', 'Y' and 'Z'  indicating one of the
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

        (QubitOperator):

            Logical representation of the operator <operator>.

    """

    # sample first term and obtain the syndromes
    sample_term = QubitOperator(list(operator.terms.keys())[0])

    # take the first term out and multiply it to every stabilizer
    # checking whether the product has an imaginary coefficient
    # if so, terms anticommute with stabilizers.
    # which means that the terms and the stabilizer anticommute.
    for i, x in enumerate(stabilizer_list):
        if abs(list((sample_term * x).terms.values())[0].imag) > EQ_TOLERANCE:
            syndromes[i] = (syndromes[i] + 1) % 2

    # going through the list of stabilizers selecting one stab at a time
    for i in range(len(fixed_positions)):
        new_terms = QubitOperator()
        # fix the representation of every term in the given operator
        for term in operator:
            new_terms += fix_single_term(
                term, fixed_positions[i], fixed_ops[i], other_ops[i],
                stabilizer_list[i] * ((-1.) ** syndromes[i]))

        operator = new_terms

    # Remove the qubits
    #    going through all the fixed terms
    skimmed_operator = QubitOperator()
    for term, coef in operator.terms.items():

        # if the term is the identity, remove nothing and skip this term
        if term == ():
            skimmed_operator += QubitOperator((), coef)
            continue

        tap_tpls = []

        # looking through the Pauli operators of one Pauli string
        for p in term:

            # unless the position is to be removed, change it to the
            #    new position given in the qubit_order list
            if qubit_order[p[0]] != 'rm':
                tap_tpls.append((qubit_order[p[0]], p[1]))

        skimmed_operator += QubitOperator(tuple(tap_tpls), coef)

    return skimmed_operator


def anticom_syndr(
        op: QubitOperator,
        stab_list: Sequence) -> numpy.ndarray:
    r"""
    Computes the syndrome pattern that a given Pauli string creates with
    respect to a list of stabilizer generators. A syndrome is created
    when the Pauli operator anticommutes with a stabilizer generator.

    Args:

        op (QubitOperator):

            Symbolic expression of the input Pauli string.

        stab_list (list):

            List of symbolic expressions of type QubitOperator
            signifying the stabilizer generators with respect to which
            <op> creates syndromes.

    Returns:

        (numpy.ndarray):

            One-dimensional array of integer entries 0 / 1, where 1 on
            the <n>-th position indicates that <op> anticommutes with
            <stab_list[n]>.
    """
    outp = []
    op = QubitOperator(list(op.terms.keys())[0])
    for x in stab_list:

        if abs(list((op * x).terms.values())[0].imag) > EQ_TOLERANCE:
            outp += [1]

        else:
            outp += [0]

    return numpy.array(outp, dtype=int)


def fix_single_term_plus(
        term: QubitOperator,
        position: int,
        fixed_op: str,
        other_op: str,
        stabilizer: QubitOperator) -> QubitOperator:
    """
    Same as <fix_single_term>, but additionally outputs an integer flag
    0 or 1, where 1 indicates that the correction has been applied.
    Args:

        term (QubitOperator):

            Weighted Pauli string; a single Pauli string with any coefficient,
            that is going to be checked for correction.

        position (int):

            Position on which the Pauli string is checked for a certain type
            of Pauli operator; label of the qubit on which this operator acts.

        fixed_op (str):

            One type of Pauli operator ('X', 'Y' or 'Z') that will trigger a
            correction of <term> by multiplication with <stabilizer>, if found
            acting on the specified qubit labeled <position>.

        other_op (str):

            The other type of Pauli operator ('X', 'Y' or 'Z') that will
            trigger a correction of <term> by multiplication with
            <stabilizer>, if found acting on the specified qubit labeled
            <position>.

        stabilizer (QubitOperator):

            Stabilizer that is multiplied to <term> in case of a correction.

    Returns:

        (QubitOperator):

            Weighted Pauli string <term>, possibly corrected.

        (int):

            Integer 0 or 1, where 1 indicates that <term> has been corrected
            by multiplication with <stabilizer>.

    """

    pauli_tuple = list(term.terms.keys())[0]
    if (position, fixed_op) in pauli_tuple or (
            position, other_op) in pauli_tuple:

        return term * stabilizer, 1

    else:
        return term, 0


def reduce_to_logical_plus(
        operator: QubitOperator,
        stabilizer_list: Sequence,
        qubit_order: Sequence,
        fixed_positions: Sequence,
        fixed_ops: str,
        other_ops: str) -> tuple:
    r"""

    Similar to <reduce_to_logical>, but this function does not take in a
    syndrome list, and instead outputs the generator dependence of every Pauli
    string in the logical operator. That is, it records how each such Pauli
    string depends on sign changes of the stabilizer generators. Since
    different physical Pauli strings can map to the same logical Pauli string
    with different generator dependencies, this routine outputs the logical
    operator as a list of its Pauli strings, along with a list of their
    dependencies.

  Args:

        operator (QubitOperator):

            Symbolic expression of the operator that is to be reduced to
            its logical representation.

        stabilizer_list (list):

            List of weighted Pauli strings of the type QubitOperator
            signifying stabilizer generators of the system's code space.

        qubit_order (list):

            List containing integers and flags 'rm'. This list indicates which
            qubits are removed after the correction, and how the rest of the
            qubits are relabeled. After the corrections by <fix_single_term>,
            the n-th qubit is relabeled as <qubit_order[n]>, or removed if
            <qubit_order[n]> = 'rm'.

        fixed_positions (list):

            List of integers, indicating on which positions
            <fix_single_term_plus> checks for corrections. The encounter
            of <fixed_ops[n]> or <other_ops[n]> on position
            <fixed_positions[n]> triggers the multiplication of the
            <n>-th stabilizer generator.

        fixed_ops (str):

            String of characters  'X', 'Y' and 'Z'  indicating one of the
            possible Pauli types against which <fix_single_term_plus> checks
            on positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

        other_ops (str):

            String of characters  'X', 'Y' and 'Z' indicating the other type
            of Pauli operators against which <fix_single_term_plus> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

    Returns:

        (list):

            List of symbolic expressions of type QubitOperator,
            signifying the logical representation of the operator <operator>.
            In the code space, one would just add all items in this list,
            in syndrome spaces, one would need to correct the signs of every
            list item according to the stabilizer dependence.

        (list):

            List of one-dimensional numpy arrays, filled with integers 0 and
            1, denoting generator dependencies. The entry 1 on the n-th
            position of the m-th array indicates that the m-th logical Pauli
            string in the list of outputs was corrected by multiplication
            with <stabilizer_list[n]>.

    """
    op_list = list(operator)
    dep_list = [numpy.zeros(len(fixed_positions), dtype=int)] * len(op_list)
    # going through the list of stabilizers selecting one stab at a time
    for x in range(len(fixed_positions)):

        # fix the representation of every term in the given operator list
        for y in range(len(op_list)):
            op_list[y], dep_list[y][x] = fix_single_term_plus(
                op_list[y],
                fixed_positions[x],
                fixed_ops[x],
                other_ops[x],
                stabilizer_list[x])

    # Remove the qubits
    #    going through all the fixed terms
    for x in range(len(op_list)):
        term, coeff = list(op_list[x].terms.items())[0]

        # if the term is the identity, remove nothing and skip this term
        if term == ():
            op_list[x] = QubitOperator((), coeff)

        else:

            tap_tpls = []

            # looking through the Pauli operators of one Pauli string
            for p in term:

                # unless the position is to be removed, change it to the
                #    new position given in the qubit_order list
                if qubit_order[p[0]] != 'rm':
                    tap_tpls.append((qubit_order[p[0]], p[1]))

            op_list[x] = QubitOperator(tuple(tap_tpls), coeff)

    return op_list, dep_list


def generator_dependence(
        op: QubitOperator,
        gens: list,
        fixed_positions: list,
        fixed_ops: str,
        other_ops: str) -> Tuple[numpy.ndarray, int]:
    r"""
    Similar to <reduce_to_logical_plus>, but it only takes in one weighted
    Pauli string and does not remove qubits.
    This routine takes a symbolic expression proportional to a Pauli string
    and fixes its representation according to <fix_single_term>, while
    outputting its stabilizer generator dependence, that is it indicates
    whether the Pauli string has been corrected by  multiplication with
    either stabilizer generator during <fix_single_term>.

    Args:

        op (QubitOperator):

            Symbolic expression of the weighted Pauli string that is to be
            fixed.

        gens (list):

            List of weighted Pauli strings of the type QubitOperator
            signifying stabilizer generators of the system's code space.

        fixed_positions (list):

            List of integers, indicating on which positions
            <fix_single_term_plus> checks for corrections. The encounter
            of <fixed_ops[n]> or <other_ops[n]> on position
            <fixed_positions[n]> triggers the multiplication of the
            <n>-th stabilizer generator.

        fixed_ops (str):

            String of characters  'X', 'Y' and 'Z'  indicating one of the
            possible Pauli types against which <fix_single_term_plus> checks
            on positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

        other_ops (str):

            String of characters  'X', 'Y' and 'Z' indicating the other type
            of Pauli operators against which <fix_single_term_plus> checks on
            positions specified in <fixed_positions>. The <n>-th character
            corresponds to the Pauli type that is checked against on position
            <fixed_positions[n]>.

    Returns:

        (QubitOperator):

            Weighted Pauli string <op> in fixed form, where it does not have
            Pauli operators of types <fixed_ops[n]> and <other_ops[n]> on
            qubit <fixed_positions[n]> for all integers n within the length
            of <gens>.

        (numpy.ndarray):

            List of integers 0 and 1 signifying the generator dependence of
            the fixed Pauli string, where 1 at position n of the list means
            that the n-th stabilizer generator has been multiplied with the
            operator during <fix_single_term_plus>.
    """

    n_log_qubits = len(fixed_ops)
    gen_dep = numpy.zeros(n_log_qubits, dtype=int)

    for x in range(n_log_qubits):
        op, gen_dep[x] = fix_single_term_plus(
            op,
            fixed_positions[x],
            fixed_ops[x],
            other_ops[x],
            gens[x])

    return op, gen_dep

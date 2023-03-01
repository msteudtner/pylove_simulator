from openfermion import QubitOperator


class NoiseError(Exception):
    """Noise model error class."""

    def __init__(self, message):
        """
        Throw custom errors connected to noise.

        Args:

            message (str):

                Custom error message.
        """
        Exception.__init__(self, message)


def single_qubit_dephasing(
        rate: float,
        dephasing_operator='Z') -> list:
    """
    Dephasing channel for a single qubit. The strength of the noise can be set
    by the noise rate p, mapping a single-qubit density matrix Rho to the state
                        (1 - p) Rho + p C(Rho),
    where C(.) is the complete dephasing channel, i.e. C(Rho) completely
    removes the off-diagonal elements of Rho in case the dephasing operator is
    'Z'.

    Args:

        rate (float):

            Noise rate of the dephasing channel.

        dephasing_operator (str) [optional]:

            Kraus operator of the dephasing channel. 'Z' by default.

    Returns:

        (list):

            List characterizing the noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli operators (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli operators have the index 0 as a placeholder
            for the proper label of the qubit the noise acts on.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli string.

    Raises:

        (NoiseError):

            Custom error in case the noise rate is out of bounds.

    """

    if rate > 1 or rate < 0:
        raise NoiseError('Error rate expected to be between 0 and 1,' +
                         ' but is ' + str(rate) + '.')

    return [[QubitOperator(()), QubitOperator(dephasing_operator + '0')],
            [1 - rate / 2, rate / 2]]


def single_qubit_depolarizing(rate: float) -> list:
    """
    Depolarizing channel for a single qubit. The strength of the noise can be
    set by the noise rate p, mapping a single-qubit density matrix Rho to the
    state
                            (1 - p) Rho + p I / 2.

    Args:

        rate (float):

            Noise rate of the depolarizing channel.

    Returns:

        (list):

            List characterizing the noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli operators (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli operators have the index 0 as a placeholder
            for the proper label of the qubit the noise acts on.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli string.

    Raises:

        (NoiseError):

            Custom error in case the noise rate is out of bounds.
    """

    if rate > 1 or rate < 0:
        raise NoiseError('Error rate expected to be between 0 and 1,' +
                         ' but is ' + str(rate) + '.')

    return [[QubitOperator(()),
             QubitOperator('X0'),
             QubitOperator('Y0'),
             QubitOperator('Z0')],
            [1 - 3 * rate / 4, rate / 4, rate / 4, rate / 4]]

def parallel_single_qubit_depolarizing(rate: float) -> list:
    """
    A pair of single-qubit depolarizing channels acting in parallel.

        Args:
              rate (float): Noise rate of the channel.

        Returns:
              (list): noise format for the simulator
    """

    if rate > 1 or rate < 0:
        raise NoiseError('Error rate expected to be between 0 and 1,' +
                         ' but is ' + str(rate))

    return [[QubitOperator(()),
             QubitOperator('X0'),
             QubitOperator('Y0'),
             QubitOperator('Z0'),
             QubitOperator('X1'),
             QubitOperator('Y1'),
             QubitOperator('Z1'),
             QubitOperator('X0 X1'),
             QubitOperator('X0 Y1'),
             QubitOperator('X0 Z1'),
             QubitOperator('Y0 X1'),
             QubitOperator('Y0 Y1'),
             QubitOperator('Y0 Z1'),
             QubitOperator('Z0 X1'),
             QubitOperator('Z0 Y1'),
             QubitOperator('Z0 Z1'),],
            [(1 - 3 * rate / 4)**2, 
            (rate / 4)*(1 - 3 * rate / 4), 
            (rate / 4)*(1 - 3 * rate / 4), 
            (rate / 4)*(1 - 3 * rate / 4),
            (rate / 4)*(1 - 3 * rate / 4),
            (rate / 4)*(1 - 3 * rate / 4),
            (rate / 4)*(1 - 3 * rate / 4),
            (rate / 4)**2,
            (rate / 4)**2,
            (rate / 4)**2,
            (rate / 4)**2,
            (rate / 4)**2,
            (rate / 4)**2,
            (rate / 4)**2,
            (rate / 4)**2,
            (rate / 4)**2,]]


def two_qubit_depolarizing(rate: float) -> list:
    """
    Two-qubit depolarizing error, a concatenation of
    <single_qubit_depolarizing> on two qubits at once.

    Args:

        rate (float):

            Noise rate of depolarizing channels on each qubit.

    Returns:

        (list):

            List characterizing the noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli strings (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli strings itself are on two qubits, where
            the integers 0 and 1 function as placeholders for the labels of
            control and target qubit, respectively.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli string.

    Raises:

        (NoiseError):

            Custom error in case the noise rate is out of bounds.
    """

    if rate > 1 or rate < 0:
        raise NoiseError('Error rate expected to be between 0 and 1,' +
                         ' but is ' + str(rate) + '.')

    return [[QubitOperator(()),
             QubitOperator('X0'),
             QubitOperator('Y0'),
             QubitOperator('Z0'),
             QubitOperator('X1'),
             QubitOperator('Y1'),
             QubitOperator('Z1'),
             QubitOperator('X0 X1'),
             QubitOperator('Y0 X1'),
             QubitOperator('Z0 X1'),
             QubitOperator('X0 Y1'),
             QubitOperator('Y0 Y1'),
             QubitOperator('Z0 Y1'),
             QubitOperator('X0 Z1'),
             QubitOperator('Y0 Z1'),
             QubitOperator('Z0 Z1')],
            [(1 - 3 * rate / 4) ** 2] +
            6 * [(rate / 4) * (1 - 3 * rate / 4)] +
            9 * [(rate / 4) ** 2]]


def two_qubit_dephasing(
        rate: float,
        dephasing_operator='Z') -> list:
    """
    Two-qubit dephasing error, a concatenation of
    <single_qubit_dephasing> on two qubits at once.

    Args:

        rate (float):

            Noise rate of the dephasing channels on each qubits.

        dephasing_operator (str) [optional]:

            Kraus operator of the dephasing channels. 'Z' by default.

    Returns:

        (list):

            List characterizing the noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli strings (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli strings itself are on two qubits, where
            the integers 0 and 1 function as placeholders for the labels of
            control and target qubit, respectively.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli string.

    Raises:

        (NoiseError):

            Custom error in case the noise rate is out of bounds.
    """

    if rate > 1 or rate < 0:
        raise NoiseError('Error rate expected to be between 0 and 1,' +
                         ' but is ' + str(rate))

    return [[QubitOperator(()),
             QubitOperator(dephasing_operator + '0'),
             QubitOperator(dephasing_operator + '1'),
             QubitOperator(dephasing_operator + '0 ' +
                           dephasing_operator + '1')],
            [(1 - rate / 2) ** 2] + 2 * [rate / 2 * (1 - rate / 2)]
            + [.25 * rate ** 2]]


def noiseless() -> list:
    """
    Noiseless model.

    Returns:

        (list):

            List characterizing the noise model. It has the form
            [[#1], [#2]], holding two sublists [#1] and [#2]:

            [#1] is a list of Pauli strings (QubitOperators) including
            the identity, that together with their respective statistical
            weight in the next sublist form the error channel's Kraus
            operators. The Pauli strings itself are either on two qubits,
            where the integers 0 and 1 function as placeholders for the labels
            of control and target qubit, respectively; or on one qubit only,
            the placeholder index of which is 0.

            [#2] is a list holding float numbers corresponding to the
            statistical weights of their respective Pauli string.
    """
    return [[QubitOperator(()), ], [1, ]]


from openfermion import FermionOperator, QubitOperator, chemist_ordered
from copy import deepcopy

Aop = ({
    (1, 3): QubitOperator('X0 Z2 X3'),
    (3, 5): QubitOperator('X2 X4'),
    (1, 5): QubitOperator('Z0 X1 Z4 X5', -1),
    (5, 6): QubitOperator('Y4 Y10'),
    (3, 4): QubitOperator('Y2 Y8', -1),
    (1, 2): QubitOperator('Y0 Y6', -1),
    (2, 4): QubitOperator('X6 Z8 X9', -1),
    (4, 6): QubitOperator('X8 X10', -1),
    (2, 6): QubitOperator('Z6 X7 Z10 X11'),

    (7, 9): QubitOperator('X12 Z14 X15'),
    (9, 11): QubitOperator('X14 X16'),
    (7, 11): QubitOperator('Z12 X13 Z16 X17', -1),
    (11, 12): QubitOperator('Y16 Y22'),
    (9, 10): QubitOperator('Y14 Y20', -1),
    (7, 8): QubitOperator('Y12 Y18', -1),
    (8, 10): QubitOperator('X18 Z20 X21', -1),
    (10, 12): QubitOperator('X20 X22', -1),
    (8, 12): QubitOperator('Z18 X19 Z22 X23')

})


Aop_d = ({
    (1, 2): QubitOperator('Z0 Y1 Z6 Y7'),
    (3, 4): QubitOperator('Z2 Y3 Z8 Y9'),
    (5, 6): QubitOperator('Z4 Y5 Z10 Y11', -1),
    
    (7, 8): QubitOperator('Z12 Y13 Z18 Y19'),
    (9, 10): QubitOperator('Z18 Y15 Z20 Y21'),
    (11, 12): QubitOperator('Z16 Y17 Z22 Y23', -1)

})

Bop = ({
    1: QubitOperator('Z0 Z1'),
    3: QubitOperator('Z2 Z3'),
    5: QubitOperator('Z4 Z5'),
    2: QubitOperator('Z6 Z7'),
    4: QubitOperator('Z8 Z9'),
    6: QubitOperator('Z10 Z11'),
    
    7: QubitOperator('Z12 Z13'),
    8: QubitOperator('Z14 Z15'),
    9: QubitOperator('Z16 Z17'),
    10: QubitOperator('Z18 Z19'),
    11: QubitOperator('Z20 Z21'),
    12: QubitOperator('Z22 Z23')
})

Aop.update({(1, 4): Aop[(1, 2)] * Aop[(2, 4)] * 1j})
Aop.update({(1, 6): Aop[(1, 3)] * Aop[(3, 5)] * Aop[(5, 6)] * (-1)})
Aop.update({(3, 6): Aop[(3, 5)] * Aop[(5, 6)] * 1j})
Aop.update({(2, 5): Aop[(1, 5)] * Aop[(1, 2)] * 1j})
Aop.update({(4, 5): Aop[(1, 5)] * Aop[(1, 4)] * 1j})
Aop.update({(2, 3): Aop[(1, 3)] * Aop[(1, 2)] * 1j})

Aop.update({(7, 10): Aop[(7, 8)] * Aop[(8, 10)] * 1j})
Aop.update({(7, 12): Aop[(7, 9)] * Aop[(9, 11)] * Aop[(11, 12)] * (-1)})
Aop.update({(9, 12): Aop[(9, 11)] * Aop[(11, 12)] * 1j})
Aop.update({(8, 11): Aop[(7, 11)] * Aop[(7, 8)] * 1j})
Aop.update({(10, 11): Aop[(7, 11)] * Aop[(7, 10)] * 1j})
Aop.update({(8, 9): Aop[(7, 9)] * Aop[(7, 8)] * 1j})

stabs = ([
    Aop[(1, 3)] * Aop_d[(3, 4)] * Aop[(2, 4)] * Aop[(1, 2)],
    Aop[(3, 5)] * Aop[(5, 6)] * Aop[(4, 6)] * Aop[(3, 4)],
    Aop[(1, 2)] * Aop_d[(1, 2)],
    Aop[(3, 4)] * Aop_d[(3, 4)],
    Aop[(5, 6)] * Aop_d[(5, 6)],
    Aop[(1, 3)] * Aop[(3, 5)] * Aop[(1, 5)] * 1j,
    Aop[(2, 4)] * Aop[(4, 6)] * Aop[(2, 6)] * 1j,
    
    Aop[(7, 9)] * Aop_d[(9, 10)] * Aop[(8, 10)] * Aop[(7, 8)],
    Aop[(9, 11)] * Aop[(11, 12)] * Aop[(10, 12)] * Aop[(9, 10)],
    Aop[(7, 8)] * Aop_d[(7, 8)],
    Aop[(9, 10)] * Aop_d[(9, 10)],
    Aop[(11, 12)] * Aop_d[(11, 12)],
    Aop[(7, 9)] * Aop[(9, 11)] * Aop[(7, 11)] * 1j,
    Aop[(8, 10)] * Aop[(10, 12)] * Aop[(8, 12)] * 1j
])

def fermapping(terms):
    """
    Generate mapping.

    Args:
            terms (FermionicOperator): Fermionic Hamiltonian.

    Return:
            qubit_ops (QubitOperator): Mapped Hamiltonian.
    """
    fermion_ops = deepcopy(terms)
    fermion_ops = chemist_ordered(fermion_ops)
    fermion_ops = list(fermion_ops.terms.items())
    qubit_ops = QubitOperator()

    for x in fermion_ops:
        oneterm = QubitOperator((), x[1])

        for y in range(len(x[0]) // 2):

            if x[0][2 * y][1] == 1 and x[0][2 * y + 1][1] == 0:
                oneterm = oneterm * \
                    one_body(x[0][2 * y][0], x[0][2 * y + 1][0])

            else:
                print("Ordering has failed \n")

        qubit_ops += oneterm

    return qubit_ops


def one_body(a_index, b_index):
    """Calculate one body operators."""
    a_index += 1
    b_index += 1

    if a_index == b_index:

        return .5 * (QubitOperator(()) - Bop[a_index])

    else:

        if a_index < b_index:
            return (.25j * (QubitOperator(()) - Bop[a_index]) *
                    Aop[(a_index, b_index)] *
                    (QubitOperator(()) - Bop[b_index]))

        else:
            return (.25j * (QubitOperator(()) - Bop[a_index]) *
                    Aop[(b_index, a_index)] * (-1) *
                    (QubitOperator(()) - Bop[b_index]))
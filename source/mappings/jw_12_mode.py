"""
"""

from openfermion.ops import QubitOperator
from openfermion.transforms import chemist_ordered
from copy import deepcopy

Aop = ({
    (0, 1): QubitOperator('Y0 X1'),#Horizontal edge ops
    (6,7): QubitOperator('Y6 X7'),#
    (2,3): QubitOperator('Y2 X3'),#
    (8,9): QubitOperator('Y8 X9'),#
    (4,5): QubitOperator('Y4 X5'),#
    (10,11): QubitOperator('Y10 X11'),#
    (0,2): QubitOperator('Y0 Z1 X2'),#Vertical edge ops
    (2,4): QubitOperator('Y2 Z3 X4'),#
    (1,3): QubitOperator('Y1 Z2 X3'),#
    (3,5): QubitOperator('Y3 Z4 X5'),
    (6,8): QubitOperator('Y6 Z7 X8'),
    (8,10): QubitOperator('Y8 Z9 X10'),
    (7,9): QubitOperator('Y7 Z8 X9'),
    (9,11): QubitOperator('Y9 Z10 X11'),
    (5,6): QubitOperator('Y5 X6')
})
Aop[(1,6)] = Aop[(1,3)]*Aop[(3,5)]*Aop[(5,6)]
Aop[(3,8)] = Aop[(3,5)]*Aop[(5,6)]*Aop[(6,8)]
Aop[(5,10)] = Aop[(5,6)]*Aop[(6,8)]*Aop[(8,10)]

Bop = ({
    0: QubitOperator('Z0'),
    1: QubitOperator('Z1'),
    2: QubitOperator('Z2'),
    3: QubitOperator('Z3'),
    4: QubitOperator('Z4'),
    5: QubitOperator('Z5'),
    6: QubitOperator('Z6'),
    7: QubitOperator('Z7'),
    8: QubitOperator('Z8'),
    9: QubitOperator('Z9'),
    10: QubitOperator('Z10'),
    11: QubitOperator('Z11'),
    })

stabs = [QubitOperator('Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11')]



def fermapping(terms):
    """
    Generate mapping.

    Args:
            terms (FermionicOperator): Fermionic Hamiltonian.

    Return:
            qubit_ops (QubitOperator): Mapped Hamiltonian.
    """
    fermion_ops = deepcopy(terms)
    fermion_ops = list(chemist_ordered(fermion_ops).terms.items())
    qubit_ops = QubitOperator()

    for x in fermion_ops:
        #print(x)
        oneterm = QubitOperator((), x[1])

        for y in range(len(x[0]) // 2):
            # iterate over creation-annihilation pairs
            if x[0][2 * y][1] == 1 and x[0][2 * y + 1][1] == 0:
                oneterm = oneterm * \
                    one_body(x[0][2 * y][0], x[0][2 * y + 1][0])
            else:
                print("Ordering has failed \n")

        qubit_ops += oneterm

    return qubit_ops


def one_body(a_index, b_index):
    """Calculate one body operators."""
    #a_index += 1
    #b_index += 1
    #print(f'{a_index}, {b_index}')

    if a_index == b_index:
        # Number operator - same site index
        return .5 * (QubitOperator(()) - Bop[a_index])

    else:
        # Hopping operator - different site index
        if a_index < b_index:
            return (.25j * (QubitOperator(()) - Bop[a_index]) *
                    Aop[(a_index, b_index)] * (QubitOperator(()) -
                                               Bop[b_index]))

        else:
            return (.25j * (QubitOperator(()) - Bop[a_index]) *
                    Aop[(b_index, a_index)] * (-1) * (QubitOperator(()) -
                                                      Bop[b_index]))

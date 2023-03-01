"""
"""

from openfermion.ops import QubitOperator
from openfermion.transforms import chemist_ordered
from copy import deepcopy

qubits=11

Aop = ({
    (0,1): QubitOperator('X0 Y1 X9'), # horizontal
    (1,2): QubitOperator('X1 Y2'),
    (3,4): (-1)*QubitOperator('Y3 X4 X9'),
    (4,5): (-1)*QubitOperator('Y4 X5 X10'),
    (6,7): QubitOperator('X6 Y7'),
    (7,8): QubitOperator('X7 Y8 X10'),
    (0,3): QubitOperator('X0 Y3 Y9'), # vertical
    (1,4): (-1)*QubitOperator('Y1 X4 Y9'),
    (2,5): QubitOperator('X2 Y5'), 
    (3,6): QubitOperator('X3 Y6'),
    (4,7): (-1)*QubitOperator('Y4 X7 Y10'),
    (5,8): QubitOperator('X5 Y8 Y10'),
})

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
    })

stabs = [
    Aop[(1,2)]*Aop[(2,5)]*Aop[(4,5)]*Aop[(1,4)],
    Aop[(3,4)]*Aop[(4,7)]*Aop[(6,7)]*Aop[(3,6)]
]



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

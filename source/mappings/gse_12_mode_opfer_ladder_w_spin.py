"""
12 mode GSE code - 2 6-mode ladders connected in the middle rungs
16 qubits with 6 stabilizers


"""



from openfermion.ops import QubitOperator
from openfermion.transforms import chemist_ordered
from copy import deepcopy

Aop = ({
    (0, 1): QubitOperator('X0 Y1'),# Horizontal
    (2, 3): QubitOperator('Y2 Z4 X5'),#
    (4, 5): QubitOperator('Y6 X7'),# 
    (6, 7): QubitOperator('X8 Y9'),#
    (8, 9): QubitOperator('Y10 Z12 X13'),# 
    (10, 11): QubitOperator('Y14 X15'),# 
    (0, 2): QubitOperator('Y0 X2'),# Vertical
    (2, 4): QubitOperator('Z2 Y3 X6'),#
    (1, 3): QubitOperator('X1 Z4 Y5'),#
    (3, 5): QubitOperator('X4 Y7'),#
    (6, 8): QubitOperator('Y8 X10'),#
    (8, 10): QubitOperator('Z10 Y11 X14'),#
    (7, 9): QubitOperator('X9 Z12 Y13'),#
    (9, 11): QubitOperator('X12 Y15'),#
})

# If odd number of particles in each spin sector is desired, uncomment these two lines
#Aop[(0,1)] = -1*Aop[(0,1)]
#Aop[(6,7)] = -1*Aop[(6,7)]

Aop_aux1 = QubitOperator('Z2 X3 Y4') # 2,3
Aop_aux2 = QubitOperator('Z10 X11 Y12') # 8,9 X10

# Corrected
Bop = ({
    0: QubitOperator('Z0'),
    1: QubitOperator('Z1'),
    2: QubitOperator('Z2 Z3'),
    3: QubitOperator('Z4 Z5'),
    4: QubitOperator('Z6'),
    5: QubitOperator('Z7'),
    6: QubitOperator('Z8'),
    7: QubitOperator('Z9'),
    8: QubitOperator('Z10 Z11'),
    9: QubitOperator('Z12 Z13'),
    10: QubitOperator('Z14'),
    11: QubitOperator('Z15'),
    })

# Corrected
stabs = ([
    Aop[(0, 1)] * Aop[(1, 3)] * Aop[(2, 3)] * Aop[(0, 2)],#
    Aop[(2, 3)] * Aop_aux1,
    Aop_aux1 * Aop[(3, 5)] * Aop[(4, 5)] * Aop[(2, 4)],
    Aop[(6, 7)] * Aop[(7, 9)] * Aop[(8, 9)] * Aop[(6, 8)],#
    Aop[(8, 9)] * Aop_aux2,
    Aop_aux2 * Aop[(9, 11)] * Aop[(10, 11)] * Aop[(8, 10)]
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
    fermion_ops = list(chemist_ordered(fermion_ops).terms.items())
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
    #a_index += 1
    #b_index += 1

    if a_index == b_index:

        return .5 * (QubitOperator(()) - Bop[a_index])

    else:

        if a_index < b_index:
            return (.25j * (QubitOperator(()) - Bop[a_index]) *
                    Aop[(a_index, b_index)] * (QubitOperator(()) -
                                               Bop[b_index]))

        else:
            return (.25j * (QubitOperator(()) - Bop[a_index]) *
                    Aop[(b_index, a_index)] * (-1) * (QubitOperator(()) -
                                                      Bop[b_index]))

"""
12 mode Derby-Klassen low-weight encoding - 3x4 lattice
15 qubits: 
one qubit for each mode -  indexed 0-11
one qubit in the top left, top right, bottom middle faces - indices 12,13,14 respectively


"""



from openfermion.ops import QubitOperator
from openfermion.transforms import chemist_ordered
from copy import deepcopy



Aop = ({
    (0, 1): -1*QubitOperator('Y0 X1 Y12'),#Horizontal edge ops
    (6,7): -1*QubitOperator('Y6 X7 Y13'),#
    (2,3): QubitOperator('X2 Y3 Y12'),#
    (8,9): QubitOperator('X8 Y9 Y13'),#
    (4,5): -1*QubitOperator('Y4 X5'),#
    (10,11): -1*QubitOperator('Y10 X11'),#
    (0,2): QubitOperator('Y0 X2 X12'),#Vertical edge ops #######
    (2,4): QubitOperator('Y2 X4'),# ########
    (1,3): QubitOperator('X1 Y3 X12'),#
    (3,5): QubitOperator('X3 Y5'),
    (6,8): QubitOperator('Y6 X8 X13'), ########
    (8,10): QubitOperator('Y8 X10'), ########
    (7,9): QubitOperator('X7 Y9 X13'),
    (9,11): QubitOperator('X9 Y11'),
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
    9: QubitOperator('Z9'),
    10: QubitOperator('Z10'),
    11: QubitOperator('Z11'),
    })


""" 
4 = 2*2 loop stabilizers in each spin sector
1 loop across the middle of the 2 sectors
1 stab equal to parity in spin up sector
"""
stabs = ([
    Aop[(2,3)] * Aop[(3,5)] * Aop[(4,5)] * Aop[(2,4)],
    Aop[(8,9)] * Aop[(9,11)] * Aop[(10,11)] * Aop[(8,10)],
])

def get_stabs(parity=0):
    parity_op = Bop[0]
    for j in range(11):
        parity_op *= Bop[j+1]
    if parity==1:
        parity_op = (-1.+0.j)*parity_op
    stabs.append(parity_op)
    return stabs

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

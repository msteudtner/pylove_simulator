"""
12 mode GSE code
34 qubits with 8 stabilizers

6(loops) + 9(horizontal doubled edges) + 8(vertical doubled edges) = 23 stabilizers
"""



from openfermion.ops import QubitOperator
from openfermion.transforms import chemist_ordered
from copy import deepcopy


qubit_map = {
    0: [0,1],
    1: [2,3,4],
    2: [5,6,7],
    3: [8,9,10,11],
    4: [12,13],
    5: [14,15,16],
    6: [17,18,19],
    7: [20,21],
    8: [22,23,24,25],
    9: [26,27,28],
    10: [29,30,31],
    11: [32,33]
}

vertex_degrees = {
    0: 4,
    1: 6,
    2: 6,
    3: 8,
    4: 4,
    5: 6,
    6: 6,
    7: 4,
    8: 8,
    9: 6,
    10: 6,
    11: 4
}

c = {}
for j in range(12):
    c[j] = {}
# 0,4,7,11 two-qubit vertices
for j in [0,4,7,11]:
    c[j][0] = QubitOperator(f'X{qubit_map[j][0]}')
    c[j][1] = QubitOperator(f'Y{qubit_map[j][0]}')
    c[j][2] = QubitOperator(f'Z{qubit_map[j][0]} X{qubit_map[j][1]}')
    c[j][3] = QubitOperator(f'Z{qubit_map[j][0]} Y{qubit_map[j][1]}')
# 1,2,3,5,6,8,9,10 three-qubit vertices
for j in [1,2,5,6,9,10]:
    c[j][0] = QubitOperator(f'X{qubit_map[j][0]}')
    c[j][1] = QubitOperator(f'Y{qubit_map[j][0]}')
    c[j][2] = QubitOperator(f'Z{qubit_map[j][0]} X{qubit_map[j][1]}')
    c[j][3] = QubitOperator(f'Z{qubit_map[j][0]} Y{qubit_map[j][1]}')
    c[j][4] = QubitOperator(f'Z{qubit_map[j][0]} Z{qubit_map[j][1]} X{qubit_map[j][2]}')
    c[j][5] = QubitOperator(f'Z{qubit_map[j][0]} Z{qubit_map[j][1]} Y{qubit_map[j][2]}')
# four-qubit vertices
for j in [3,8]:
    c[j][0] = QubitOperator(f'X{qubit_map[j][0]}')
    c[j][1] = QubitOperator(f'Y{qubit_map[j][0]}')
    c[j][2] = QubitOperator(f'Z{qubit_map[j][0]} X{qubit_map[j][1]}')
    c[j][3] = QubitOperator(f'Z{qubit_map[j][0]} Y{qubit_map[j][1]}')
    c[j][4] = QubitOperator(f'Z{qubit_map[j][0]} Z{qubit_map[j][1]} X{qubit_map[j][2]}')
    c[j][5] = QubitOperator(f'Z{qubit_map[j][0]} Z{qubit_map[j][1]} Y{qubit_map[j][2]}')
    c[j][6] = QubitOperator(f'Z{qubit_map[j][0]} Z{qubit_map[j][1]} Z{qubit_map[j][2]} X{qubit_map[j][3]}')
    c[j][7] = QubitOperator(f'Z{qubit_map[j][0]} Z{qubit_map[j][1]} Z{qubit_map[j][2]} Y{qubit_map[j][3]}')

Aop = ({ # horizontal
    (0,1):   c[0][0] * c[1][2],#
    (1,6):   c[1][0] * c[6][2],
    (6,7):   c[6][0] * c[7][1],
    (2,3):   c[2][0] * c[3][2],#
    (3,8):   c[3][0] * c[8][2],
    (8,9):   c[8][0] * c[9][2],
    (4,5):   c[4][0] * c[5][2],#
    (5,10):  c[5][0] * c[10][2],
    (10,11): c[10][0]* c[11][0],
    (0,2):   c[0][1] * c[2][2], # vertical
    (2,4):   c[2][1] * c[4][1],
    (1,3):   c[1][1] * c[3][3],
    (3,5):   c[3][1] * c[5][1],
    (6,8):   c[6][1] * c[8][3],
    (8,10):  c[8][1] * c[10][1],
    (7,9):   c[7][0] * c[9][0],
    (9,11):  c[9][1] * c[11][1]
})
# 10: 2,3,0
Aop_aux = ({ # add the 1/2 the degree of the vertex to the operator indices
    (0,1):   c[0][0 + int(vertex_degrees[0]/2)] * c[1][2 + int(vertex_degrees[1]/2)],# horizontal
    (1,6):   c[1][0 + int(vertex_degrees[1]/2)] * c[6][2 + int(vertex_degrees[6]/2)],
    (6,7):   c[6][0 + int(vertex_degrees[6]/2)] * c[7][1 + int(vertex_degrees[7]/2)],
    (2,3):   c[2][0 + int(vertex_degrees[2]/2)] * c[3][2 + int(vertex_degrees[3]/2)],#
    (3,8):   c[3][0 + int(vertex_degrees[3]/2)] * c[8][2 + int(vertex_degrees[8]/2)],
    (8,9):   c[8][0 + int(vertex_degrees[8]/2)] * c[9][2 + int(vertex_degrees[9]/2)],
    (4,5):   c[4][0 + int(vertex_degrees[4]/2)] * c[5][2 + int(vertex_degrees[5]/2)],#
    (5,10):  c[5][0 + int(vertex_degrees[5]/2)] * c[10][2 + int(vertex_degrees[10]/2)],
    (10,11): c[10][0 + int(vertex_degrees[10]/2)]* c[11][0 + int(vertex_degrees[11]/2)],
    (0,2):   c[0][1 + int(vertex_degrees[0]/2)] * c[2][2 + int(vertex_degrees[2]/2)], # vertical
    (2,4):   c[2][1 + int(vertex_degrees[2]/2)] * c[4][1 + int(vertex_degrees[4]/2)],
    (1,3):   c[1][1 + int(vertex_degrees[1]/2)] * c[3][3 + int(vertex_degrees[3]/2)],
    (3,5):   c[3][1 + int(vertex_degrees[3]/2)] * c[5][1 + int(vertex_degrees[5]/2)],
    (6,8):   c[6][1 + int(vertex_degrees[6]/2)] * c[8][3 + int(vertex_degrees[8]/2)],
    (8,10):  c[8][1 + int(vertex_degrees[8]/2)] * c[10][1 + int(vertex_degrees[10]/2)],
    (7,9):   c[7][0 + int(vertex_degrees[7]/2)] * c[9][0 + int(vertex_degrees[9]/2)],
    (9,11):  c[9][1 + int(vertex_degrees[9]/2)] * c[11][1 + int(vertex_degrees[11]/2)]
})

Bop = ({
    0: QubitOperator(f'Z{qubit_map[0][0]} Z{qubit_map[0][1]}'),
    1: QubitOperator(f'Z{qubit_map[1][0]} Z{qubit_map[1][1]} Z{qubit_map[1][2]}'),
    2: QubitOperator(f'Z{qubit_map[2][0]} Z{qubit_map[2][1]} Z{qubit_map[2][2]}'),
    3: QubitOperator(f'Z{qubit_map[3][0]} Z{qubit_map[3][1]} Z{qubit_map[3][2]} Z{qubit_map[3][3]}'),
    4: QubitOperator(f'Z{qubit_map[4][0]} Z{qubit_map[4][1]}'),
    5: QubitOperator(f'Z{qubit_map[5][0]} Z{qubit_map[5][1]} Z{qubit_map[5][2]}'),
    6: QubitOperator(f'Z{qubit_map[6][0]} Z{qubit_map[6][1]} Z{qubit_map[6][2]}'),
    7: QubitOperator(f'Z{qubit_map[7][0]} Z{qubit_map[7][1]}'),
    8: QubitOperator(f'Z{qubit_map[8][0]} Z{qubit_map[8][1]} Z{qubit_map[8][2]} Z{qubit_map[8][3]}'),
    9: QubitOperator(f'Z{qubit_map[9][0]} Z{qubit_map[9][1]} Z{qubit_map[9][2]}'),
    10: QubitOperator(f'Z{qubit_map[10][0]} Z{qubit_map[10][1]} Z{qubit_map[10][2]}'),
    11: QubitOperator(f'Z{qubit_map[11][0]} Z{qubit_map[11][1]}'),
})

stabs = ([
    Aop[(0,1)] * Aop[(1,3)] * Aop[(2,3)] * Aop[(0,2)],
    Aop[(1,6)] * Aop[(6,8)] * Aop[(3,8)] * Aop[(1,3)],
    Aop[(6,7)] * Aop[(7,9)] * Aop[(8,9)] * Aop[(6,8)],
    Aop[(2,3)] * Aop[(3,5)] * Aop[(4,5)] * Aop[(2,4)],
    Aop[(3,8)] * Aop[(8,10)] * Aop[(5,10)] * Aop[(3,5)],
    Aop[(8,9)] * Aop[(9,11)] * Aop[(10,11)] * Aop[(8,10)],
    (-1)*Aop[(0,1)]*Aop_aux[(0,1)],# horizontal doubled edges
    (-1)*Aop[(1,6)]*Aop_aux[(1,6)],
    (-1)*Aop[(6,7)]*Aop_aux[(6,7)],
    (-1)*Aop[(2,3)]*Aop_aux[(2,3)],
    (-1)*Aop[(3,8)]*Aop_aux[(3,8)],
    (-1)*Aop[(8,9)]*Aop_aux[(8,9)],
    (-1)*Aop[(4,5)]*Aop_aux[(4,5)],
    (-1)*Aop[(5,10)]*Aop_aux[(5,10)],
    (-1)*Aop[(10,11)]*Aop_aux[(10,11)],
    (-1)*Aop[(0,2)]*Aop_aux[(0,2)], # vertical doubled edges
    (-1)*Aop[(2,4)]*Aop_aux[(2,4)],
    (-1)*Aop[(1,3)]*Aop_aux[(1,3)],
    (-1)*Aop[(3,5)]*Aop_aux[(3,5)],
    (-1)*Aop[(6,8)]*Aop_aux[(6,8)],
    (-1)*Aop[(8,10)]*Aop_aux[(8,10)],
    (-1)*Aop[(7,9)]*Aop_aux[(7,9)],
    (-1)*Aop[(9,11)]*Aop_aux[(9,11)],
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

"""
14(=2*7) mode GSE code
Geometry: a pair of 7 vertex complete graphs
Total qubits 42
"""

from openfermion.ops import QubitOperator
from openfermion.transforms import chemist_ordered
from copy import deepcopy


local_majs = []
for j in range(14):
    local_majs.append([])
    local_majs[j].append(QubitOperator(f'X{3*j} Z{3*j+2}')) # XIZ
    local_majs[j].append(QubitOperator(f'Y{3*j} Z{3*j+2}')) # YIZ
    local_majs[j].append(QubitOperator(f'Z{3*j} X{3*j+1}')) # ZXI
    local_majs[j].append(QubitOperator(f'Z{3*j} Y{3*j+1}')) # ZYI
    local_majs[j].append(QubitOperator(f'Z{3*j+1} X{3*j+2}')) # IZX
    local_majs[j].append(QubitOperator(f'Z{3*j+1} Y{3*j+2}')) # IZY

# The error might be in the interplay of the ordering here with the loop ops
Aop = ({})
for j in range(0,7):
    for k in range(j+1,7):
        Aop[(j,k)] = local_majs[j][k-j-1] * local_majs[k][j-k %7]
for j in range(7,14):
    for k in range(j+1,14):
        Aop[(j,k)] = local_majs[j][(k % 7)-(j % 7)-1] * local_majs[k][(j % 7)-(k % 7) %7]

# If odd number of particles in each spin sector -> uncomment these two lines
# Note though that because the overall parity is set to be even by the stabilizers, the odd number of physical
# particles is taken care of by the extra mode having the same parity as the physical modes
Aop[(0,1)] = -1*Aop[(0,1)]
Aop[(7,8)] = -1*Aop[(7,8)]

# Construct Bops
Bop = ({})
for k in range(14):
    Bop[k] = (-1.0j)**3 * local_majs[k][0]
    Bop[k] = Bop[k] * local_majs[k][1]
    Bop[k] = Bop[k] * local_majs[k][2]
    Bop[k] = Bop[k] * local_majs[k][3]
    Bop[k] = Bop[k] * local_majs[k][4]
    Bop[k] = Bop[k] * local_majs[k][5]

# Build a basis of loops within each spin sector
stabs = ([])
for k in range(0,1):
    for l in range(k+1,6):
        for m in range(l+1,7):
            stabs.append(1j**3 * Aop[(k,l)] * Aop[(l,m)] * -1 * Aop[(k,m)])
for k in range(7,8):
    for l in range(k+1,13):
        for m in range(l+1,14):
            stabs.append(1j**3 * Aop[(k,l)] * Aop[(l,m)] * -1 * Aop[(k,m)])

def get_stabs(parity=0):
    #parity_op = Bop[0]
    #for j in range(1,11):
    #    parity_op *= Bop[j+1]
    #if parity==1:
    #    parity_op = (-1.+0.j)*parity_op
    #stabs.append(parity_op)
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
    # modes are indexed 0-5,6-11 but qubits are 0-6,7-13 freezing out 6,13, add one to indices higher than 5
    
    if a_index > 5:
        a_index +=1
    if b_index > 5:
        b_index +=1
    
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

"""
"""

import sys
sys.path.append("../")
from openfermion.ops import QubitOperator
from openfermion.transforms import chemist_ordered
from copy import deepcopy
import numpy as np
import networkx as nx
import random
from copy import deepcopy

from mappings.GSE_constructor import GSE

qubit_map = {
    0: [0],
    1: [1,2],
    2: [3],
    3: [4,5],
    4: [6,7],
    5: [8,9],
    6: [10],
    7: [11,12],
    8: [13],
}

# create upper triangular adjacency matrix
adj = np.zeros((9,9))
adj[0,1] = 1
adj[1,2] = 1
adj[3,4] = 1
adj[4,5] = 1
adj[6,7] = 1
adj[7,8] = 1
adj[0,3] = 1
adj[3,6] = 1
adj[1,4] = 1
adj[4,7] = 1
adj[2,5] = 1
adj[5,8] = 1
adj[1,5] = 1
adj[3,7] = 1

adj = adj + adj.T

Aop, Bop, stabs, qubits = GSE(adj, qubit_map)

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

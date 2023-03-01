from openfermion import QubitOperator
from openfermion.transforms import chemist_ordered
from numpy import imag
from copy import deepcopy
import sys

sys.path.append('../simulator')
from simulator.utils import remap_modes


longstab = ({
    (0, 1): QubitOperator('Z0 X24'),                             # 25
    (1, 6): QubitOperator('Y0 Z1 Z2 Z3 Z4 X5 X12 Z24'),          # 13
    (6, 7): QubitOperator('Y5 X6 Z12 X13'),                      # 14
    (7, 12): QubitOperator('Y6 Z7 Z8 Z9 Z10 X11 Z13 X14'),       # 15
    (11, 12): QubitOperator('Z10 Z11 Z14 X15'),                  # 16
    (8, 11): QubitOperator('Y7 Z8 Z9 X10 Z15 X16 Z23'),          # 17
    (5, 8): QubitOperator(' Y4 Z5 Z6 X7 Z16 X17'),               # 18
    (2, 5): QubitOperator('Y1 Z2 Z3 X4 Z17 X18'),                # 19
    (2, 3): QubitOperator('Z1 Z2 Z18 X19'),                      # 20
    (3, 4): QubitOperator('Y2 X3 Z19 X20'),                      # 21
    (4, 9): QubitOperator('Y3 Z4 Z5 Z6 Z7 X8 Z20 X21'),          # 22
    (9, 10): QubitOperator('Y8 X9 Z21 X22'),                     # 23
    (10, 11): QubitOperator('Z9 Z10 X15 Z16 Y22 Y23'),           # 24
})

corr = ({
    (0, 1): QubitOperator('Z24'),                                # 25
    (1, 6): QubitOperator('Z12'),                                # 13
    (6, 7): QubitOperator('Z13'),                                # 14
    (7, 12): QubitOperator('Z14'),                               # 15
    (11, 12): QubitOperator('Z15 Z23'),                          # 16
    (8, 11): QubitOperator('Z16'),                               # 17
    (5, 8): QubitOperator('Z17'),                                # 18
    (2, 5): QubitOperator('Z18'),                                # 19
    (2, 3): QubitOperator('Z19'),                                # 20
    (3, 4): QubitOperator('Z20'),                                # 21
    (4, 9): QubitOperator('Z21'),                                # 22
    (9, 10): QubitOperator('Z22 Z23'),                           # 23
    (10, 11):  QubitOperator('Z23'),                             # 24
})


def anticoms(a, b, thresh=10**-8):
    """check if a and b anticommute"""
    return abs(imag(list((a*b).terms.values())[0])) >= thresh


def make_A(index_one, index_two):
    pstring = ((index_one-1, 'Y'),)

    for x in range(index_one, index_two-1):
        pstring += ((x, 'Z'),)

    pstring += ((index_two-1, 'X'),)
    return QubitOperator(pstring)


def correct_op(op):
    correction = QubitOperator(())
    for key, val in list(longstab.items()):
        if anticoms(op, val):
            correction *= corr[key]
    return op * correction


A_op = ({
    (1, 6): correct_op(make_A(1, 6)) * longstab[(1, 6)],
    (6, 7): correct_op(make_A(6, 7)) * longstab[(6, 7)],
    (7, 12): correct_op(make_A(7, 12)) * longstab[(7, 12)],
    (8, 11): correct_op(make_A(8, 11)) * longstab[(8, 11)],
    (5, 8): correct_op(make_A(5, 8)) * longstab[(5, 8)],
    (2, 5): correct_op(make_A(2, 5)),
    (2, 3): correct_op(make_A(2, 3)) * longstab[(2, 3)],
    (3, 4): correct_op(make_A(3, 4)) * longstab[(3, 4)],
    (4, 9): correct_op(make_A(4, 9)) * longstab[(4, 9)],
    (9, 10): correct_op(make_A(9, 10)) * longstab[(9, 10)],
    (10, 11): correct_op(make_A(10, 11)) * longstab[(10, 11)],
    (1, 2): correct_op(make_A(1, 2)),
    (2, 3): correct_op(make_A(2, 3)),
    (4, 5): correct_op(make_A(4, 5)),
    (5, 6): correct_op(make_A(5, 6)),
    (7, 8): correct_op(make_A(7, 8)),
    (8, 9): correct_op(make_A(8, 9)),
    (10, 11): correct_op(make_A(10, 11)),
    (11, 12): correct_op(make_A(11, 12)),
})

B_op = ({
    1: correct_op(QubitOperator('Z0')),
    2: correct_op(QubitOperator('Z1')),
    3: correct_op(QubitOperator('Z2')),
    4: correct_op(QubitOperator('Z3')),
    5: correct_op(QubitOperator('Z4')),
    6: correct_op(QubitOperator('Z5')),
    7: correct_op(QubitOperator('Z6')),
    8: correct_op(QubitOperator('Z7')),
    9: correct_op(QubitOperator('Z8')),
    10: correct_op(QubitOperator('Z9')),
    11: correct_op(QubitOperator('Z10')),
    12: correct_op(QubitOperator('Z11')),
})

stabs = [
    longstab[(1, 6)],                        # 0
    longstab[(2, 5)],                        # 1
    longstab[(3, 4)],                        # 2
    longstab[(2, 3)],                        # 3
    longstab[(4, 9)],                        # 4
    longstab[(5, 8)],                        # 5
    longstab[(6, 7)],                        # 6
    longstab[(7, 12)],                       # 7
    longstab[(8, 11)],                       # 8
    longstab[(9, 10)],                       # 9
    longstab[(10, 11)],                      # 10
    longstab[(11, 12)],                      # 11
    longstab[(0, 1)],                        # 12
]

table = {
    1: 0,
    2: 2,
    3: 4,
    4: 5,
    5: 3,
    6: 1,
    7: 6,
    8: 8,
    9: 10,
    10: 11,
    11: 9,
    12: 7
}



(A_old, B_old) = (deepcopy(A_op), deepcopy(B_op))
A_op, B_op = remap_modes(table, (A_old, B_old))



"""table = {
    1: 0,
    2: 4,
    3: 8,
    4: 10,
    5: 6,
    6: 2,
    7: 1,
    8: 5,
    9: 9,
    10: 11,
    11: 7,
    12: 3
}"""
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

    if a_index == b_index:

        return .5 * (QubitOperator(()) - B_op[a_index])

    else:

        if a_index < b_index:
            return (.25j * (QubitOperator(()) - B_op[a_index]) *
                    A_op[(a_index, b_index)] *
                    (QubitOperator(()) - B_op[b_index]))

        else:
            return (.25j * (QubitOperator(()) - B_op[a_index]) *
                    A_op[(b_index, a_index)] * (-1) *
                    (QubitOperator(()) - B_op[b_index]))

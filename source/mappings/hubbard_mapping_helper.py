"""
2x3 lattice Fermi-Hubbard model mapping script

Contains functions to call to generate the qubit Hamiltonians 

main scripts return a pair (Ham = QubitOperator , stabs = [list of stabilizers])
"""

from openfermion.utils import up_then_down
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
import sys
sys.path.append("../")
from mappings import gse_12_mode_opfer_ladder_w_spin as gse_map
from mappings import gse_distance_3 as gse_dist3_map
from mappings import lw_2x3_wspin as lw_map
from mappings import aqm_12_modes_25_qubits as aqm_map
from mappings import jw_12_mode as jw_map
from mappings import lw_9_mode as lw_9_map
from mappings import jw_9_mode as jw_9_map
from mappings import gse_9_mode as gse_9_map
from mappings import jw_12_mode_snake as jw_snake_map
from mappings import lw_12_mode_15_qubits as lw_15_map
from mappings import gse_12_mode_20_qubits as gse_20_map
from mappings import gse_12_mode_34_qubits as gse_34_map
from copy import deepcopy
import numpy

reindex_map = {
    0: 0, 
    1: 6, 
    2: 1, 
    3: 7, 
    4: 2, 
    5: 8, 
    6: 3, 
    7: 9, 
    8: 4, 
    9: 10, 
    10: 5, 
    11: 11
}

def up_then_down_reorder(fer_ham):
    fermion_ops = deepcopy(fer_ham)
    fermion_ops = list(fermion_ops.terms.items())
    fer_ham = FermionOperator()
    for term in fermion_ops:
        #print(term)
        coeff = term[1]
        op_string = ''
        for op in term[0]:
            #print(op)
            if op[1]==0:
                op_string += f' {reindex_map[op[0]]}'
            if op[1]==1:
                op_string += f' {reindex_map[op[0]]}^'
        fer_ham += FermionOperator(op_string,coeff)
    return fer_ham

def lw_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through the Derby-Klassen low-weight mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return lw_map.fermapping(fer_ham), lw_map.stabs, lw_map.Aop, lw_map.Bop

def jw_9_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through the Derby-Klassen low-weight mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return jw_9_map.fermapping(fer_ham), jw_9_map.stabs, jw_9_map.Aop, jw_9_map.Bop

def lw_9_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through the Derby-Klassen low-weight mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return lw_9_map.fermapping(fer_ham), lw_9_map.stabs, lw_9_map.Aop, lw_9_map.Bop

def gse_9_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through the Derby-Klassen low-weight mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return gse_9_map.fermapping(fer_ham), gse_9_map.stabs, gse_9_map.Aop, gse_9_map.Bop

def gse_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through a 16 qubit 5 stabilizer GSE mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return gse_map.fermapping(fer_ham), gse_map.stabs, gse_map.Aop, gse_map.Bop

def lw_15_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through a 16 qubit 5 stabilizer GSE mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return lw_15_map.fermapping(fer_ham), lw_15_map.stabs, lw_15_map.Aop, lw_15_map.Bop

def gse_20_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through a 16 qubit 5 stabilizer GSE mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return gse_20_map.fermapping(fer_ham), gse_20_map.stabs, gse_20_map.Aop, gse_20_map.Bop

def gse_34_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through a 16 qubit 5 stabilizer GSE mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return gse_34_map.fermapping(fer_ham), gse_34_map.stabs, gse_34_map.Aop, gse_34_map.Bop


def aqm_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through a 25 qubit 13 stabilizer AQM mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return aqm_map.fermapping(fer_ham), aqm_map.stabs, aqm_map.A_op, aqm_map.B_op

def gse_dist3_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through a 42 qubit GSE mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return gse_dist3_map.fermapping(fer_ham), gse_dist3_map.stabs, gse_dist3_map.Aop, gse_dist3_map.Bop

def jw_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through a 12 qubit 1 stabilizer JW mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return jw_map.fermapping(fer_ham), jw_map.stabs, jw_map.Aop, jw_map.Bop

def jw_snake_mapping(fer_ham,up_down_reorder=True):
    """
    Give the 2x3 Fermi-Hubbard Hamiltonian in qubit form through a 12 qubit 1 stabilizer JW mapping
    Args:
        fer_ham (FermionOperator): OpenFermion FermionOperator
    Return:
        gse_qubit_ham (QubitOperator): Transformed Hamiltonian
        stabs (list): List of QubitOperator stabilizers
        Aops: (dict): Dictionary containing Aops 
        Bops: (dict): Dictionary containing Bops
    """
    if up_down_reorder:
        fer_ham = up_then_down_reorder(fer_ham)
    return jw_snake_map.fermapping(fer_ham), jw_snake_map.stabs, jw_snake_map.Aop, jw_snake_map.Bop

nn_singles = []
# Hopping terms
nn_singles.append(-1*(FermionOperator('0^ 1') + FermionOperator('1^ 0')))# spin up sector
nn_singles.append(-1*(FermionOperator('0^ 2') + FermionOperator('2^ 0')))
nn_singles.append(-1*(FermionOperator('1^ 3') + FermionOperator('3^ 1')))
nn_singles.append(-1*(FermionOperator('2^ 3') + FermionOperator('3^ 2')))
nn_singles.append(-1*(FermionOperator('2^ 4') + FermionOperator('4^ 2')))
nn_singles.append(-1*(FermionOperator('3^ 5') + FermionOperator('5^ 3')))
nn_singles.append(-1*(FermionOperator('4^ 5') + FermionOperator('5^ 4')))
nn_singles.append(-1*(FermionOperator('6^ 7') + FermionOperator('7^ 6')))# spin down sector
nn_singles.append(-1*(FermionOperator('6^ 8') + FermionOperator('8^ 6')))
nn_singles.append(-1*(FermionOperator('7^ 9') + FermionOperator('9^ 7')))
nn_singles.append(-1*(FermionOperator('8^ 9') + FermionOperator('9^ 8')))
nn_singles.append(-1*(FermionOperator('8^ 10') + FermionOperator('10^ 8')))
nn_singles.append(-1*(FermionOperator('9^ 11') + FermionOperator('11^ 9')))
nn_singles.append(-1*(FermionOperator('10^ 11') + FermionOperator('11^ 10')))
# Number terms
for j in range(12):
    nn_singles.append(-2*FermionOperator(f'{j}^ {j}'))

# Hubbard terms
nn_doubles = []
nn_doubles.append( 4*FermionOperator('0^ 0 6^ 6') - (2)*FermionOperator('0^ 0')-(2)*FermionOperator('6^ 6')   )
nn_doubles.append( 4*FermionOperator('1^ 1 7^ 7') -(2)*FermionOperator('1^ 1')-(2)*FermionOperator('7^ 7')   )
nn_doubles.append( 4*FermionOperator('2^ 2 8^ 8') -(2)*FermionOperator('2^ 2')-(2)*FermionOperator('8^ 8')   )
nn_doubles.append( 4*FermionOperator('3^ 3 9^ 9') -(2)*FermionOperator('3^ 3')-(2)*FermionOperator('9^ 9')   )
nn_doubles.append( 4*FermionOperator('4^ 4 10^ 10') -(2)*FermionOperator('4^ 4')-(2)*FermionOperator('10^ 10')   )
nn_doubles.append( 4*FermionOperator('5^ 5 11^ 11') -(2)*FermionOperator('5^ 5')-(2)*FermionOperator('11^ 11')   )


def get_ferm_circuit_old(angles, layers):
    """
    Iterate over the following circuit structure Hoppers -> Numbers -> Number-Number
    for the desired number of layers
    Args:
        angles (list): Rotation angles for each generator should be 32*layers
    Returns:
        ferm_circuit (list): List of FermionOperators, each layer is given as an entry in the list and represents
            a Hamiltonian
    """
    
    ferm_circuit=[]
    for j in range(layers):
        layer_gen = angles[32*j]*nn_singles[0]
        for k in range(1,len(nn_singles)):
            layer_gen += angles[32*j +k]*nn_singles[k]
        for k in range(len(nn_doubles)):
            layer_gen += angles[32*j + 26 +k]*nn_doubles[k]
        ferm_circuit.append(layer_gen)
    return ferm_circuit

def get_ferm_circuit(angles, layers=0):
    """
    Iterate over the following circuit structure Hoppers -> Numbers -> Number-Number
    for the desired number of layers
    Args:
        angles (list): Rotation angles for each generator should be 32*layers
    Returns:
        ferm_circuit (list): List of FermionOperators, each layer is given as an entry in the list and represents
            a Hamiltonian
    """

    ferm_circuit = []
    operator_bank = nn_singles + nn_doubles
    for j in range(len(angles)):
        ferm_circuit.append( angles[j]*operator_bank[j % len(operator_bank)]   )
    #
    #ferm_circuit=[]
    #for j in range(layers):
    #    for k in range(len(nn_singles)):
    #        ferm_circuit.append(angles[32*j +k]*nn_singles[k])
    #    for k in range(len(nn_doubles)):
    #        ferm_circuit.append(angles[32*j + 26 +k]*nn_doubles[k])
    return ferm_circuit

def gse_circuit_alt(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(gse_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule


def jw_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(jw_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule

def jw_snake_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(jw_snake_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule

def lw_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(lw_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule

def gse_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(gse_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule

def lw_15_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(lw_15_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule

def gse_20_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(gse_20_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule

def gse_34_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(gse_34_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule


def aqm_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(aqm_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule

def gse_dist3_circuit(ferm_circuit):
    """
    """
    schedule = []
    # Map the Fermion Hams to qubit hams
    qubit_hams = []
    for j in range(len(ferm_circuit)):
        qubit_hams.append(gse_dist3_map.fermapping(ferm_circuit[j]))
    # Collect the terms as QubitOperators into a schedule list
    for j in range(len(qubit_hams)):
        terms = list(qubit_hams[j].terms.items())
        for term in terms:
            paulis = term[0]
            if len(paulis)==0:
                continue
            coeff = term[1]
            op_string = ''
            for pauli in paulis:
                op_string += pauli[1]+str(pauli[0])+' '
            op = QubitOperator(op_string, term[1])
            schedule.append(op)
    return schedule
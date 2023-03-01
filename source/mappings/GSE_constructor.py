from openfermion.ops import QubitOperator
from copy import deepcopy
import numpy as np
import networkx as nx
import random
from copy import deepcopy
import sys
sys.path.append("../")
from simulator.utils import check_mapping

def generate_neighbors(adj):
    neighbors = {}
    for j in range(adj.shape[0]):
        neighbors[j] = {}
        neighbor_counter = 0
        for k in range(adj.shape[1]):
            if adj[j,k]==1:
                neighbors[j][k] = [neighbor_counter]
                neighbor_counter +=1
            if adj[j,k]>1:
                neighbors[j][k] = []
                for l in range(int(adj[j,k])):
                    neighbors[j][k].append(neighbor_counter)
                    neighbor_counter +=1
    return neighbors

def cyclic(a):
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b

def cyclic_shift(a):
    return [a[(j-1)%len(a)] for j in range(len(a))]

def get_cops(adj,qubit_map,op_type):

    if op_type is not None and op_type not in ['JW','3']:
        op_type ='JW'
    if op_type is None or op_type=='JW':
        # Here I'll be mentally lazy and just enumerate more than I should ever need, currently good up to degree 18 vertices
        vertex_op_strings = {}
        vertex_op_strings[1] = ['X','Y']
        vertex_op_strings[2] = ['X','Y','ZX','ZY']
        vertex_op_strings[3] = ['X','Y','ZX','ZY','ZZX','ZZY']
        vertex_op_strings[4] = ['X','Y','ZX','ZY','ZZX','ZZY','ZZZX','ZZZY']
        vertex_op_strings[5] = ['X','Y','ZX','ZY','ZZX','ZZY','ZZZX','ZZZY','ZZZZX','ZZZZY']
        vertex_op_strings[6] = ['X','Y','ZX','ZY','ZZX','ZZY','ZZZX','ZZZY','ZZZZX','ZZZZY','ZZZZZX','ZZZZZY']
        vertex_op_strings[7] = ['X','Y','ZX','ZY','ZZX','ZZY','ZZZX','ZZZY','ZZZZX','ZZZZY','ZZZZZX','ZZZZZY','ZZZZZZX','ZZZZZZY']
        vertex_op_strings[8] = ['X','Y','ZX','ZY','ZZX','ZZY','ZZZX','ZZZY','ZZZZX','ZZZZY','ZZZZZX','ZZZZZY','ZZZZZZX','ZZZZZZY','ZZZZZZZX','ZZZZZZZY']
        vertex_op_strings[9] = ['X','Y','ZX','ZY','ZZX','ZZY','ZZZX','ZZZY','ZZZZX','ZZZZY','ZZZZZX','ZZZZZY','ZZZZZZX','ZZZZZZY','ZZZZZZZX','ZZZZZZZY','ZZZZZZZZX','ZZZZZZZZY']
    elif op_type=='3':
        vertex_op_strings = {}
        for j in range(3,10): # iterate over number of qubits
            vertex_op_strings[j] = []
            if j%2==1: # if number of qubits is odd
                # make generating lists
                gen1 = []
                gen2 = []
                for k in range(int((j-1)/2)):
                    gen1.append('Z')
                    gen2.append('Z')
                gen1.append('X')
                gen2.append('Y')
                for k in range(int((j-1)/2)):
                    gen1.append('I')
                    gen2.append('I')
                for k in range(j):
                    gen1_copy = deepcopy(gen1)
                    gen2_copy = deepcopy(gen2)
                    for l in range(k):
                        gen1_copy = cyclic_shift(gen1_copy)
                        gen2_copy = cyclic_shift(gen2_copy)
                    op_str1 = ''
                    op_str2 = ''
                    for l in range(j):
                        op_str1 += gen1_copy[l]
                        op_str2 += gen2_copy[l]
                    vertex_op_strings[j].append(op_str1)
                    vertex_op_strings[j].append(op_str2)
            
            if j%2==0: # if number of qubits is even j = 2*k
                # make generating lists
                gen1 = []
                gen2 = []
                gen3 = ['X']
                gen4 = ['Y']
                for k in range(int(j/2)): 
                    gen1.append('Z')
                    gen2.append('Z')
                    gen3.append('I')
                    gen4.append('I')
                gen1.append('X')
                gen2.append('Y')
                for k in range(int(j/2)-1):
                    gen1.append('I')
                    gen2.append('I')
                    gen3.append('Z')
                    gen4.append('Z')
                for k in range(int(j/2)):
                    gen1_copy = deepcopy(gen1)
                    gen2_copy = deepcopy(gen2)
                    gen3_copy = deepcopy(gen3)
                    gen4_copy = deepcopy(gen4)
                    for l in range(k):
                        gen1_copy = cyclic_shift(gen1_copy)
                        gen2_copy = cyclic_shift(gen2_copy)
                        gen3_copy = cyclic_shift(gen3_copy)
                        gen4_copy = cyclic_shift(gen4_copy)
                    op_str1 = ''
                    op_str2 = ''
                    op_str3 = ''
                    op_str4 = ''
                    for l in range(len(gen1_copy)):
                        op_str1 += gen1_copy[l]
                        op_str2 += gen2_copy[l]
                        op_str3 += gen3_copy[l]
                        op_str4 += gen4_copy[l]
                    vertex_op_strings[j].append(op_str1)
                    vertex_op_strings[j].append(op_str2)
                    vertex_op_strings[j].append(op_str3)
                    vertex_op_strings[j].append(op_str4)
    c_dict = {}
    for j in range(adj.shape[0]): # iterate over modes
        valence = int(np.sum(adj[j,:]))
        qubits = int(np.ceil(valence/2))
        c_dict[j] = {}
        for k in range(2*qubits):
            builder_str = vertex_op_strings[qubits][k]
            op_str = ''
            for l in range(len(builder_str)):
                if builder_str[l]!='I':
                    op_str = op_str + builder_str[l] + f'{qubit_map[j][l]} '
            c_dict[j][k] = QubitOperator(op_str)
    return c_dict

def get_Aops(adj,neighbors,c):
    ops = {}
    for j in range(adj.shape[0]):
        for k in range(adj.shape[1]):
            if adj[j,k] > 0:
                if j < k:
                    ops[(j,k)]= c[j][neighbors[j][k][0]] * c[k][neighbors[k][j][0]]
                elif j > k:
                    ops[(j,k)]= (-1)*c[j][neighbors[j][k][0]] * c[k][neighbors[k][j][0]]
    return ops

def get_Aops_multi(adj,neighbors,c):
    ops = {}
    for j in range(adj.shape[0]):
        for k in range(adj.shape[1]):
            ops[(j,k)] = []
            for l in range(int(adj[j,k])):
                if j < k:
                    ops[(j,k)].append(c[j][neighbors[j][k][l]] * c[k][neighbors[k][j][l]])
                elif j > k:
                    ops[(j,k)].append((-1)*c[j][neighbors[j][k][l]] * c[k][neighbors[k][j][l]])
    return ops

def get_Bops(adj,c):
    ops = {}
    for j in range(adj.shape[0]):
        valence = int(np.sum(adj[j,:]))
        qubits = int(np.ceil(valence/2))
        op = c[j][0]
        for k in range(1,2*qubits):
            op = op * c[j][k]
        ops[j] = (-1j)**(qubits) * op
    return ops

def get_stabs(adj,Aop,Aop_multi):
    stab_list = []
    # First deal with multi-edges - each multi-edge of value n supplies n-1 stabilizers
    adj_copy = deepcopy(adj)
    for j in range(adj.shape[0]):
        #for k in range(adj.shape[1]):
        for k in range(j):
            if adj[j,k] > 1:
                for l in range(int(adj[j,k]-1)):
                    stab_list.append( (-1)**2 *  Aop_multi[j,k][l]  *  Aop_multi[j,k][(l+1)]    ) 
                adj_copy[j,k] = 1
    # Now adj_copy is a standard (non-multi graph)
    G = nx.Graph(adj_copy)
    cycles = nx.minimum_cycle_basis(G)
    for j in range(len(cycles)):
        # make tuples
        edge_tuples = []
        # make sure consecutive vertices are adjacent on the graph
        fully_ordered=False
        while fully_ordered is False:
            
            for k in range(len(cycles[j])):
                if adj[ cycles[j][k], cycles[j][ (k+1) % len(cycles[j]) ] ] <1:
                    # make a swap
                    neighboring=False
                    counter = 2
                    while neighboring is False:
                        if adj[cycles[j][k],cycles[j][(k+counter) % len(cycles[j])]] > 0:
                            cycles[j][(k+1) % len(cycles[j])], cycles[j][(k+counter) % len(cycles[j])] = cycles[j][(k+counter) % len(cycles[j])], cycles[j][(k+1) % len(cycles[j])]
                            neighboring = True 
                        else:
                            counter +=1
            partially_ordered = True
            for k in range(len(cycles[j])):
                if adj[cycles[j][k],cycles[j][(k+1) % len(cycles[j])]] <1:
                    partially_ordered = False
                    break
            if partially_ordered:
                fully_ordered = True
            else: 
                random.shuffle(cycles[j])
        for k in range(len(cycles[j])):
            edge_tuples.append((cycles[j][k],cycles[j][(k + 1) % len(cycles[j])]))
        op = (1j)**len(cycles[j]) *  Aop[edge_tuples[0]]
        for k in range(1,len(cycles[j])):
            op = op * Aop[edge_tuples[k]]
        stab_list.append(op)
    return stab_list



def GSE(adj, qubit_map = None, op_type = None):
    """
    Generate a GSE-type encoding for fermionic modes connected according to the edges of a supplied
    upper triangular matrix.
    Upper triangular means edges should be assigned from (lower_index,higher index)

    Args:
            adj (numpy.ndarray): Adjacency matrix, can have entries >1 for multiedges
            (optional) qubit_map (dict): A dictionary of lists specifying which qubits are assigned to which modes
            (optional) vertex_op_strings (dict): A dictionary specifying the basis of primitive operators at each site 
    Returns:
            Aop (dict): dictionary containing edge operators
            Bop (dict): dictionary containing vertex operators
            stabs (list): list of stabilizers
            qubits (int): number of qubits used in the mapping
    """
    if not np.allclose(adj + adj.T, 2*adj):
        print('Error: graph adjacency matrix is not symmetric')
        raise

    if qubit_map is None:
        qubit_map = {}
        min_unused_qubit = 0
        # for each mode
        for j in range(adj.shape[0]):
            qubit_map[j] = []
            valence = int(np.sum(adj[j,:]))
            qubits = int(np.ceil(valence/2))
            # assign some qubits to this mode
            for k in range(qubits):
                qubit_map[j].append(min_unused_qubit)
                min_unused_qubit+=1

    qubits = 0
    for j in range(adj.shape[0]):
        qubits = max( [qubits, max(qubit_map[j])]  )

    neighbors = generate_neighbors(adj)
    c = get_cops(adj,qubit_map,op_type)
    Aop = get_Aops(adj,neighbors,c)
    Aop_multi = get_Aops_multi(adj,neighbors,c)
    Bop = get_Bops(adj,c)
    stabs = get_stabs(adj,Aop,Aop_multi)

    return Aop, Bop, stabs, qubits



def custom_code_constructor(adj_target, adj_sim, mode_vertex_map, qubit_map=None, vertex_op_strings=None):
    """
    Constructor for generating encodings in the custom code framework

    The construction proceeds in two steps
    1) A GSE encoding over the simulation graph adj_sim is constructed, suppling the primitive Aops, Bops, and
        stabilizer generators
    2) An encoding of the target system specified by adj_target is constructed from the primitive operators, 
        Bops for unused vertices are added to the list of stabilizer generators
    
    Args:
        adj_target (numpy.ndarray): Upper triangular binary matrix
        adj_sim (numpy.ndarray): Upper triangular matrix, not necessarily binary
        mode_vertex_map (dict):
        (optional)
        qubit_map (dict):
        vertex_op_strings (dict):
    """
    adj_target = adj_target + adj_target.T
    G_sim = nx.Graph(adj_sim)
    prim_Aops, prim_Bops, stabs, qubits = GSE(adj_sim,qubit_map,vertex_op_strings)
    check_mapping(prim_Aops,prim_Bops,stabs)
    Aops = ({})
    Bops = ({})
    active_vertices = []
    frozen_vertices = [j for j in range(adj_sim.shape[0])]
    # get active and inactive (frozen) vertices
    for j in range(adj_target.shape[0]):
        frozen_vertices = [x for x in frozen_vertices if x != mode_vertex_map[j]]
        active_vertices.append(mode_vertex_map[j])
    # add frozen vertex Bops to stabilizers
    for j in frozen_vertices:
        stabs.append(prim_Bops[j])
    # pass the corresponding active vertex Bops
    for j in range(adj_target.shape[0]):
        Bops[j] = prim_Bops[mode_vertex_map[j]]
    # generate Aops from paths of Aops on the simulation graph
    for j in range(adj_target.shape[0]):
        for k in range(j+1,adj_target.shape[1]):
            if adj_target[j,k] >0:
                path = nx.shortest_path(G_sim,mode_vertex_map[j],mode_vertex_map[k])
                if len(path)==0:
                    print('Error: not connected')
                    raise
                path_edges = []

                """fully_ordered=False
                while fully_ordered is False:
                    for k in range(len(path)-1):
                        if adj_sim[ path[k], path[ (k+1)] ] == 0:
                            print(path)
                            print(adj_sim)
                            # make a swap
                            neighboring=False
                            counter = 2
                            while neighboring is False:
                                if adj_sim[path[k],path[(k+counter) % len(path)] ] == 1:
                                    path[(k+1) % len(path)], path[(k+counter) % len(path)] = path[(k+counter) % len(path)], path[(k+1) % len(path)]
                                    neighboring = True 
                                else:
                                    counter +=1
                    partially_ordered = True
                    for k in range(len(path)-1):
                        if adj_sim[path[k],path[(k+1)]] == 0:
                            partially_ordered = False
                            break
                    if partially_ordered:
                        fully_ordered = True
                    else: 
                        print('shuffling')
                        random.shuffle(path)"""

                for l in range(len(path)-1):
                    path_edges.append((path[l],path[l+1]))
                Aops[(j,k)] = prim_Aops[(path_edges[0][0],path_edges[0][1])]
                for l in range(1,len(path_edges)):
                    Aops[(j,k)] = Aops[(j,k)] * prim_Aops[(path_edges[l][0],path_edges[l][1])]
                Aops[(k,j)] = (-1)*Aops[(j,k)]
    

    return Aops, Bops, stabs, qubits


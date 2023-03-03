![pylove simultor](/pylove.svg)

## What is Pylove?

Pylove is a quantum circuit simulator specifically for noisy circuits acting on stabilizer codes.
The types of circuits it is build to handle consist of rotations generated by logical operators.
Noisy circuits are simulated via statevector simulation, where a mixed state density matrix is
constructed from a mixture of pure states coming from a specified number of shots.

For an $[[n,k,d]]$ stabilizer code, the simulation complexity is exponential in $k$ and poly 
in $(n-k)$. 

Pylove makes use of the OpenFermion software package for manipulating Pauli operators and Fermionic operators.

This simulator was used in a study of quantum error mitigation using encodings of fermionic systems into stabilizer codes. As such, included in this simulator code are a number of tools for generating fermionic encodings and encodings fermionic Hamiltonians into qubit Hamiltonians acting on a stabilizer code. Fermionic operators are mapped to logical operators of 
the code.

Let us now discuss the usage of the simulator code with a simple example that is given in more detail in `tutorial.ipynb` and example notebooks. Also refer to the documentation in `docs.ipynb`.

## Specifying a stabilizer code

Specify a stabilizer group by a list of generators given as OpenFermion `QubitOperator` objects

```python
stabs = [
    QubitOperator('X1 Z0 Z2 Z4'),
    QubitOperator('X3 Z2 Z4 Z8'),
    QubitOperator('X5 Z0 Z4 Z6'),
    QubitOperator('X7 Z4 Z6 Z8')
]
```
## Fixing the initial state

The initial state is specified to be the $(+1)$ eigenstate of a commuting collection of Pauli operators. Such a state is assumed to be easily prepared. As the stabilizers usually do not uniquely fix a particular pure state, we supplement the stabilizers with a collection of logical operators.

```python
log_ops = [
    QubitOperator('Z0'),
    QubitOperator('Z2', -1),
    QubitOperator('Z4'),
    QubitOperator('Z6', -1),
    QubitOperator('Z8', -1)]
```

## Building the circuit

The circuit is assumed to consist of a sequence of rotations
$$U_1 U_2\dots = \exp^{i\theta_1 Q_1}\exp^{i\theta_2 Q_2}\ldots$$
generated by logical operators $\{Q_1,Q_2,\ldots\}$.

Specified by a list of pairs, specifyin the angle and the logical operator
```python
circuit_schedule = [
    angle[0]*logical_op[0],
    angle[1]*logical_op[1],
    ...
]
```

In the above, `logical_op[i]` must all commute with all of the stabilizers

## Pauli noise channels

Pauli noise channels can be simulated, one specifies the Pauli Kraus operators as always as OpenFermion `QubitOperator` objects. Noise acting following CNOT gates can be constructed as below, where qubit 0 is the control qubit and qubit 1 is te target.

```python
my_gate_noise = [
    [QubitOperator(()), QubitOperator('Y0'), QubitOperator('Y0 Y1')],
    [.998, .001, .001]]
```

## Syndrome spaces

The simulator is originally intended to be used for studies of quantum error mitigation in the presence of a collection of $\mathbb{Z}_2$ symmetries or more specifically a stabilizer code. The code enables the simulation of postselection onto different subspaces:
* codespace: Only the codespace is kept, this is the cheapest simulation in terms of memory
* full: All syndrome spaces are kept, this is exponentially costly in the rank of the stabilizer group
* custom: One can specify which syndrome spaces to keep by giving the syndrome patters of the desired subspaces

## Simulation

The most important function is `pylove_simulation` which contains all the relevant functions for simulation.

```python
output_state = pylove_simulation(
    stabilizers=stabs,
    logical_operators=log_ops,
    quantum_circuit=circuit,
    shots=100
)
```

After the simulation, the resulting object can be manipulated. One can calculate expectation values of observables or simulation fidelities with  

```python
tr(output_state, output_state.ideal())
```

## Further reference

See the notebook `tutorial.ipynb` for an example of Pylove in action and refer to the documentation in `docs.ipynb`.
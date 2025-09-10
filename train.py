#!/usr/bin/env python
# coding: utf-8

# Quantum MNIST

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit_aer.primitives import EstimatorV2
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download, list_repo_files

algorithm_globals.random_seed = 12345

# Global variables - properly initialized
objective_func_vals = []
last_checkpoint = None
weights_history = []
error_rates = []
batch_boundaries = []  # Track where each batch ends for plotting
best_checkpoint = None
best_objective = float('inf')  # Track best objective value


def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        #qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        #qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def generalized_pooling(num_qubits, params):
    """
    Generalized n->1 pooling block.
    All qubits [0..n-2] are pooled into qubit (n-1), the sink.
    """
    qc = QuantumCircuit(num_qubits)
    sink = num_qubits - 1
    sources = list(range(num_qubits - 1))

    # Basis change on sink (consistent with 2->1 pooling)
    qc.rz(-np.pi/2, sink)
    # Entangle sink with all sources
    for q in sources:
        qc.cx(sink, q)

    qc.barrier()

    # Local rotations on sources (discarded qubits)
    for i, q in enumerate(sources):
        qc.rz(params[i], q)

    
    qc.ry(params[len(sources)], sink)

    # Multi-controlled X (generalized Toffoli) from all sources into sink
    qc.mcx(sources, sink)

    # Final learnable rotation on sink
    qc.ry(params[len(sources)+1], sink)

    return qc

def pool_layer(pools, param_prefix):
    """ poools is a 2d array where each row will be pooled into the list qbit of the group """
    num_qubits = sum(count for count in pools)
    param_count = sum(count + 1 for count in pools)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    circuit_index = 0
    sinks = []
    params = ParameterVector(param_prefix, param_count)
    for length in pools:
        sub_param_count = length + 1
        qbit_range = range(circuit_index, circuit_index + length)
        sub_params = params[param_index : (param_index + sub_param_count)]
        sub_circuit = generalized_pooling(length, sub_params)
        qc = qc.compose(sub_circuit, qbit_range)
        #qc.barrier()
        sinks.append(circuit_index + length - 1)
        param_index += sub_param_count
        circuit_index += length
    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return [qc, sinks]

def load_dataset_via_pandas(repo_id: str, revision: str = "main") -> DatasetDict:
    """
    Load a DatasetDict from a remote JSONL-based dataset repo on Hugging Face Hub
    using pandas as a backend to avoid ArrowIndexError.
    """
    files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    jsonl_files = [f for f in files if f.endswith(".jsonl")]

    dataset_dict = {}
    for f in jsonl_files:
        split = f.split("/")[-1].replace(".jsonl", "")
        
        local_path = hf_hub_download(repo_id, f, repo_type="dataset", revision=revision)
        
        df = pd.read_json(local_path, lines=True)
        dataset_dict[split] = Dataset.from_pandas(df, preserve_index=False)

    return DatasetDict(dataset_dict)

def get_topology_from_array(array):
    return [len([item for item in row if item != [None, None]]) for row in array]

def create_complex_feature_map(qbit_count):
    n_features = qbit_count * 2
    qc = QuantumCircuit(qbit_count)
    
    # Create parameters for real and imaginary parts
    all_params = [
        Parameter(
            f'{i:03d}'
        ) for i in range(n_features)
    ]
    
    for i in range(qbit_count):
        current_param = i * 2
        qc.rz(2 * all_params[current_param], i)
        qc.ry(2 * all_params[current_param + 1], i)
    
    return qc

def class_projector_observable(n_qubits, output_qubits, class_label):
    """
    Create a projector observable |class_label><class_label| on output_qubits
    - n_qubits: total qubits in circuit
    - output_qubits: list of qubit indices (LSB -> MSB)
    - class_label: integer, e.g., 0..9
    """
    binary = np.array(list(np.binary_repr(class_label, width=len(output_qubits))), dtype=int)
    # convert '0'/'1' to int array
    paulis = ["I"] * n_qubits
    coeff = 1.0

    # Start with coefficient 1 and multiply (I ± Z)/2 per bit
    terms = [("I"*n_qubits, 1.0)]  

    for qubit_idx, bit_val in zip(output_qubits, binary[::-1]):  # LSB->MSB
        new_terms = []
        for s, c in terms:
            # choose I+Z or I-Z
            s_list = list(s)
            s_list[n_qubits - 1 - qubit_idx] = "I"
            c_i = c * 0.5
            new_terms.append(("".join(s_list), c_i))
            
            s_list_z = list(s)
            s_list_z[n_qubits - 1 - qubit_idx] = "Z"
            c_z = c * (0.5 * (-1 if bit_val==1 else 1))
            new_terms.append(("".join(s_list_z), c_z))
        terms = new_terms

    # combine terms with same Pauli string
    from collections import defaultdict
    term_dict = defaultdict(float)
    for s, c in terms:
        term_dict[s] += c

    op = SparsePauliOp.from_list(list(term_dict.items())).simplify()
    return op

def callback_graph(weights, obj_func_eval, 
                   checkpoint_interval=5, 
                   filename_obj="objective_values.json",
                   filename_weights="weights_checkpoint.json",
                   filename_best="best_weights.json",
                   save_weights=True,
                   plot_filename="training_progress.png"):
    
    # Access global variables properly
    global objective_func_vals, weights_history, last_checkpoint, best_checkpoint, best_objective

    print("Callback graph")
    
    plt.rcParams["figure.figsize"] = (12, 6)
    
    # Append objective value
    objective_func_vals.append(obj_func_eval)
    
    # Check if this is the best model so far (lower objective is better for COBYLA)
    if obj_func_eval < best_objective:
        best_objective = obj_func_eval
        best_checkpoint = weights.copy() if hasattr(weights, 'copy') else np.array(weights)
        # Save best weights immediately
        with open(filename_best, "w") as f:
            best_weights_data = {
                "best_objective": float(best_objective),
                "best_weights": best_checkpoint.tolist() if hasattr(best_checkpoint, "tolist") else list(best_checkpoint),
                "iteration": len(objective_func_vals)
            }
            json.dump(best_weights_data, f, indent=2)
        print(f"*** New best model found! Objective = {obj_func_eval:.6f} ***")
    
    # Save objective values every iteration
    with open(filename_obj, "w") as f:
        json.dump(objective_func_vals, f, indent=2)
    
    # Save weights every `checkpoint_interval` steps if enabled
    if save_weights:
        weights_history.append(weights.tolist() if hasattr(weights, "tolist") else list(weights))
        if len(weights_history) % checkpoint_interval == 0:
            with open(filename_weights, "w") as f:
                json.dump(weights_history, f, indent=2)

    last_checkpoint = weights
    
    # Create plot and save to file (no clearing for command line)
    plt.figure(figsize=(12, 6))
    
    # Plot objective function values
    plt.subplot(1, 2, 1)
    plt.plot(range(len(objective_func_vals)), objective_func_vals, marker='o', label='Objective Value')
    
    # Mark the best objective
    if best_objective < float('inf'):
        best_iter = [i for i, val in enumerate(objective_func_vals) if val == best_objective][0]
        plt.plot(best_iter, best_objective, marker='*', markersize=15, color='gold', 
                label=f'Best: {best_objective:.3f}')
    
    # Add batch boundaries if they exist
    for i, boundary in enumerate(batch_boundaries):
        plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, 
                   label='Batch Boundary' if i == 0 else "")
    
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Objective Function Value vs Iteration")
    plt.grid(True)
    plt.legend()
    
    # Plot error rates if available
    plt.subplot(1, 2, 2)
    if error_rates:
        batch_numbers = range(1, len(error_rates) + 1)
        plt.plot(batch_numbers, [rate * 100 for rate in error_rates], 
                marker='s', color='red', label='Error Rate')
        plt.xlabel("Batch Number")
        plt.ylabel("Error Rate (%)")
        plt.title("Error Rate vs Batch")
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No error rates yet', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Error Rate vs Batch")
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    # Print progress to console
    print(f"Iteration {len(objective_func_vals)}: Objective = {obj_func_eval:.6f}")

def callback_step(evaluations, parameters, loss, stepsize, accepted):
    print(f"Step {evaluations}: Loss = {loss:.6f}, Stepsize = {stepsize:.6f}, Accepted = {accepted}")


def process_item(item):
    data = item["data"]
    # Transpose
    data = [[row[i] for row in data] for i in range(len(data[0]))]
    # Remove Nones
    data = [[pair for pair in row if pair[0]] for row in data]
    # Flatten
    data = [value for row in data for pair in row for value in pair]
    return {
        "features": data,
        "label": item["label"]
    }

def circuit_stats(circuit: QuantumCircuit, observables=None):
    """
    Print useful statistics for a QuantumCircuit to estimate resources and runtime.
    
    Parameters:
        circuit : QuantumCircuit
            The circuit to analyze.
        observables : list or SparsePauliOp, optional
            List of observables or a single observable to estimate number of Pauli terms.
    """
    print("===== Circuit Statistics =====")
    
    # Number of qubits
    n_qubits = circuit.num_qubits
    print("Number of qubits:", n_qubits)
    
    # Circuit depth
    depth = circuit.depth()
    print("Circuit depth:", depth)
    
    # Total number of instructions / gates
    instr_count = len(circuit.data)
    print("Total instructions in circuit:", instr_count)
    
    # Gate counts
    gate_count = circuit.count_ops()
    print("Gate counts:", gate_count)
    
    # Number of parameters
    param_count = len(circuit.parameters)
    print("Number of parameters:", param_count)
    
    # Estimated statevector memory
    mem_bytes = 2**n_qubits * 16  # complex128 = 16 bytes
    mem_mb = mem_bytes / 1024**2
    print(f"Estimated statevector memory: {mem_mb:.1f} MB")
    
    # Observables info
    if observables is not None:
        if hasattr(observables, "__len__") and not isinstance(observables, (str, QuantumCircuit)):
            num_observables = len(observables)
            total_pauli_terms = sum(len(o) for o in observables)
        else:  # single observable
            num_observables = 1
            total_pauli_terms = len(observables)
        print("Number of observables:", num_observables)
        print("Total number of Pauli terms:", total_pauli_terms)
    
    print("==============================")

def estimate_runtime(circuit, observables, n_samples=1, device='cpu'):
    """
    Rough estimate of runtime for simulating a circuit with given observables.

    Parameters:
        circuit : QuantumCircuit
            The circuit to analyze.
        observables : list or SparsePauliOp
            List of observables or single observable.
        n_samples : int
            Number of input samples to evaluate.
        device : str
            'cpu' or 'gpu' to scale estimate.

    Returns:
        None (prints estimated time)
    """
    n_qubits = circuit.num_qubits
    depth = circuit.depth()
    
    # Handle observables
    if hasattr(observables, "__len__") and not isinstance(observables, (str, QuantumCircuit)):
        total_terms = sum(len(o) for o in observables)
        num_observables = len(observables)
    else:
        total_terms = len(observables)
        num_observables = 1

    # Base time per Pauli term per sample (rough)
    # These are empirical approximations for statevector simulation
    base_ms = 10  # ms per term for shallow circuit on CPU
    # scale linearly with depth relative to a "unit" depth 4
    base_ms *= depth / 4  

    # Scale for GPU
    if device.lower() == 'gpu':
        base_ms /= 5  # assume ~5x speedup

    # Total per sample
    time_per_sample_sec = total_terms * base_ms / 1000  # ms → sec

    # Total for all samples
    total_time_sec = n_samples * time_per_sample_sec

    print("===== Runtime Estimate =====")
    print(f"Circuit depth: {depth}")
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of observables: {num_observables}")
    print(f"Total Pauli terms: {total_terms}")
    print(f"Number of samples: {n_samples}")
    print(f"Estimated time per sample: {time_per_sample_sec:.2f} sec")
    print(f"Estimated total time: {total_time_sec:.2f} sec ≈ {total_time_sec/60:.2f} min")
    print("============================")

def get_topology(sample_row):
    row_data = sample_row
    row_data = [[row[i] for row in row_data] for i in range(len(row_data[0]))]
    raw_complex_array = np.array([
        [
            complex(r if r is not None else np.nan, i if i is not None else np.nan)
            for r, i in row
        ]
        for row in row_data
    ])
    raw_complex_array = raw_complex_array.T
    topology = get_topology_from_array(row_data)
    raw_complex_array = raw_complex_array.T # <- Transpose
    flat = raw_complex_array.flatten()
    flat_filtered = flat[~(np.isnan(flat.real) & np.isnan(flat.imag))]
    circuit_size = len(flat_filtered)
    print("Flatened 1d array size", flat.shape)
    print("Flatened 1d array size minus Nan", flat_filtered.shape)
    print("\nOriginal\n", raw_complex_array)
    print("\nFlattened Array minus Nan\n", flat_filtered)
    print("Qbit length", circuit_size)
    print("Shape of the array", raw_complex_array.shape)
    return [topology, circuit_size]

def get_qnn(topology, circuit_size, class_count, max_threads=20,):
    feature_map = create_complex_feature_map(qbit_count=circuit_size)
    ansatz = QuantumCircuit(circuit_size, name="Ansatz")
    # First Convolutional Layer
    convolution = conv_layer(circuit_size, "c1")
    ansatz.compose(convolution, list(range(circuit_size)), inplace=True)
    # First Pooling Layer
    [pool, sinks] = pool_layer(topology, "p1")
    ansatz.compose(pool, list(range(circuit_size)), inplace=True)
    # Combining the feature map and ansatz
    circuit = QuantumCircuit(circuit_size, name="QNN Circuit")
    circuit.compose(feature_map, range(circuit_size), inplace=True)
    circuit.compose(ansatz, range(circuit_size), inplace=True)
    observables = [
        class_projector_observable(circuit_size, sinks, k) for k in range(class_count)
    ]
    estimator = EstimatorV2(options = {
        "backend_options": {
            "method": "statevector",                # exact simulation
            "max_parallel_threads": max_threads,    # multi-thread CPU
        },
        "run_options": {
            "shots": 0                              # statevector -> no shots
        },
        "default_precision": 0.0
    })
    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

def train(ds, batches=1, batch_size=20, test_batch_size=10, class_count=10):
    global batch_boundaries, error_rates, objective_func_vals

    #ds = ds.train_test_split(test_size=0.1, seed=42)
    train_dataset = ds["train"]
    test_dataset = ds["test"]
    sample_row = ds["train"][0]["data"]
    
    print("Sample row topology data:", sample_row)
    [topology, circuit_size] = get_topology(sample_row)
    qnn = get_qnn(topology, circuit_size, class_count)
    
    circuit_stats(qnn.circuit.decompose(), qnn.observables)
    estimate_runtime(qnn.circuit.decompose(), qnn.observables, n_samples=20, device='cpu')
    
    for batch in range(batches):
        print(f"=== Batch {batch+1}/{batches} ===")
        # Fixed: proper batch indexing - select from original dataset directly
        train_start = batch * batch_size
        train_end = min((batch + 1) * batch_size, len(train_dataset))
        test_start = batch * test_batch_size
        test_end = min((batch + 1) * test_batch_size, len(test_dataset))
        
        print(f"Training on samples {train_start}-{train_end-1}, Testing on samples {test_start}-{test_end-1}")
        
        # Select directly from the full datasets, not from already selected subsets
        train = train_dataset.select(range(train_start, train_end)).map(process_item)
        test = test_dataset.select(range(test_start, test_end)).map(process_item)
        
        # Mark batch boundary for plotting
        if batch > 0:  # Don't mark boundary before first batch
            batch_boundaries.append(len(objective_func_vals))
        
        classifier = train_pseudoepoch(train, qnn, maxiter=3, initial_point=last_checkpoint)
        err_rate = test_pseudoepoch(test, classifier)
        error_rates.append(err_rate)
        print(f"Error rate for batch {batch+1}: {err_rate*100:.2f}%")
        
        # Update the final plot with current error rates
        callback_graph(last_checkpoint if last_checkpoint is not None else np.zeros(1), 
                      objective_func_vals[-1] if objective_func_vals else 0,
                      save_weights=False)
    
    print(f"Average error rate over {batches} batches: {np.mean(error_rates)*100:.2f}%")
    print(f"Best objective value achieved: {best_objective:.6f}")
    print("Best model weights saved to 'best_weights.json'")
    
def train_pseudoepoch(items, qnn, maxiter=20, initial_point=None):
    global last_checkpoint

    def make_schedule(init=0.2, decay=0.99):
        def generator():
            i = 0
            while True:
                yield init * (decay**i)
                i += 1
        return generator  # return the *function*, not the iterator
    
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=SPSA(
            maxiter=maxiter, 
            learning_rate = make_schedule(init=0.2, decay=0.99), 
            perturbation = make_schedule(init=0.1, decay=0.99),
            callback=callback_step,
        ),
        callback=callback_graph,
        initial_point=initial_point,
    )

    features = [sample["features"] for sample in items]
    labels = [sample["label"] for sample in items]
    x = np.asarray(features)
    y = np.asarray(labels)

    classifier.fit(x, y)

    # Update global checkpoint with the trained weights
    last_checkpoint = classifier.weights

    # score classifier
    print(f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%")
    return classifier

def test_pseudoepoch(test_ds, classifier):
    test_features = [sample["features"] for sample in test_ds]
    test_labels = [sample["label"] for sample in test_ds]
    y_predict = classifier.predict(test_features)
    x = np.asarray(test_features)
    y = np.asarray(test_labels)
    print(f"Accuracy from the test data : {np.round(100 * classifier.score(x, y), 2)}%")
    print("Predicted labels:", y_predict)
    print("True labels:", y)
    err_rate = np.sum(y != y_predict) / len(y)
    return err_rate

def main():
    ds = load_dataset_via_pandas("ljcamargo/quantum_mnist")
    # Increase batches to run multiple pseudoepochs
    train(ds, batches=1, batch_size=200, test_batch_size=1, class_count=10)

if __name__ == "__main__":
    main()
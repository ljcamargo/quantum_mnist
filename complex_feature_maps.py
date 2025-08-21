"""
Complex Feature Maps Library for Qiskit
========================================

A collection of quantum feature maps designed to handle complex-valued classical data,
extending beyond the standard real-valued feature maps in Qiskit.

Classes:
- ComplexFeatureMap: Basic complex feature map using real/imaginary decomposition
- ComplexZZFeatureMap: Complex feature map with ZZ-style entangling interactions
- PolarFeatureMap: Feature map using polar (magnitude/phase) representation
- PolarZZFeatureMap: Polar feature map with entangling interactions

Utility Functions:
- create_complex_feature_map: Simple parameterized complex feature map
- create_polar_feature_map: Simple parameterized polar feature map
- get_entangling_pairs: Generate entanglement topologies
- bind_complex_data: Helper for parameter binding
- preprocess_complex_data: Data preprocessing for ML integration
"""

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import BlueprintCircuit
import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Any


class ComplexFeatureMap(BlueprintCircuit):
    """
    Basic complex feature map encoding complex data as separate real/imaginary rotations.
    Uses RY for real parts and RZ for imaginary parts on each qubit.
    """
    
    def __init__(self, feature_dimension: int, reps: int = 1, data_map_func=None):
        self._feature_dimension = feature_dimension
        self._reps = reps
        self._data_map_func = data_map_func or (lambda x: x)
        
        # Create parameters: alternating real/imaginary for each feature
        self._parameters = []
        for rep in range(reps):
            for i in range(feature_dimension):
                self._parameters.append(Parameter(f'x_real_{i}_rep_{rep}'))
                self._parameters.append(Parameter(f'x_imag_{i}_rep_{rep}'))
        
        super().__init__(name='ComplexFeatureMap')
    
    @property
    def num_qubits(self) -> int:
        return self._feature_dimension
    
    @property 
    def parameters(self):
        return set(self._parameters)
    
    def _build(self):
        """Build circuit with real->RY and imaginary->RZ rotations"""
        param_idx = 0
        
        for rep in range(self._reps):
            for i in range(self._feature_dimension):
                real_param = self._parameters[param_idx]
                imag_param = self._parameters[param_idx + 1]
                param_idx += 2
                
                # Real part controls Y rotation, imaginary part controls Z rotation
                self.ry(2 * real_param, i)
                self.rz(2 * imag_param, i)
                
            # Add simple entangling layer between repetitions
            if rep < self._reps - 1:
                for i in range(self._feature_dimension - 1):
                    self.cx(i, i + 1)


class ComplexZZFeatureMap(BlueprintCircuit):
    """
    Complex feature map with ZZ-style entangling interactions.
    Encodes complex products between features using RZZ and optionally RYY gates.
    """
    
    def __init__(self, feature_dimension: int, reps: int = 1, entanglement: Union[str, List] = 'linear',
                 include_phase_interactions: bool = True):
        self._feature_dimension = feature_dimension
        self._reps = reps
        self._entanglement = entanglement
        self._include_phase_interactions = include_phase_interactions
        
        # Create parameters
        self._parameters = []
        for rep in range(reps):
            for i in range(feature_dimension):
                self._parameters.append(Parameter(f'x_real_{i}_rep_{rep}'))
                self._parameters.append(Parameter(f'x_imag_{i}_rep_{rep}'))
        
        super().__init__(name='ComplexZZFeatureMap')
    
    @property
    def num_qubits(self) -> int:
        return self._feature_dimension
    
    @property 
    def parameters(self):
        return set(self._parameters)
    
    def _build(self):
        """Build circuit with single-qubit rotations and complex ZZ interactions"""
        param_idx = 0
        
        for rep in range(self._reps):
            # Single qubit rotations
            real_params = []
            imag_params = []
            
            for i in range(self._feature_dimension):
                real_param = self._parameters[param_idx]
                imag_param = self._parameters[param_idx + 1]
                real_params.append(real_param)
                imag_params.append(imag_param)
                param_idx += 2
                
                self.ry(2 * real_param, i)
                self.rz(2 * imag_param, i)
            
            # Entangling layer with complex products
            entangling_pairs = self._get_entangling_pairs()
            
            for i, j in entangling_pairs:
                # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                real_product = (real_params[i] * real_params[j] - 
                               imag_params[i] * imag_params[j])
                
                self.rzz(2 * real_product, i, j)
                
                if self._include_phase_interactions:
                    imag_product = (real_params[i] * imag_params[j] + 
                                   imag_params[i] * real_params[j])
                    self.ryy(2 * imag_product, i, j)
    
    def _get_entangling_pairs(self) -> List[Tuple[int, int]]:
        """Generate qubit pairs for entanglement based on topology"""
        if self._entanglement == 'linear':
            return [(i, i+1) for i in range(self._feature_dimension-1)]
        elif self._entanglement == 'full':
            return [(i, j) for i in range(self._feature_dimension) 
                    for j in range(i+1, self._feature_dimension)]
        elif self._entanglement == 'circular':
            pairs = [(i, i+1) for i in range(self._feature_dimension-1)]
            pairs.append((self._feature_dimension-1, 0))
            return pairs
        else:
            return self._entanglement  # assume custom list of pairs


class PolarFeatureMap(BlueprintCircuit):
    """
    Feature map using polar representation of complex numbers.
    Encodes magnitude with RY and phase with RZ rotations.
    """
    
    def __init__(self, feature_dimension: int, reps: int = 1):
        self._feature_dimension = feature_dimension
        self._reps = reps
        
        # Create parameters for magnitude and phase
        self._parameters = []
        for rep in range(reps):
            for i in range(feature_dimension):
                self._parameters.append(Parameter(f'r_{i}_rep_{rep}'))      # magnitude
                self._parameters.append(Parameter(f'phi_{i}_rep_{rep}'))    # phase
        
        super().__init__(name='PolarFeatureMap')
    
    @property
    def num_qubits(self) -> int:
        return self._feature_dimension
    
    @property 
    def parameters(self):
        return set(self._parameters)
    
    def _build(self):
        """Build circuit with magnitude->RY and phase->RZ rotations"""
        param_idx = 0
        
        for rep in range(self._reps):
            for i in range(self._feature_dimension):
                mag_param = self._parameters[param_idx]
                phase_param = self._parameters[param_idx + 1]
                param_idx += 2
                
                self.ry(2 * mag_param, i)    # magnitude controls Y rotation
                self.rz(2 * phase_param, i)  # phase controls Z rotation
                
            # Simple entangling layer
            if rep < self._reps - 1:
                for i in range(self._feature_dimension - 1):
                    self.cx(i, i + 1)


class PolarZZFeatureMap(BlueprintCircuit):
    """
    Polar feature map with ZZ-style entangling interactions.
    Uses magnitude products and phase sums/differences for entanglement.
    """
    
    def __init__(self, feature_dimension: int, reps: int = 1, 
                 entanglement: Union[str, List] = 'linear'):
        self._feature_dimension = feature_dimension
        self._reps = reps
        self._entanglement = entanglement
        
        # Create parameters
        self._parameters = []
        for rep in range(reps):
            for i in range(feature_dimension):
                self._parameters.append(Parameter(f'r_{i}_rep_{rep}'))
                self._parameters.append(Parameter(f'phi_{i}_rep_{rep}'))
        
        super().__init__(name='PolarZZFeatureMap')
    
    @property
    def num_qubits(self) -> int:
        return self._feature_dimension
    
    @property 
    def parameters(self):
        return set(self._parameters)
    
    def _build(self):
        """Build circuit with polar interactions"""
        param_idx = 0
        
        for rep in range(self._reps):
            mag_params = []
            phase_params = []
            
            # Single qubit rotations
            for i in range(self._feature_dimension):
                mag_param = self._parameters[param_idx]
                phase_param = self._parameters[param_idx + 1]
                mag_params.append(mag_param)
                phase_params.append(phase_param)
                param_idx += 2
                
                self.ry(2 * mag_param, i)
                self.rz(2 * phase_param, i)
            
            # Entangling interactions
            entangling_pairs = self._get_entangling_pairs()
            
            for i, j in entangling_pairs:
                # Magnitude product interaction
                mag_interaction = mag_params[i] * mag_params[j]
                self.rzz(2 * mag_interaction, i, j)
                
                # Phase interactions
                phase_sum = phase_params[i] + phase_params[j]
                phase_diff = phase_params[i] - phase_params[j]
                
                self.ryy(2 * phase_sum, i, j)
                self.rxx(2 * phase_diff, i, j)
    
    def _get_entangling_pairs(self) -> List[Tuple[int, int]]:
        """Generate entangling pairs based on topology"""
        if self._entanglement == 'linear':
            return [(i, i+1) for i in range(self._feature_dimension-1)]
        elif self._entanglement == 'full':
            return [(i, j) for i in range(self._feature_dimension) 
                    for j in range(i+1, self._feature_dimension)]
        elif self._entanglement == 'circular':
            pairs = [(i, i+1) for i in range(self._feature_dimension-1)]
            pairs.append((self._feature_dimension-1, 0))
            return pairs
        else:
            return self._entanglement


# Utility Functions
def create_complex_feature_map(n_features: int) -> QuantumCircuit:
    """
    Simple parameterized circuit for complex data encoding.
    Returns circuit with separate real/imaginary parameters.
    """
    qc = QuantumCircuit(n_features)
    
    real_params = [Parameter(f'x_real_{i}') for i in range(n_features)]
    imag_params = [Parameter(f'x_imag_{i}') for i in range(n_features)]
    
    for i in range(n_features):
        qc.ry(2 * real_params[i], i)
        qc.rz(2 * imag_params[i], i)
    
    return qc


def create_polar_feature_map(n_features: int) -> QuantumCircuit:
    """
    Simple parameterized circuit for polar complex data encoding.
    Returns circuit with magnitude/phase parameters.
    """
    qc = QuantumCircuit(n_features)
    
    mag_params = [Parameter(f'r_{i}') for i in range(n_features)]
    phase_params = [Parameter(f'phi_{i}') for i in range(n_features)]
    
    for i in range(n_features):
        qc.ry(2 * mag_params[i], i)
        qc.rz(2 * phase_params[i], i)
    
    return qc


def get_entangling_pairs(n_qubits: int, entanglement: Union[str, List]) -> List[Tuple[int, int]]:
    """
    Generate entangling qubit pairs based on topology specification.
    
    Args:
        n_qubits: Number of qubits
        entanglement: 'linear', 'full', 'circular', or custom list of pairs
    """
    if entanglement == 'linear':
        return [(i, i+1) for i in range(n_qubits-1)]
    elif entanglement == 'full':
        return [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
    elif entanglement == 'circular':
        pairs = [(i, i+1) for i in range(n_qubits-1)]
        pairs.append((n_qubits-1, 0))
        return pairs
    else:
        return entanglement


def bind_complex_data(feature_map, complex_array: List[complex], rep: int = 0) -> Dict[Parameter, float]:
    """
    Create parameter binding dictionary for complex data.
    Works with ComplexFeatureMap and ComplexZZFeatureMap.
    """
    param_dict = {}
    
    for i, z in enumerate(complex_array):
        param_dict[f'x_real_{i}_rep_{rep}'] = z.real
        param_dict[f'x_imag_{i}_rep_{rep}'] = z.imag
    
    return param_dict


def bind_polar_data(feature_map, complex_array: List[complex], rep: int = 0) -> Dict[Parameter, float]:
    """
    Create parameter binding dictionary for polar complex data.
    Works with PolarFeatureMap and PolarZZFeatureMap.
    """
    param_dict = {}
    
    for i, z in enumerate(complex_array):
        param_dict[f'r_{i}_rep_{rep}'] = abs(z)
        param_dict[f'phi_{i}_rep_{rep}'] = np.angle(z)
    
    return param_dict


def preprocess_complex_data(complex_data: List[List[complex]], encoding: str = 'rectangular') -> np.ndarray:
    """
    Preprocess complex data for quantum machine learning integration.
    
    Args:
        complex_data: List of complex arrays (samples)
        encoding: 'rectangular' (real/imag) or 'polar' (mag/phase)
    
    Returns:
        Preprocessed real-valued array suitable for parameter binding
    """
    processed = []
    
    for sample in complex_data:
        if encoding == 'rectangular':
            real_imag_pairs = []
            for z in sample:
                real_imag_pairs.extend([z.real, z.imag])
            processed.append(real_imag_pairs)
        elif encoding == 'polar':
            mag_phase_pairs = []
            for z in sample:
                mag_phase_pairs.extend([abs(z), np.angle(z)])
            processed.append(mag_phase_pairs)
        else:
            raise ValueError("encoding must be 'rectangular' or 'polar'")
    
    return np.array(processed)


def create_complex_zz_feature_map(n_features: int, entanglement: str = 'linear') -> QuantumCircuit:
    """
    Simple function to create a complex ZZ feature map circuit.
    For more advanced usage, use the ComplexZZFeatureMap class.
    """
    qc = QuantumCircuit(n_features)
    
    real_params = [Parameter(f'x_real_{i}') for i in range(n_features)]
    imag_params = [Parameter(f'x_imag_{i}') for i in range(n_features)]
    
    # Single qubit rotations
    for i in range(n_features):
        qc.ry(2 * real_params[i], i)
        qc.rz(2 * imag_params[i], i)
    
    # Add ZZ interactions
    entangling_pairs = get_entangling_pairs(n_features, entanglement)
    
    for i, j in entangling_pairs:
        # Complex product real part: Re(z_i * z_j) = Re_i*Re_j - Im_i*Im_j
        real_product = real_params[i] * real_params[j] - imag_params[i] * imag_params[j]
        qc.rzz(2 * real_product, i, j)
        
        # Complex product imaginary part: Im(z_i * z_j) = Re_i*Im_j + Im_i*Re_j
        imag_product = real_params[i] * imag_params[j] + imag_params[i] * real_params[j]
        qc.ryy(2 * imag_product, i, j)
    
    return qc
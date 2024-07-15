import numpy as np

from pyvqnet.qnn.vqc.qcircuit import isingxx, isingyy, isingzz, u3, i, x, hadamard, s, I, PauliX, PauliY, PauliZ
from pyvqnet.qnn.vqc.qmachine import QMachine
from pyvqnet.qnn.vqc.qmeasure import SparseHamiltonian, expval
from pyvqnet.qnn.vqc import VQC_AngleEmbedding
from pyvqnet.tensor import tensor, QTensor
from pyvqnet.nn import Module, Parameter
from pyvqnet.dtype import *

CHANGE_OF_BASIS = np.array([[1.0, 1.0, 0.0, 0.0], [-1.0, -1.0, 2.0, 0.0], [-1.0, -1.0, 0.0, 2.0], [1.0, -1.0, 0.0, 0.0]])
H = QTensor([[1.+0.j, 0.+0.j,], [0.+0.j, 0.+0.j,]], dtype=kcomplex64)
o = tensor.dense_to_csr(H)

num_wires = 25
num_qubits = 13
wires = list(range(num_wires))

def obs_list(wire):
    return [I(wires = wire), PauliX(wires = wire), PauliY(wires = wire), PauliZ(wires = wire)]

def process_tensor(results, n_prep, n_meas):
    results = tensor.reshape(results, (4, -1))
    results = tensor.transpose(results, [1, 0])
    for _ in range(n_meas):
        final_result = results

    for _ in range(n_prep):
        for i in range(len(CHANGE_OF_BASIS)):
            change = tensor.to_tensor(CHANGE_OF_BASIS[i])
            if i == 0:
                temp_result = tensor.sums(tensor.mul(change, results), axis=1)
                final_result = tensor.reshape(temp_result, (-1, 1))
            else:
                temp_result = tensor.sums(tensor.mul(change, results), axis=1)
                temp_result = tensor.reshape(temp_result, (-1, 1))
                final_result = tensor.concatenate([final_result, temp_result], axis=1)
    final_result *= np.power(2, -(n_meas + n_prep) / 2)
    return final_result

def circuit_front(qm, features, wires, weights):
    VQC_AngleEmbedding(features[:, :num_qubits], wires, qm, rotation = 'Y')
    for ind, w in enumerate(wires):
        if ind < num_qubits - 1:
            if ind == 0:
                count = 12*num_wires-12
                u3(qm, wires[ind], tensor.to_tensor(weights[count+3*ind: count+3*ind+3]))
            count = 3*num_wires-3
            u3(qm, wires[ind+1], tensor.to_tensor(weights[count+3*ind: count+3*ind+3]))
            isingxx(qm, [w, wires[ind+1]], weights[3*ind+0])
            isingyy(qm, [w, wires[ind+1]], weights[3*ind+1])
            isingzz(qm, [w, wires[ind+1]], weights[3*ind+2])
            count = 6*num_wires-6
            u3(qm, wires[ind], tensor.to_tensor(weights[count+3*ind: count+3*ind+3]))
            count = 9*num_wires-9
            u3(qm, wires[ind+1], tensor.to_tensor(weights[count+3*ind: count+3*ind+3]))
    return qm

def circuit_back(qm, features, wires, weights, idx):
    VQC_AngleEmbedding(features[:, num_qubits:], wires[1:], qm, rotation = 'Y')

    if idx == 0:
        i(qm, wires[0])
    elif idx == 1:
        x(qm, wires[0])
    elif idx == 2:
        hadamard(qm, wires[0])
    elif idx == 3:
        hadamard(qm, wires[0])
        s(qm, wires[0])

    for ind, w in enumerate(wires):
        if ind >= num_qubits - 1 and ind < num_wires - 1:
            if ind == num_qubits - 1:
                count = 12*num_wires-12
                u3(qm, wires[ind-num_qubits+1], tensor.to_tensor(weights[count+3*ind: count+3*ind+3]))
            count = 3*num_wires-3
            u3(qm, wires[ind-num_qubits+2], tensor.to_tensor(weights[count+3*ind: count+3*ind+3]))
            isingxx(qm, [w-num_qubits+1, wires[ind-num_qubits+2]], weights[3*ind+0])
            isingyy(qm, [w-num_qubits+1, wires[ind-num_qubits+2]], weights[3*ind+1])
            isingzz(qm, [w-num_qubits+1, wires[ind-num_qubits+2]], weights[3*ind+2])
            count = 6*num_wires-6
            u3(qm, wires[ind-num_qubits+1], tensor.to_tensor(weights[count+3*ind: count+3*ind+3]))
            count = 9*num_wires-9
            u3(qm, wires[ind-num_qubits+2], tensor.to_tensor(weights[count+3*ind: count+3*ind+3]))
    return qm

def q_net(weights, features):
    weights = tensor.squeeze(weights)
    #measure method
    for i in range(len(obs_list(num_qubits - 1))):
        qm_front = QMachine(num_qubits)
        circuit_front(qm_front, features, wires, weights)
        if i == 0:
            result_front = expval(qm_front, wires[num_qubits - 1], obs_list(num_qubits - 1)[i])
        else:
            result_front = tensor.concatenate([result_front, expval(qm_front, wires[num_qubits - 1], obs_list(num_qubits - 1)[i])])
    
    #prepare method
    for i in range(len(obs_list(num_qubits - 1))):
        qm_back = QMachine(num_qubits)
        circuit_back(qm_back, features, wires, weights, i)
        measure = SparseHamiltonian(obs = {"observables": o, "wires":(num_qubits - 1,)})
        if i == 0: 
            result_back = measure(qm_back)
        else:
            result_back = tensor.concatenate([result_back, measure(qm_back)])
    
    result_front = process_tensor(result_front, 0, 1)
    result_back = process_tensor(result_back, 1, 0)
    result_tensor = tensor.sums(tensor.mul(result_front, result_back), axis=1)
    return result_tensor

class QModel(Module):
    def __init__(self):
        super(QModel, self).__init__()
        self.q_net = q_net
        self.weights = Parameter(shape=(1, 15*num_wires-15), dtype=kcomplex64)

    def forward(self, input):
        return self.q_net(self.weights, input)

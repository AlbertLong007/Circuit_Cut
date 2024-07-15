import numpy as np
from tqdm.auto import tqdm

from pyvqnet.tensor import tensor, QTensor
from pyvqnet.data import data_generator as dataloader
from pyvqnet.utils.storage import load_parameters

from cutcircuit import QModel

n_test = 1000
batch_size = 64

def eval(n_test, batch_size):
    model = QModel()
    model_para = load_parameters("weights.pt")
    model.load_state_dict(model_para)

    test_data = np.load("./data/test_data.npy").reshape(-1, 25)
    test_labels = np.load("./data/test_label.npy")[:,0].flatten()
    test_data = test_data / np.linalg.norm(test_data, axis=1).reshape((-1, 1))
    test_data = test_data[:n_test]
    test_labels = test_labels[:n_test]
    test_acc_epochs = []
    
    model.eval()
    for x_test, y_test in tqdm(dataloader(test_data, test_labels, batch_size, False)):
        x_test, y_test = tensor.to_tensor(x_test), tensor.to_tensor(y_test)

        test_out = model(x_test)
        test_pred = tensor.greater(test_out, QTensor(0.5))
        test_acc = tensor.equal(test_pred, y_test)

        if len(test_acc_epochs) == 0:
            test_acc_epoch = test_acc.to_numpy()
        else:
            test_acc_epoch = np.concatenate((test_acc_epoch, test_acc.to_numpy()))
        test_acc_epochs.append(np.mean(test_acc_epoch))

    print("Acc: ")
    print(np.mean(test_acc_epoch))

eval(n_test, batch_size)

import numpy as np
from tqdm.auto import tqdm

from pyvqnet.tensor import tensor, QTensor
from pyvqnet.optim import Adam
from pyvqnet.data import data_generator as dataloader
from pyvqnet.utils.storage import save_parameters

from cutcircuit import QModel

n_test = 1000
n_epochs = 100
batch_size = 64

def train(n_train, n_epochs, batch_size):
    model = QModel()
    optimizer = Adam(model.parameters(), lr=0.1)

    train_data = np.load("./data/train_data.npy").reshape(-1, 25)
    test_data = np.load("./data/test_data.npy").reshape(-1, 25)
    train_labels = np.load("./data/train_label.npy")[:,0].flatten()
    test_labels = np.load("./data/test_label.npy")[:,0].flatten()
    train_data = train_data / np.linalg.norm(train_data, axis=1).reshape((-1, 1))
    test_data = test_data / np.linalg.norm(test_data, axis=1).reshape((-1, 1))
    train_data = train_data[:n_train]
    test_data = test_data[:n_test]
    train_labels = train_labels[:n_train]
    test_labels = test_labels[:n_test]
    train_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], []
    max_acc = 0

    for _ in tqdm(range(n_epochs)):
        model.train()
        for x_train, y_train in dataloader(train_data, train_labels, batch_size, True):
            x_train, y_train = tensor.to_tensor(x_train), tensor.to_tensor(y_train)
            optimizer.zero_grad()

            train_out = model(x_train)
            train_cost = -y_train * tensor.log(train_out) - (1 - y_train) * tensor.log(1 - train_out)
            train_cost.backward()
            optimizer.step()
            train_pred = tensor.greater(train_out, QTensor(0.5))
            train_acc = tensor.equal(train_pred, y_train)
            
            if len(train_acc_epochs) == 0:
                train_acc_epoch = train_acc.to_numpy()
                train_cost_epoch = train_cost.to_numpy()
            else:
                train_acc_epoch = np.concatenate((train_acc_epoch, train_acc.to_numpy()))
                train_cost_epoch = np.concatenate((train_cost_epoch, train_cost.to_numpy()))
            train_acc_epochs.append(np.mean(train_acc_epoch))
            train_cost_epochs.append(np.mean(train_cost_epoch))

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

        final_acc = np.mean(test_acc_epoch)
        if final_acc > max_acc:
            max_acc = final_acc
            save_parameters(model.state_dict(), "weights.pt")

# run training for multiple sizes
train_sizes = [2, 50, 200, 1000, 5000]
for n_train in train_sizes:
    train(n_train, n_epochs, batch_size)

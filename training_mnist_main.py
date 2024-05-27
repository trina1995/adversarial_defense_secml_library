import pandas as pd
import torch
from torch import nn
from secml.data.loader import CDataLoaderMNIST
from torch import optim
import secml
secml.settings.SECML_PYTORCH_USE_CUDA = True
from secml.ml.classifiers import CClassifierPyTorch
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers.reject import CClassifierDNR
from secml.array import CArray
from collections import OrderedDict
from secml.data.selection import CPSRandom
from secml.data.splitter import CDataSplitterShuffle
import numpy as np
from mnist_functions import Flatten, MNISTCNN, MNISTCNNRBF, MNISTPyTorch, RBFNetwork, nr_performance, dnr_performance, white_box_attack, black_box_attack
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

def main():
    loader = CDataLoaderMNIST()

    # Data Preparation
    random_state = 999

    n_train = 30000 # Number of training set samples
    n_test = 1000  # Number of test set samples

    digits = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    train = loader.load('training', digits=digits)
    dnn_test = loader.load('testing', digits=digits)
    dnn_train, dnr_nr_train = train[:n_train, :], train[n_train:, :]

    # Normalize the features in `[0, 1]`
    dnn_train.X /= 255
    dnr_nr_train.X /= 255
    dnn_test.X /= 255
    print("Input data shape:", dnn_train.X.shape)
    print("Target labels shape:", dnn_train.Y.shape)
    print("Input data shape:", dnr_nr_train.X.shape)
    print("Target labels shape:", dnr_nr_train.Y.shape)
    print("Input data shape:", dnn_test.X.shape)
    print("Target labels shape:", dnn_test.Y.shape)

    ####################### Training and loading DNN model ########################

    net = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0)

    mnist_model_1 = CClassifierPyTorch(model=net,
                            loss=criterion,
                            optimizer=optimizer,
                            epochs=50,
                            batch_size=128,
                            input_shape=(1, 28, 28),
                            preprocess=None,
                            softmax_outputs=False)

    #mnist_model_1.fit(dnn_train.X, dnn_train.Y)
    #torch.save(mnist_model_1, "model_mnist/mnist.pt")

    # ############################# Surrogate Model###################################

    model = MNISTPyTorch().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    clf_dnn = CClassifierPyTorch(model = model, 
                                 loss=criterion, 
                                 optimizer=optimizer, 
                                 input_shape=(1, 28, 28), 
                                 epochs=10)
    clf_dnn.fit(dnn_train.X, dnn_train.Y)
    torch.save(clf_dnn, "model_mnist/surrogate_model.pt")

    # Loading the DNN model
    loaded_pytorch_model = torch.load("model_mnist/mnist.pt")

    # # ##################################### Training DNR model ######################

    layers = ['features:relu2', 'features:relu3', 'features:relu4']
    params = {'features:relu4.C': 1, 'features:relu2.kernel.gamma': 1e-3}
    path = "model_mnist/dnr_mnist_rej.pkl"
    rej = True
    dnr_training = dnr_performance(dnr_nr_train, dnn_test, loaded_pytorch_model, rej, 
                                   layers, params, path, num_runs=5)

    ################################# Training NR model ###########################

    layers = ['features:relu4']
    path = "model_mnist/nr_mnist_rej.pkl"
    rej = True
    nr_training = nr_performance(dnr_nr_train, dnn_test, loaded_pytorch_model, rej, 
                                layers, path, num_runs=5)

    # Load the RBF model
    mnist_rbf_model = torch.load("model_mnist/mnist_rbf_model.pt")

    print("Training NR done")
    ########################### Training DNR-RBF model ############################

    layers = ['features:relu2', 'features:relu3', 'features:relu4']
    params = {'features:relu4.C': 1, 'features:relu2.kernel.gamma': 1e-3}
    path = "model_mnist/dnr_rbf_mnist_rej.pkl"
    rej = True
    dnr_training = dnr_performance(dnr_nr_train, dnn_test, mnist_rbf_model, rej, 
                                   layers, params, path, num_runs=5)
    print("Training DNR-RBF done")
    ######################### Training NR-RBF model ###############################

    layers = ['features:relu4']
    path = "model_mnist/'nr_rbf_mnist_rej.pkl"
    rej = True
    nr_training = nr_performance(dnr_nr_train, dnn_test, mnist_rbf_model, rej, 
                                 layers, path, num_runs=5)
    print("Training NR-RBF done")

if __name__ == "__main__":
    main()

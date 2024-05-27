import pandas as pd
import numpy as np
import secml
import torch
from torch import nn
from secml.data.loader.c_dataloader_cifar import CDataLoaderCIFAR10
from torch import optim
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers.reject import CClassifierDNR
from collections import OrderedDict
from secml.data.selection import CPSRandom
from secml.data.splitter import CDataSplitterShuffle
from secml.ml.features.normalization import CNormalizerMinMax
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from cifar10_functions import CIFAR10CNN, CIFAR10CNNRBF, RBF_Layer, nr_performance, dnr_performance
import matplotlib.pyplot as plt

secml.settings.SECML_PYTORCH_USE_CUDA=True
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

def main():
    train_ds, test_ds = CDataLoaderCIFAR10().load()

    # Data Preparation
    random_state = 999

    n_train = 40000 # Number of training set samples

    train_dnn, train_dnr_nr = train_ds[:n_train, :], train_ds[n_train:, :]

    # Normalize the features in `[0, 1]`
    train_dnn.X /= 255
    train_dnr_nr.X /= 255
    test_ds.X /= 255
    print("Input data shape:", train_dnn.X.shape)
    print("Target labels shape:", train_dnn.Y.shape)
    print("Input data shape:", train_dnr_nr.X.shape)
    print("Target labels shape:", train_dnr_nr.Y.shape)
    print("Input data shape:", test_ds.X.shape)
    print("Target labels shape:", test_ds.Y.shape)

    ######################### Train and load Surrogate Model ######################

    net = SurrogateCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    metric = CMetricAccuracy()

    surrogate_model = CClassifierPyTorch(model=net,
                            loss=criterion,
                            optimizer=optimizer,
                            epochs=50,
                            batch_size=100,
                            input_shape=(3, 32, 32),
                            preprocess=None,
                            softmax_outputs=False)

    surrogate_model.fit(train_dnn.X, train_dnn.Y)
    torch.save(surrogate_model, "model_cifar10/cifar10_surr_model.pt")

    ########################## Train and Load DNN Model ###########################

    net = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    metric = CMetricAccuracy()

    cifar10_model = CClassifierPyTorch(model=net,
                            loss=criterion,
                            optimizer=optimizer,
                            epochs=75,
                            batch_size=100,
                            input_shape=(3, 32, 32),
                            preprocess=None,
                            softmax_outputs=False)

    #cifar10_model.fit(train_dnn.X, train_dnn.Y)
    dnn_model = torch.load("model_cifar10/cifar10_model.pt")

    ####################### Train DNR Model #######################################

    layers = ['features:dropout3', 'features:relu7', 'linear']
    path = "model_cifar10/dnr_cifar10.pkl"
    params = {'features:dropout3.C': 10, 'linear.kernel.gamma': 1e-2}
    rej=False
    dnr_training = dnr_performance(train_dnn, test_ds, dnn_model, rej, 
                                layers, params, path, num_runs=5)

    ################################# Training NR model ###########################

    layers = ['linear']
    path = "model_cifar10/'nr_cifar10.pkl"
    rej = False
    nr_training = nr_performance(train_dnn, test_ds, dnn_model, rej, 
                                layers, path, num_runs=5)

    ################################ FADER ########################################

    ################################## Training RBF Network #######################

    model = CIFAR10CNNRBF(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    cifar10rbf_model = CClassifierPyTorch(model=model,
                            loss=criterion,
                            optimizer=optimizer,
                            epochs=250,
                            batch_size=100,
                            input_shape=(3, 32, 32),
                            preprocess=None,
                            softmax_outputs=False)


    rbf_cifar10_model = torch.load("model_cifar10/cifar10rbf_model.pt")

    ################################ Training DNR-RBF Model #######################

    layers = ['features:dropout3', 'features:relu7', 'linear']
    path = "model_cifar10/dnr_rbf_cifar10.pkl"
    params = {'features:dropout3.C': 10, 'linear.kernel.gamma': 1e-2}
    rej=False
    dnr_training = dnr_performance(train_dnn, test_ds, rbf_cifar10_model, rej, 
                                layers, params, path, num_runs=5)

    ############################## Loading NR-RBF Model ###########################

    layers = ['linear']
    path = "model_cifar10/nr_rbf_cifar10.pkl"
    rej = False
    nr_training = nr_performance(train_dnn, test_ds, rbf_cifar10_model, rej, 
                                layers, path, num_runs=5)

if __name__ == "__main__":
    main()

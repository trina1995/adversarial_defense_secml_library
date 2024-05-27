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
from mnist_functions import white_box_attack, black_box_attack
import matplotlib.pyplot as plt

secml.settings.SECML_PYTORCH_USE_CUDA=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load the data
    train_ds, test_ds = CDataLoaderCIFAR10().load()

    # Data Preparation
    random_state = 999
    metric = CMetricAccuracy()

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

    # Load the DNN model
    dnn_model = torch.load("model_cifar10/cifar10_model.pt")
    
    # Evaluate the DNN model
    pred = dnn_model.predict(test_ds.X)
    acc = metric.performance_score(test_ds.Y, pred)
    print("The Accuracy based on DNN model: ", acc)
    
    # Attacking undefended model
    # Define a list of dmax values
    dmax_values = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    sample = 20
    
    # white-box attack 
    for dmax in dmax_values:
        path = "dnn_cifar/white_box_attack_on_dnn_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(dnn_model, test_ds, dmax, sample, metric, path)
    
    # Load DNR model
    layers = ['features:dropout3', 'features:relu7', 'linear']
    params = {'features:dropout3.C': 10, 'linear.kernel.gamma': 1e-2}
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1.0), C=1e-4)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-3), C=1.0)
    dnr = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=dnn_model, 
                        layers=layers, threshold=-10000)

    dnr.set_params(params)
    dnr_model = dnr.load("model_cifar10/dnr_cifar10.pkl.gz")
    
    dnr_rej_model = dnr.load("model_cifar10/dnr_cifar10_rej.pkl.gz")
    
    # Attacking the DNR model
    dmax = 0.0

    # White-box attack with rejection
    path = "dnr_cifar/white_box_attack_on_dnr_wor_{}.csv".format(dmax)
    dnn_attack_acc = white_box_attack(dnr_model, test_ds, dmax, sample, metric, path)
    
    # white-box attack
    dmax_values_white = [0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    for dmax in dmax_values_white:
        white_path = "dnr_cifar/white_box_attack_on_dnr_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(dnr_model, test_ds, dmax, sample, metric, white_path)
    
    # Black-box attack
    dmax_values_black = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    for dmax in dmax_values_black:
        black_path = "dnr_cifar/black_box_attack_on_dnr_{}.csv".format(dmax)
        dnn_attack_acc = black_box_attack(dnr_model, train_dnn, test_ds, 
                                  dmax, sample, black_path)
    
    # Load the NR model
    layers = ['linear']
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1.0), C=1e-4)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-3), C=1.0)
    nr = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=dnn_model, 
                        layers=layers, threshold=-10000)

    nr_model = nr.load("model_cifar10/nr_cifar10.pkl.gz")
    
    nr_rej_model = nr.load("model_cifar10/nr_cifar10_rej.pkl.gz")
    
    # Attacking the NR model
    dmax = 0.0

    # White-box attack with rejection
    path = "nr_cifar/white_box_attack_on_nr_wor_{}.csv".format(dmax)
    dnn_attack_acc = white_box_attack(nr_model, test_ds, dmax, sample, metric, path)
    
    # white-box attack
    dmax_values_white = [0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    for dmax in dmax_values_white:
        white_path = "nr_cifar/white_box_attack_on_nr_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(dnr_model, test_ds, dmax, sample, metric, white_path)
    
    # Black-box attack
    dmax_values_black = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    for dmax in dmax_values_black:
        black_path = "nr_cifar/black_box_attack_on_nr_{}.csv".format(dmax)
        dnn_attack_acc = black_box_attack(nr_model, train_dnn, test_ds, 
                                  dmax, sample, black_path)
    
    # Load the RBF model
    rbf_cifar10_model = torch.load("model_cifar10/cifar10rbf_model.pt")

    # Load the DNR-RBF model
    layers = ['features:dropout3', 'features:relu7', 'linear']
    params = {'features:dropout3.C': 10, 'linear.kernel.gamma': 1e-2}
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1.0), C=1e-4)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-3), C=1.0)
    dnr_rbf = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=rbf_cifar10_model, 
                        layers=layers, threshold=-10000)

    dnr_rbf.set_params(params)
    dnr_rbf_model = dnr_rbf.load("model_cifar10/dnr_rbf_cifar10.pkl.gz")
    
    dnr_rbf_rej_model = dnr_rbf.load("model_cifar10/dnr_rbf_cifar10_rej.pkl.gz")
    
    # Attacking the DNR-RBF model
    dmax = 0.0

    # White-box attack with rejection
    path = "dnr_rbf_cifar/white_box_attack_on_dnr_rbf_wor_{}.csv".format(dmax)
    dnn_attack_acc = white_box_attack(dnr_rbf_model, test_ds, dmax, sample, metric, path)
    
    # white-box attack
    dmax_values_white = [0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    for dmax in dmax_values_white:
        white_path = "dnr_rbf_cifar/white_box_attack_on_dnr_rbf_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(dnr_rbf_model, test_ds, dmax, sample, metric, white_path)
    
    # Black-box attack
    dmax_values_black = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    for dmax in dmax_values_black:
        black_path = "dnr_rbf_cifar/black_box_attack_on_dnr_rbf_{}.csv".format(dmax)
        dnn_attack_acc = black_box_attack(dnr_rbf_model, train_dnn, test_ds, 
                                  dmax, sample, black_path)
    
    # Load the NR-RBF model
    layers = ['linear']
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1.0), C=1e-4)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-3), C=1.0)
    nr_rbf = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=rbf_cifar10_model, 
                        layers=layers, threshold=-10000)

    nr_rbf_model = nr_rbf.load("model_cifar10/nr_rbf_cifar10.pkl.gz")
    
    nr_rbf_rej_model = nr_rbf.load("model_cifar10/nr_rbf_cifar10_rej.pkl.gz")
    
    # Attacking the NR-RBF model
    dmax = 0.0

    # White-box attack with rejection
    path = "nr_rbf_cifar/white_box_attack_on_nr_rbf_wor_{}.csv".format(dmax)
    dnn_attack_acc = white_box_attack(nr_rbf_model, test_ds, dmax, sample, metric, path)
    
    # white-box attack
    dmax_values_white = [0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    for dmax in dmax_values_white:
        white_path = "nr_rbf_cifar/white_box_attack_on_nr_rbf_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(nr_rbf_model, test_ds, dmax, sample, metric, white_path)
    
    # Black-box attack
    dmax_values_black = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    for dmax in dmax_values_black:
        black_path = "nr_rbf_cifar/black_box_attack_on_nr_rbf_{}.csv".format(dmax)
        dnn_attack_acc = black_box_attack(nr_rbf_model, train_dnn, test_ds, 
                                  dmax, sample, black_path)

if __name__ == "__main__":
    main()

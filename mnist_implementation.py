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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def main():
    loader = CDataLoaderMNIST()

    # Data Preparation
    random_state = 999

    n_train = 30000 # Number of training set samples
    n_test = 1000  # Number of test set samples
    metric = CMetricAccuracy()
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

    # Loading the DNN model
    loaded_pytorch_model = torch.load("model_mnist/mnist.pt")
    
    # Evaluate the DNN model
    dnn_predict = loaded_pytorch_model.predict(dnn_test.X)
    orig_acc = metric.performance_score(y_true=dnn_test.Y, y_pred=dnn_predict)
    print("Accuracy on test set for DNN model: {:.2%}".format(orig_acc))
    
    # Load the surrogate model
    surr_model = torch.load("model_mnist/surrogate_model.pt")
    
    # Attacking undefended model
    # Define a list of dmax values
    dmax_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    sample = 1000
    
    # Iterate over each dmax value
    for dmax in dmax_values:
        # Perform white-box attack
        path = "dnn_mnist/white_box_attack_on_dnn_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(loaded_pytorch_model, dnn_test, dmax, sample, metric, path)
    
    # Load the DNR model
    layers = ['features:relu2', 'features:relu3', 'features:relu4']
    params = {'features:relu4.C': 1, 'features:relu2.kernel.gamma': 1e-3}
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=0.1)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-2), C=10)
    dnr = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=loaded_pytorch_model, 
                         layers=layers, threshold=-1000)

    dnr.set_params(params)
    dnr_model = dnr.load("model_mnist\\dnr_mnist.pkl.gz")
    
    dnr_rej_model = dnr.load("model_mnist\\dnr_mnist_rej.pkl.gz")
    
    # Attacking the DNR model
    dmax = 0.0

    # White-box attack with rejection
    path = "dnr_mnist/white_box_attack_on_dnr_wor_{}.csv".format(dmax)
    dnn_attack_acc = white_box_attack(dnr_model, dnn_test, dmax, sample, metric, path)
    
    # white-box attack
    dmax_values_white = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    for dmax in dmax_values_white:
        white_path = "dnr_mnist/white_box_attack_on_dnr_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(dnr_model, dnn_test, dmax, sample, metric, white_path)
    
    # Black-box attack
    dmax_values_black = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    for dmax in dmax_values_black:
        black_path = "dnr_mnist/black_box_attack_on_dnr_{}.csv".format(dmax)
        dnn_attack_acc = black_box_attack(surr_model, dnr_model, dnr_nr_train, dnn_test, 
                                        dmax, sample, black_path)
    
    # Load the NR model
    layers = ['features:relu4']
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=0.1)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-2), C=10)
    nr = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=loaded_pytorch_model, 
                         layers=layers, threshold=-1000)
    nr_model = nr.load("model_mnist\\nr_mnist.pkl.gz")
    
    nr_rej_model = nr.load("model_mnist\\nr_mnist_rej.pkl.gz")
    
    # Attacking the NR model
    # White-box attack with rejection
    path = "nr_mnist/white_box_attack_on_nr_wor_{}.csv".format(0.0)
    dnn_attack_acc = white_box_attack(nr_model, dnn_test, 0.0, sample, metric, path)
    
    # white-box attack
    for dmax in dmax_values_white:
        white_path = "nr_mnist/white_box_attack_on_nr_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(nr_model, dnn_test, dmax, sample, metric, white_path)
    
    # Black-box attack
    for dmax in dmax_values_black:
        black_path = "nr_mnist/black_box_attack_on_nr_{}.csv".format(dmax)
        dnn_attack_acc = black_box_attack(surr_model, nr_model, dnr_nr_train, dnn_test, 
                                        dmax, sample, black_path)
    
    # Load the RBF model
    mnist_rbf_model = torch.load("model_mnist/mnist_rbf_model.pt")
    pred = mnist_rbf_model.predict(dnn_test.X)
    acc = metric.performance_score(dnn_test.Y, pred)
    print("Accuracy on test set for RBF model: {:.2%}".format(acc))

    # Load the DNR-RBF model
    layers = ['features:relu2', 'features:relu3', 'features:relu4']
    params = {'features:relu4.C': 1, 'features:relu2.kernel.gamma': 1e-3}
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=0.1)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-2), C=10)
    dnr_rbf = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=mnist_rbf_model, 
                        layers=layers, threshold=-1000)

    dnr_rbf.set_params(params)
    dnr_rbf_model = dnr_rbf.load("model_mnist\\dnr_rbf_mnist.pkl.gz")
    
    dnr_rbf_rej_model = dnr_rbf.load("model_mnist\\dnr_rbf_mnist_rej.pkl.gz")
    
    # Attacking the DNR-RBF model
    # White-box attack with rejection
    path = "dnr_rbf_mnist/white_box_attack_on_dnr_rbf_wor_{}.csv".format(0.0)
    dnn_attack_acc = white_box_attack(dnr_rbf_model, dnn_test, 0.0, sample, metric, path)
    
    # white-box attack
    for dmax in dmax_values_white:
        white_path = "dnr_rbf_mnist/white_box_attack_on_dnr_rbf_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(dnr_rbf_model, dnn_test, dmax, sample, metric, white_path)
    
    # Black-box attack
    for dmax in dmax_values_black:
        black_path = "dnr_rbf_mnist/black_box_attack_on_dnr_rbf_{}.csv".format(dmax)
        dnn_attack_acc = black_box_attack(surr_model, dnr_rbf_model, dnr_nr_train, dnn_test, 
                                        dmax, sample, black_path)
    
    # Load the NR-RBF model
    layers = ['features:relu4']
    combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=0.1)
    layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-2), C=10)
    nr_rbf = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=mnist_rbf_model, 
                         layers=layers, threshold=-1000)

    nr_rbf_model = nr_rbf.load("model_mnist\\nr_rbf_mnist.pkl.gz")
    
    nr_rbf_rej_model = nr_rbf.load("model_mnist\\nr_rbf_mnist_rej.pkl.gz")
    
    # Attacking the NR-RBF model
    # White-box attack with rejection
    path = "nr_rbf_mnist/white_box_attack_on_nr_rbf_wor_{}.csv".format(0.0)
    dnn_attack_acc = white_box_attack(nr_rbf_model, dnn_test, 0.0, sample, metric, path)
    
    # white-box attack
    for dmax in dmax_values_white:
        white_path = "nr_rbf_mnist/white_box_attack_on_nr_rbf_{}.csv".format(dmax)
        dnn_attack_acc = white_box_attack(nr_rbf_model, dnn_test, dmax, sample, metric, white_path)
    
    # Black-box attack
    for dmax in dmax_values_black:
        black_path = "nr_rbf_mnist/black_box_attack_on_nr_rbf_{}.csv".format(dmax)
        dnn_attack_acc = black_box_attack(surr_model, nr_rbf_model, dnr_nr_train, dnn_test, 
                                        dmax, sample, black_path)
    

if __name__ == "__main__":
    main()

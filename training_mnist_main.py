import torch
from torch import nn
import secml
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers.reject import CClassifierDNR
from secml.array import CArray
from collections import OrderedDict
from secml.data.selection import CPSRandom
from secml.data.splitter import CDataSplitterShuffle
import pandas as pd
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
secml.settings.SECML_PYTORCH_USE_CUDA = True 
############################ DNN Architecture #################################

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x

class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10, init_strategy='default'):

        nb_filters = 64

        super(MNISTCNN, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, nb_filters, kernel_size=5)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=nb_filters,
                                out_channels=nb_filters,
                                kernel_size=3)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=nb_filters,
                                out_channels=nb_filters,
                                kernel_size=3)),
            ('relu3', nn.ReLU(inplace=True)),
            ('flatten', Flatten()),
            ('fc1', nn.Linear(64*20*20, out_features=32)),
            ('relu4', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(p=.5, inplace=False)),
            #('flatten2', Flatten()),
        ]))

        self.classifier = nn.Linear(32, num_classes)

        if init_strategy == "default":
            pass
        else:
            raise ValueError("Unknown initialization strategy!")

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
  
############################ Surrogate Model ##################################

class MNISTPyTorch(nn.Module):
    def __init__(self):
        super(MNISTPyTorch, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  
        x = self.fc1(x)
        x = self.relu3(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
############################## RBF Network ####################################

class RBFNetwork(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super(RBFNetwork, self).__init__()
        self.hidden_features = hidden_features  # Define the hidden_features attribute properly
        self.centers = nn.Parameter(torch.Tensor(hidden_features, input_features))
        self.sigma = nn.Parameter(torch.Tensor(1))  # Uniform sigma for all RBF neurons
        self.linear = nn.Linear(hidden_features, output_features)  # Mapping the RBF outputs to class scores
        self.init_params()  # Call to initialize parameters

    def init_params(self):
        nn.init.uniform_(self.centers, -1, 1)
        nn.init.constant_(self.sigma, 1)  # Adjust initial guess based on feature scaling

    def forward(self, inputs):
        size = (inputs.size(0), self.hidden_features, inputs.size(1))
        inputs = inputs.unsqueeze(1).expand(size)
        centers = self.centers.unsqueeze(0).expand(size)
        distances = torch.sum((inputs - centers)**2, -1)
        rbf_outputs = torch.exp(-distances / (2 * self.sigma**2))
        outputs = self.linear(rbf_outputs)
        return outputs

class MNISTCNNRBF(nn.Module):
    def __init__(self, num_classes=10, init_strategy='default'):

        nb_filters = 64

        super(MNISTCNNRBF, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, nb_filters, kernel_size=5, stride=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=nb_filters,
                                out_channels=nb_filters,
                                kernel_size=3, stride=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=nb_filters,
                                out_channels=nb_filters,
                                kernel_size=3, stride=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('flatten', Flatten()),
            ('fc1', nn.Linear(576, out_features=32)),
            ('relu4', nn.ReLU(inplace=True)),
            ('rbf_net', RBFNetwork(input_features=32, hidden_features=784, output_features=10)),
            ('dropout', nn.Dropout(p=.5, inplace=False)),
            ('flatten2', Flatten()),
        ]))

        self.classifier = nn.Linear(10, num_classes)

        if init_strategy == "default":
            pass
        else:
            raise ValueError("Unknown initialization strategy!")

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

########################### NR implementation ################################

def nr_performance(train_data, test_data, model, rej, layers, path, num_runs=5):

    test_accuracy_list = []
    metric = CMetricAccuracy()

    for run in range(num_runs):
        # Shuffle and split the training data
        splitter_train = CDataSplitterShuffle(num_folds=1, train_size=0.8, test_size=0.2, random_state=1)
        splitter_train.compute_indices(train_data)

        # Shuffle and select a subset of the test data
        splitter_test = CDataSplitterShuffle(num_folds=1, test_size=1000, random_state=run)
        splitter_test.compute_indices(test_data)
        
        # Select random samples from the shuffled training and test sets
        tr_set = CPSRandom().select(dataset=train_data[splitter_train.tr_idx[0], :], n_prototypes=10000, random_state=run)
        ts_set = CPSRandom().select(dataset=test_data[splitter_test.tr_idx[0], :], n_prototypes=1000, random_state=run)
        
        # Initialize the NR model with SVM combiner and layer classifier
        layers = layers
        combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=0.1)
        layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-2), C=10)
        nr = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=model, 
                             layers=layers, threshold=-1000)

        # Train the NR model on the selected training data
        nr.fit(x=tr_set.X, y=tr_set.Y)
        
        if rej==True:
            nr.threshold = nr.compute_threshold(rej_percent=0.1, ds=ts_set)
        else:
            pass
        # save the NR model
        nr.save(path)
        
        # Evaluate the model on the selected test data
        acc = metric.performance_score(y_true=ts_set.Y, y_pred=nr.predict(ts_set.X))
        
        # Append the accuracy of the current run to the test accuracy list
        test_accuracy_list.append(acc)

    return test_accuracy_list

########################### DNR implementation ################################

def dnr_performance(train_data, test_data, model, rej, layers, params, path, num_runs=5):

    test_accuracy_list = []
    metric = CMetricAccuracy()

    for run in range(num_runs):
        # Shuffle and split the training data
        splitter_train = CDataSplitterShuffle(num_folds=1, train_size=0.8, test_size=0.2, random_state=1)
        splitter_train.compute_indices(train_data)

        # Shuffle and select a subset of the test data
        splitter_test = CDataSplitterShuffle(num_folds=1, test_size=1000, random_state=run)
        splitter_test.compute_indices(test_data)
        
        # Select random samples from the shuffled training and test sets
        tr_set = CPSRandom().select(dataset=train_data[splitter_train.tr_idx[0], :], n_prototypes=10000, random_state=run)
        ts_set = CPSRandom().select(dataset=test_data[splitter_test.tr_idx[0], :], n_prototypes=1000, random_state=run)
        
        # Initialize DNR with SVM combiner and layer classifier 
        layers = layers
        combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1), C=0.1)
        layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-2), C=10)
        dnr = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=model, 
                             layers=layers, threshold=-1000)

        dnr.set_params(params)

        # Train the DNR model on the selected training data
        dnr.fit(x=tr_set.X, y=tr_set.Y)
        
        if rej==True:
            dnr.threshold = dnr.compute_threshold(rej_percent=0.1, ds=ts_set)
        else:
            pass
        
        # save the DNR model
        dnr.save(path)
        
        # Evaluate the model on the selected test data
        acc = metric.performance_score(y_true=ts_set.Y, y_pred=dnr.predict(ts_set.X))
        
        # Append the accuracy of the current run to the test accuracy list
        test_accuracy_list.append(acc)

    return test_accuracy_list

########################### White-box Attack Function #########################

def white_box_attack(classifier, test_set, dmax, sample, metric, path):
    # For simplicity, let's attack a subset of the test set
    rand_test_set = test_set[:sample, :]
    
    # Define PGD attack parameters
    noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
    lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
    y_target = None  # None if `error-generic` or a class label for `error-specific`
    solver_params = {'eta': 1e-1, 'eta_min': 1e-1, 'max_iter': 30, 'eps': 1e-8}

    # Create PGD attack object
    pgd_ls_attack = CAttackEvasionPGDLS(classifier=classifier,
                                        double_init=False,
                                        distance='l2',
                                        dmax=dmax,
                                        lb=lb, ub=ub,
                                        solver_params=solver_params,
                                        y_target=y_target)
    
    print("Attack started...")
    # Execute the attack on the test set
    eva_y_pred, _, _, _ = pgd_ls_attack.run(rand_test_set.X, rand_test_set.Y)
    print("Attack complete!")

    # Evaluate the performance of the attacked DNR model on the test set
    acc_attack = metric.performance_score(y_true=rand_test_set.Y, y_pred=eva_y_pred)
    print("Accuracy on test set after attack: {:.2%}".format(acc_attack))
    
    # Save accuracy result to csv
    accuracy_to_save = pd.DataFrame({'White-box success': [acc_attack]})
    accuracy_to_save.to_csv(path, index=False)
    
    return acc_attack

######################### Black-box Attack Function ###########################

def black_box_attack(surrogate_model, target_model, tr_set, ts_set, dmax, 
                     sample_size, path):
    distance='l2'
    lb=0
    ub=1
    attack_params={'eta': 0.1, 'max_iter': 100, 'eps': 1e-4}
    
    # Configure the PGD attack for the surrogate model
    pgd_attack_surrogate = CAttackEvasionPGDLS(
        classifier=surrogate_model,
        double_init_ds=tr_set,
        distance=distance,
        dmax=dmax,
        lb=lb,
        ub=ub,
        solver_params=attack_params
    )
    
    # To attack samples from the test set
    x_test, y_test = ts_set[:sample_size, :].X, ts_set[:sample_size, :].Y
    
    # Run the attack to generate adversarial examples
    y_pred, _, x_adv, _ = pgd_attack_surrogate.run(x_test, y_test)
    
    # Predict with the target model on the generated adversarial examples
    y_pred_target = target_model.predict(x_adv.X)
    
    # Calculate accuracy of the target model on adversarial examples
    accuracy_target = CMetricAccuracy().performance_score(y_true=y_test, y_pred=y_pred_target)
    print(f"Accuracy on adversarial examples against the target model: {accuracy_target:.2%}")
    
    # Save accuracy result to CSV
    accuracy_to_save = pd.DataFrame({'Black-box success': [accuracy_target]})
    accuracy_to_save.to_csv(path, index=False)
    
    return accuracy_target

import secml
import torch
from torch import nn
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.ml.classifiers.reject import CClassifierDNR
from collections import OrderedDict
from secml.data.selection import CPSRandom
from secml.data.splitter import CDataSplitterShuffle
import pandas as pd

secml.settings.SECML_PYTORCH_USE_CUDA=True

############################## DNN Architecture ###############################

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        nb_filters = 64
        super(CIFAR10CNN, self).__init__()

        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, nb_filters, kernel_size=3, padding=1)),
                ('bn1', nn.BatchNorm2d(nb_filters)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=1)),
                ('bn2', nn.BatchNorm2d(nb_filters)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(2, stride=2)),
                ('dropout1', nn.Dropout(0.1)),
                ('conv3', nn.Conv2d(nb_filters, 128, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(128)),
                ('relu3', nn.ReLU(inplace=True)),
                ('conv4', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(128)),
                ('relu4', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(2, stride=2)),
                ('dropout2', nn.Dropout(0.2)),
                ('conv5', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('conv6', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn6', nn.BatchNorm2d(256)),
                ('relu6', nn.ReLU(inplace=True)),
                ('maxpool3', nn.MaxPool2d(2, stride=2)),
                ('dropout3', nn.Dropout(0.3)),
                ('conv7', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
                ('bn7', nn.BatchNorm2d(512)),
                ('relu7', nn.ReLU(inplace=True)),
                ('maxpool4', nn.MaxPool2d(2, stride=2)),
                ('dropout4', nn.Dropout(0.4))
            ])
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512*2*2, 512)
        self.classifier = nn.Linear(512, num_classes)  

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.classifier(x)
        return x

############################### Surrogate Model ###############################

class SurrogateCNN(nn.Module):
    def __init__(self):
        super(SurrogateCNN, self).__init__()

        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
                ('pool1', nn.MaxPool2d(2, stride=2)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
                ('pool2', nn.MaxPool2d(2, stride=2)),
                ('relu2', nn.ReLU(inplace=True))
            ])
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(128*8*8, 256), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

############################## RBF Network ####################################

class RBF_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBF_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        self.beta = nn.Parameter(torch.Tensor(1, out_features))
        nn.init.uniform_(self.centers, -1, 1)
        nn.init.uniform_(self.beta, 0, 1)

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = torch.norm(x - c, 2, 2)
        return torch.exp(-self.beta * distances * distances)
    
class CIFAR10CNNRBF(nn.Module):
    def __init__(self, num_classes=10):
        nb_filters = 64
        super(CIFAR10CNNRBF, self).__init__()

        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, nb_filters, kernel_size=3, padding=1)),
                ('bn1', nn.BatchNorm2d(nb_filters)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=1)),
                ('bn2', nn.BatchNorm2d(nb_filters)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(2, stride=2)),
                ('dropout1', nn.Dropout(0.1)),
                ('conv3', nn.Conv2d(nb_filters, 128, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(128)),
                ('relu3', nn.ReLU(inplace=True)),
                ('conv4', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(128)),
                ('relu4', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(2, stride=2)),
                ('dropout2', nn.Dropout(0.2)),
                ('conv5', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('conv6', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn6', nn.BatchNorm2d(256)),
                ('relu6', nn.ReLU(inplace=True)),
                ('maxpool3', nn.MaxPool2d(2, stride=2)),
                ('dropout3', nn.Dropout(0.3)),
                ('conv7', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
                ('bn7', nn.BatchNorm2d(512)),
                ('relu7', nn.ReLU(inplace=True)),
                ('maxpool4', nn.MaxPool2d(2, stride=2)),
                ('dropout4', nn.Dropout(0.4)),
            ])
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512*2*2, 512)
        self.rbf_detector = RBF_Layer(512, 3072)
        self.classifier = nn.Linear(3072, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.rbf_detector(x)
        x = self.classifier(x)
        return x

############################## NR Implementation ##############################

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
        combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1.0), C=1e-4)
        layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-2), C=1.0)
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

############################ DNR Implementation ###############################

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
        combiner = CClassifierSVM(kernel=CKernelRBF(gamma=1.0), C=1e-4)
        layer_clf = CClassifierSVM(kernel=CKernelRBF(gamma=1e-3), C=1.0)
        dnr = CClassifierDNR(combiner=combiner, layer_clf=layer_clf, dnn=model, 
                             layers=layers, threshold=-10000)

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

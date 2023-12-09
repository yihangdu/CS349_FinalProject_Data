import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import sklearn.metrics as SKM


device = "cuda" if torch.cuda.is_available() else "cpu"

def read_data(file_name):
    
    data_set = []
    header = True
    with open(file_name,'rt') as f:
        for line in f:
            if header:
                header = False
            else:
                line = line.replace('\n','')
                tokens = line.split(',')
                label = tokens[0]
                attribs = []
                for i in range(len(tokens)-1):
                    attribs.append(float(tokens[i+1]))
                data_set.append([label,attribs])
    return(data_set)
        
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
                   
def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)

def _train(dataloader, model, loss_func, optimizer, lamb):
    model.train()
    train_loss = []

    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        iters = 10 * len(X)
        then = datetime.datetime.now()
        iters /= (then - now).total_seconds() + 0.0000001
        print(f"loss: {loss:>f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
        now = then
        train_loss.append(loss)
    return train_loss

def _validate(dataloader, model, loss_func):
    num_batches = len(dataloader)
    model.eval()
    pred_list = []
    actual_list = []
    validation_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            actual_list += [int(i) for i in y]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predLabels = torch.max(pred.data, 1)
            pred_list += [int(i) for i in predLabels]
            validation_loss += loss_func(pred, y).item()
    validation_loss /= num_batches
    print(f"Avg Validation Loss: {validation_loss:>8f}\n")
    return validation_loss, pred_list, actual_list

def _test(dataloader, model, loss_func):
    num_batches = 0
    model.eval()
    pred_list = []
    actual_list = []
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            actual_list += [int(i) for i in y]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predLabels = torch.max(pred.data, 1)
            pred_list += [int(i) for i in predLabels]
            test_loss += loss_func(pred, y).item()
            num_batches = num_batches + 1
    test_loss /= num_batches
    print(f"Test Avg Loss: {test_loss:>8f}\n")
    return test_loss, pred_list, actual_list


def classify_insurability():
    train = "Diagnostics_trainHealthy.csv"
    valid = "Diagnostics_validHealthy.csv"
    test = "Diagnostics_testHealthy.csv"
    train = read_data(train)
    valid = read_data(valid)
    test = read_data(test)
    
    #sc = MinMaxScaler()
    #train_features = [item[1] for item in train]
    #sc.fit(train_features)
    input_length = len(train[0][1])
    def prepare_data(data):
        features = [list(np.float32(item[1])) for item in data]
        labels = [int(item[0]) for item in data]

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features)
        labels_tensor = torch.tensor(labels)

        return features_tensor, labels_tensor

    def create_dataloader(data, batch_size=1, shuffle=True):
        features, labels = prepare_data(data)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    
    def softmax(x):
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
        return torch.log(exp_x / sum_exp_x)    
    
    class FeedForward(nn.Module):
        def __init__(self):
            super(FeedForward, self).__init__()
            self.linear1 = nn.Linear(input_length, 16, bias=True)
            self.relu1 = nn.LeakyReLU()
            self.linear2 = nn.Linear(16, 8)
            self.relu2 = nn.LeakyReLU()
            self.linear_out = nn.Linear(8, 2)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu1(x)
            x = self.linear2(x)
            x = self.relu2(x)
            x = self.linear_out(x)
            return x
      
    
    # ff = FeedForward()
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(ff.parameters(), lr=0.1)
    # a, b = prepare_data(train)
    # a = a[0].unsqueeze(0)
    # b = b[0]
    # with torch.no_grad():
    #     pred = ff(a)
    #     loss = loss_func(pred, b.unsqueeze(0))
    #     _, predicted_class = torch.max(pred, 1)
    # print('a:',a)
    # print('b:',b)
    # print('prediction:', pred)
    # print('predicted class:', predicted_class.item())
    # print('error:', loss.item())
    

    
    
    ff = FeedForward().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ff.parameters(), lr=0.01)
    epochs = 20
    train_loss = []
    validate_loss = []

    validPred_list = []
    validActual_list = []

    train_loader = create_dataloader(train, batch_size=1)
    valid_loader = create_dataloader(valid, batch_size=1)
    test_loader = create_dataloader(test, batch_size=1)
    #early stop parameters
    best_score = None
    patience = 3
    count = 0

    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------- \n")
        train_loss_epoch = _train(train_loader, ff, loss_func, optimizer, 0.01)
        train_loss.append(train_loss_epoch)
        valid_loss_epoch, tmpPred_list, tmpActual_list = _validate(valid_loader, ff, loss_func)
        #check if stop early
        if best_score is None:
            best_score = valid_loss_epoch
        elif valid_loss_epoch > best_score:
            count += 1
            if count >= patience:
                print('early stop when iter: ', t)
                break
        else:
            best_score = valid_loss_epoch
            count = 0
            
        validate_loss.append(valid_loss_epoch)
        validPred_list += tmpPred_list
        validActual_list += tmpActual_list
        #test_loss.append(_test(test_loader, ff, loss_func))
        
    print('Validation Done!\n')
    print("Acc Validation", SKM.accuracy_score(validActual_list, validPred_list))

    test_loss, testPred_list, testActual_list = _test(test_loader, ff, loss_func)
    print("Acc Test", SKM.accuracy_score(testActual_list, testPred_list))
    
    
    plt.plot([i for i in range(len(train_loss))], torch.tensor(train_loss).mean(axis=1))
    plt.plot([i for i in range(len(validate_loss))], validate_loss)
    plt.show()
    
    #plt.plot([i for i in range(len(test_loss))], test_loss)
    


def pca_transform(train, valid, test, n_components=.95):
    train_labels = [item[0] for item in train]
    train_data = [item[1] for item in train]

    valid_labels = [item[0] for item in valid]
    valid_data = [item[1] for item in valid]

    test_labels = [item[0] for item in test]
    test_data = [item[1] for item in test]

    pca = PCA(n_components)
    pca.fit(train_data)
    p_components_train = pca.transform(train_data)
    p_components_valid = pca.transform(valid_data)
    p_components_test = pca.transform(test_data)

    train = [[train_labels[i], p_components_train[i]] for i in range(len(train_labels))]
    valid = [[valid_labels[i], p_components_valid[i]] for i in range(len(valid_labels))]
    test = [[test_labels[i], p_components_test[i]] for i in range(len(test_labels))]
    return train, valid, test

def _train_reg(dataloader, model, loss_func, optimizer, lamb):
    model.train()
    train_loss = []

    now = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)

        # l2-regularization
        weights = [model.linear1.weight, model.linear2.weight, model.linear_out.weight]
        R = 0
        for w in weights:
            R += torch.sum(torch.mul(w, w))

        loss = loss_func(pred, y) + lamb * R/2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            iters = 10 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            print(f"loss: {loss:>f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)
    return train_loss


def classify_mnist():
    train = "Diagnostics_trainHealthy.csv"
    valid = "Diagnostics_validHealthy.csv"
    test = "Diagnostics_testHealthy.csv"
    train = read_data(train)
    valid = read_data(valid)
    test = read_data(test)
    # show_mnist('mnist_test.csv','pixels')

    # reduce dimensionality using PCA
    #train, valid, test = pca_transform(train, valid, test)
    input_length = len(train[0][1])

    def prepare_data(data):
        features = [list(np.float32(item[1])) for item in data]
        labels = [int(item[0]) for item in data]

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features)
        labels_tensor = torch.tensor(labels)

        return features_tensor, labels_tensor

    def create_dataloader(data, batch_size=1, shuffle=True):
        features, labels = prepare_data(data)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    # add a regularizer of your choice to classify_mnist()

    class FeedForward(nn.Module):
        def __init__(self):
            super(FeedForward, self).__init__()
            self.linear1 = nn.Linear(input_length, 8)
            self.TS1 = nn.PReLU()
            self.linear2 = nn.Linear(8, 4)
            self.TS2 = nn.PReLU()
            self.linear_out = nn.Linear(4, 2)

        def forward(self, x):
            x = self.linear1(x)
            x = self.TS1(x)
            x = self.linear2(x)
            x = self.TS2(x)
            x = self.linear_out(x)
            return x

    ff = FeedForward().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ff.parameters(), lr=0.001)
    epochs = 20
    train_loss = []
    valid_loss = []

    validPred_list = []
    validActual_list = []

    train_loader = create_dataloader(train)
    valid_loader = create_dataloader(valid)
    test_loader = create_dataloader(test)
    for t in range(epochs):
        print(f"WithReg Epoch {t + 1}\n------------------------- \n")
        train_loss.append(_train(train_loader, ff, loss_func, optimizer, 0.001))
        tmpValid_loss, tmpPred_list, tmpActual_list = _validate(valid_loader, ff, loss_func)
        valid_loss.append(tmpValid_loss)
        validPred_list += tmpPred_list
        validActual_list += tmpActual_list
    print('Validation Done!\n')
    print("Acc Validation", SKM.accuracy_score(validActual_list, validPred_list))

    test_loss, testPred_list, testActual_list = _test(test_loader, ff, loss_func)
    print("Acc Test", SKM.accuracy_score(testActual_list, testPred_list))

    evaluate(testActual_list, testPred_list, "nnMnist")

    plt.plot([i for i in range(len(train_loss))], torch.tensor(train_loss).mean(axis=1))
    plt.plot([i for i in range(len(valid_loss))], valid_loss)
    plt.show()



def classify_mnist_reg():
    train = "Diagnostics_train.csv"
    valid = "Diagnostics_valid.csv"
    test = "Diagnostics_test.csv"
    train = read_data(train)
    valid = read_data(valid)
    test = read_data(test)
    # show_mnist('mnist_test.csv','pixels')

    # reduce dimensionality using PCA
    #train, valid, test = pca_transform(train, valid, test)
    input_length = len(train[0][1])

    def prepare_data(data):
        features = [list(np.float32(item[1])) for item in data]
        labels = [int(item[0]) for item in data]

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features)
        labels_tensor = torch.tensor(labels)

        return features_tensor, labels_tensor

    def create_dataloader(data, batch_size=16, shuffle=True):
        features, labels = prepare_data(data)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    # add a regularizer of your choice to classify_mnist()

    class FeedForward(nn.Module):
        def __init__(self):
            super(FeedForward, self).__init__()
            self.linear1 = nn.Linear(input_length, 6, bias=False)
            self.TS1 = nn.PReLU()
            self.linear2 = nn.Linear(6, 4, bias=False)
            self.TS2 = nn.PReLU()
            self.linear_out = nn.Linear(4, 2, bias=False)

        def forward(self, x):
            x = self.linear1(x)
            x = self.TS1(x)
            x = self.linear2(x)
            x = self.TS2(x)
            x = self.linear_out(x)
            return x

    ff = FeedForward().to(device)
    loss_func = nn.MultiMarginLoss()
    optimizer = torch.optim.Adam(ff.parameters(), lr=0.001)
    epochs = 10

    train_loss = []
    valid_loss = []

    validPred_list = []
    validActual_list = []

    train_loader = create_dataloader(train)
    valid_loader = create_dataloader(valid)
    test_loader = create_dataloader(test)
    for t in range(epochs):
        print(f"WithReg Epoch {t + 1}\n------------------------- \n")
        train_loss.append(_train(train_loader, ff, loss_func, optimizer, 0.01))
        tmpValid_loss, tmpPred_list, tmpActual_list = _validate(valid_loader, ff, loss_func)
        valid_loss.append(tmpValid_loss)
        validPred_list += tmpPred_list
        validActual_list += tmpActual_list
    print('Validation Done!\n')
    print("Acc Validation", SKM.accuracy_score(validActual_list, validPred_list))

    test_loss, testPred_list, testActual_list = _test(test_loader, ff, loss_func)
    print("Acc Test", SKM.accuracy_score(testActual_list, testPred_list))

    #evaluate(testActual_list, testPred_list, "nnMnistWithReg")

    plt.plot([i for i in range(len(train_loss))], torch.tensor(train_loss).mean(axis=1))
    plt.plot([i for i in range(len(valid_loss))], valid_loss)
    plt.show()


def classify_insurability_manual():
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN


def evaluate(actual_label, predict_label, algorithm):

    '''
    Confusion matrix whose i-th row and j-th column entry indicates the number of samples
    with true label being i-th class and predicted label being j-th class.
    '''
    conMat = SKM.confusion_matrix(actual_label, predict_label)

    Recall = SKM.recall_score(actual_label, predict_label, average = None)
    RecallAvg = SKM.recall_score(actual_label, predict_label, average = "macro")

    Precision = SKM.precision_score(actual_label, predict_label, average = None, zero_division = 0.0)
    PrecisionAvg = SKM.precision_score(actual_label, predict_label, average = "macro")

    Accuracy = SKM.accuracy_score(actual_label, predict_label)
    
    with open(f"evaluate_{algorithm}.txt", "w") as f:
        labels = sorted([int(x) for x in set(actual_label)])
        f.write(f"Confusion Matrix for {algorithm}:\n")
        f.write(f"{conMat}\n")
        f.write("\n")
        f.write(f"Recall: {list(Recall)}\n")
        f.write(f"Recall Average: {RecallAvg}\n\n")
        f.write(f"Precision: {list(Precision)}\n")
        f.write(f"Precision Average: {PrecisionAvg}\n\n")
        f.write(f"Accuracy: {Accuracy}")

def main():
    #classify_insurability()
    classify_mnist()
    #classify_mnist_reg()
    # classify_insurability_manual()


if __name__ == "__main__":
    main()

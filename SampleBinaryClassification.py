from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import Linear, Sigmoid, Module, BCELoss, ReLU
from torch.optim import SGD
from torch.nn.init import kaiming_normal, xavier_normal
from torch import Tensor

# dataset definition
class CSVData(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the CSV file as a dataset
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:,:-1]
        self.y = df.values[:,-1]
        # ensure input data is float
        self.X = self.X.astype('float32')
        # label encode target and ensure that values are float
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of row in the dataset
    def __len__(self):
        return len(self.X)

    # get the row and index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_split(self, n_test=0.33):
        # determine size
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# Model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()

        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_normal(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_normal(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        # third hidden layer
        self.hidden3 = Linear(8, 1)
        xavier_normal(self.hidden3.weight)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first layer
        X = self.hidden1(X)
        X = self.act1(X)

        # first to second layer
        X = self.hidden2(X)
        X = self.act2(X)

        # second to third layer
        X = self.hidden3(X)
        X = self.act3(X)

        return X

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVData(path)
    # calculate split
    train, test = dataset.get_split()
    # prepare data loader
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # enumerate epochs
    for epochs in range(5000):
        # enumerate minibatches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradient
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # compute the losses
            loss = criterion(yhat, targets)
            # credit assignments
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class value
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions = vstack(predictions)
    actuals = vstack(actuals)
    # calculate accuracies
    acc = accuracy_score(predictions, actuals)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # model predict
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))

# define the network
model = MLP(34)

# train the model
train_model(train_dl, model)

# evaluate the model
acc = evaluate_model(test_dl, model)

print('Accuracy: %.3f' %acc)



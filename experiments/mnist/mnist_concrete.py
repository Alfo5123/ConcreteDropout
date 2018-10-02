from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
import sys

class ConcreteDropout(nn.Module):
    """This module allows to learn the dropout probability for any given input layer.
    ```python
        # as the first layer in a model
        model = nn.Sequential(ConcreteDropout(Linear_relu(1, nb_features),
        input_shape=(batch_size, 1), weight_regularizer=1e-6, dropout_regularizer=1e-5))
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = nn.Sequential(ConcreteDropout(Conv2D_relu(channels_in, channels_out),
        input_shape=(batch_size, 3, 128, 128), weight_regularizer=1e-6,
        dropout_regularizer=1e-5))
    ```
    # Arguments
        layer: a layer Module.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """
    def __init__(self, layer, input_shape, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, device = 0):
        super(ConcreteDropout, self).__init__()
        # Post drop out layer
        self.layer = layer
        # Input dim for regularisation scaling
        self.input_dim = np.prod(input_shape[1:])
        # Regularisation hyper-parameters
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform_(self.p_logit, a=init_min, b=init_max)
        #Device
        self.device = device

    def forward(self, x):
        return self.layer(self._concrete_dropout(x))

    def regularisation(self):
        """Computes weights and dropout regularisation for the layer, has to be
        extracted for each layer within the model and added to the total loss
        """
        weights_regularizer = self.weight_regularizer * self.sum_n_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer

    def _concrete_dropout(self, x):
        """Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 0.1
        self.p = nn.functional.sigmoid(self.p_logit)

        # Check if batch size is the same as unif_noise, if not take care
        unif_noise = Variable(torch.FloatTensor(np.random.uniform(size=tuple(x.size()))))#.to(self.device)

        drop_prob = (torch.log(self.p + eps)
                    - torch.log(1 - self.p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

    def sum_n_square(self):
        """Helper function for paramater regularisation
        """
        sum_of_square = 0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square
    
class Linear_relu(nn.Module):
    
    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())
        
    def forward(self, x):
        return self.model(x)

class Linear_softmax(nn.Module):
    
    def __init__(self, inp, out):
        super(Linear_softmax, self).__init__()
        self.f1 = nn.Linear(inp, out)

    def forward(self, x):
        x = self.f1( x )
        return F.softmax(x, dim=1)


class Net(nn.Module):

    def __init__(self,  wr, dr , batch_size ):
        super(Net, self).__init__()

        ## MLP 3x512
        self.fc1 = nn.Sequential(
                        ConcreteDropout(Linear_relu(784, 512), input_shape=(batch_size,784), 
                                        weight_regularizer=wr, dropout_regularizer=dr), #device = device),
                        ConcreteDropout(Linear_relu(512, 512), input_shape=(batch_size,512), 
                                        weight_regularizer=wr, dropout_regularizer=dr), #device = device),
                        ConcreteDropout(Linear_relu(512, 512), input_shape=(batch_size,512), 
                                        weight_regularizer=wr, dropout_regularizer=dr), #device = device),
                        ConcreteDropout(Linear_softmax(512, 10), input_shape=(batch_size,512), 
                                        weight_regularizer=wr, dropout_regularizer=dr), #device = device)
                    )

        self.fmean = ConcreteDropout(Linear_relu(10, D), input_shape=(batch_size,10),
                                            weight_regularizer=wr, dropout_regularizer=dr)
        self.flogvar = ConcreteDropout(Linear_relu(10, D), input_shape=(batch_size,10), 
                                              weight_regularizer=wr, dropout_regularizer=dr)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():

    # Experiment settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Donwload MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    ### Load model and start training iterations

    N = 60000 #MNIST train set size
    wr = 1e-2 / N
    dr = 2. / N

    model = Net(wr,dr,args.batch_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        print(  np.array([module.p.cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p')]) )


if __name__ == '__main__':
    main()
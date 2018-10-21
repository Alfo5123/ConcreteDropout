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
from model import ConcreteDropout, Linear_relu, Linear_softmax, Net
import operator
from sklearn.model_selection import train_test_split


def train(log_interval, model, device, train_loader, optimizer, epoch ):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) + model.regularisation_loss()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, validation ):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).sum().item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)

    if validation == 0 :

        test_loss /= total
        print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total,
        100. * correct / total)) 

    elif validation == 1 :

        test_loss /= total
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, total,
        100. * correct / total)) 

    else:
        test_loss /= total
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))


    return test_loss 


def make_dataloaders(train_set, test_set, train_size, valid_size, 
                    train_batch_size, valid_batch_size , test_batch_size):

    # Split training into train and validation
    #indices = torch.randperm(len(train_set))
    #train_indices = indices[:len(indices)-valid_size][:train_size or None]
    #valid_indices = indices[len(indices)-valid_size:] if valid_size else None

    # Stratified train / validation split
    labels = np.asarray(list(map(operator.itemgetter(1), train_set)))
    train_indices, valid_indices, _ , _ = train_test_split( np.arange(len(train_set)), labels, 
                                                            train_size = train_size , test_size=valid_size, 
                                                            stratify = labels, shuffle = True  )

    train_indices = torch.from_numpy(train_indices)
    valid_indices = torch.from_numpy(valid_indices)


    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=train_batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
    
    test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=test_batch_size)
    
    if valid_size:
        valid_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=valid_batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices))
    else:
        valid_loader = None

    return train_loader, valid_loader, test_loader 

def run_experiment ( train_batch_size = 128 , test_batch_size = 128, valid_batch_size = 128, 
                     lr = 0.01 , momentum = 0.5 , seed = 1, no_cuda = False , 
                     epochs = 20 , train_size = 50000, validation_size = 10000, test_size = 10000, 
                     log_interval = 100 ):

    # Set training conditions
    torch.manual_seed(seed)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Split dataset 
    train_set = datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    test_set = datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    train_loader , valid_loader , test_loader = make_dataloaders( train_set , test_set ,
                                                                  train_size, validation_size , 
                                                                  train_batch_size, valid_batch_size, 
                                                                  test_batch_size  )

    N = train_size #MNIST trainset size fraction
    wr = 1e-4 / N
    dr = 2. / N

    # Prepare model 
    model = Net(wr,dr,train_batch_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    training_curve = []
    validation_curve = []
    test_curve = [] 
    dropout_rates = []

    for epoch in range(1, epochs + 1):

        #Training
        train(log_interval, model, device, train_loader, optimizer, epoch)

        #Testing
        training_curve.append(test( model, device, train_loader, validation = 0 ))

        validation_curve.append(test(model, device, valid_loader, validation = 1  ))
        
        test_curve.append(test(model, device, test_loader, validation = 2 ) )

        dropout_rates.append( np.array([module.p.cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p')]) )
        
    #return  dropout_rates[-1], training_curve, validation_curve, test_curve 
    return  dropout_rates, training_curve, validation_curve, test_curve 

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

    # Generate training and validation set 
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
        train(args.log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, validation = 2, validation_size = -1) 
        print(  np.array([module.p.cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p')]) )


if __name__ == '__main__':
    main()



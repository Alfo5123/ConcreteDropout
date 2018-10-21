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

        #self.fmean = ConcreteDropout(Linear_relu(10, D), input_shape=(batch_size,10),
        #                                    weight_regularizer=wr, dropout_regularizer=dr)
        #self.flogvar = ConcreteDropout(Linear_relu(10, D), input_shape=(batch_size,10), 
        #                                      weight_regularizer=wr, dropout_regularizer=dr)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x

    def regularisation_loss(self):

        reg_loss = self.fc1[0].regularisation()+self.fc1[1].regularisation()+\
        		   self.fc1[2].regularisation()+self.fc1[3].regularisation()

        return reg_loss
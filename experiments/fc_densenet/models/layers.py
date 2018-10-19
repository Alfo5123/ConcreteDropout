import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class SpatialConcreteDropout(nn.Module):
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
    def __init__(self, layer, gpu_id, weight_regularizer=1e-6, dropout_regularizer=1e-5,
            init_min=0.1, init_max=0.1):
        super(SpatialConcreteDropout, self).__init__()
        # Post drop out layer
        self.layer = layer
        # Input dim for regularisation scaling
        #self.input_dim = np.prod(input_shape[1:])
        # Regularisation hyper-parameters
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)

        self.gpu_id = gpu_id

        self.use_dropout = True

    def forward(self, x):

        if self.use_dropout:
            return self.layer(self._concrete_dropout(x))

        return self.layer(x)

    def regularisation(self):
        """Computes weights and dropout regularisation for the layer, has to be
        extracted for each layer within the model and added to the total loss
        """

        if not self.use_dropout:
            return 0.0

        self.p = nn.functional.sigmoid(self.p_logit)
        
        weights_regularizer = self.weight_regularizer * self.sum_n_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer*self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer

    def _concrete_dropout(self, x):
        """Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 2.0/3.0
        self.p = nn.functional.sigmoid(self.p_logit)

        self.input_dim = float(x.size()[1])

        # Check if batch size is the same as unif_noise, if not take care
        #unif_noise = Variable(torch.FloatTensor(np.random.uniform(size=tuple(x.size()))),
        #        requires_grad=False).cuda(self.gpu_id)

        uniform_distribution = torch.distributions.uniform.Uniform(0, 1)

        unif_noise = uniform_distribution.sample(sample_shape=x.size()[:2]).cuda(self.gpu_id)

        drop_prob = (torch.log(self.p + eps)
                    - torch.log(1 - self.p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)

        #bernoulli_distribution = torch.distributions.bernoulli.Bernoulli(self.p[0])

        #drop_prob = bernoulli_distribution.sample(sample_shape=x.size()[:2])

        drop_prob = drop_prob.view(x.size()[:2] + (1, 1))

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        return x

    def sum_n_square(self):
        """Helper function for paramater regularisation
        """
        sum_of_square = 0.0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, dropout_data):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))

        conv_layer = nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                stride=1, padding=1, bias=True)

        if dropout_data['mode'] == 'dropout':

            self.add_module('drop', nn.Dropout2d(dropout_data['prob']))
            self.add_module('conv', conv_layer)

        elif dropout_data['mode'] == 'concrete_dropout':

            concrete_dropout_layer = SpatialConcreteDropout(conv_layer, dropout_data['gpu_id'],
                    weight_regularizer=dropout_data['weight_reg'],
                    dropout_regularizer=dropout_data['dropout_reg'],
                    init_min=dropout_data['init_min'], init_max=dropout_data['init_max'])

            self.add_module('concrete_drop', concrete_dropout_layer)

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, dropout_data,
            upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate, dropout_data)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels, dropout_data):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        conv_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                padding=0, bias=True)

        if dropout_data['mode'] == 'dropout':

            self.add_module('drop', nn.Dropout2d(dropout_data['prob']))
            self.add_module('conv', conv_layer)

        elif dropout_data['mode'] == 'concrete_dropout':

            concrete_dropout_layer = SpatialConcreteDropout(conv_layer, dropout_data['gpu_id'],
                    weight_regularizer=dropout_data['weight_reg'],
                    dropout_regularizer=dropout_data['dropout_reg'],
                    init_min=dropout_data['init_min'], init_max=dropout_data['init_max'])

            self.add_module('concrete_drop', concrete_dropout_layer)

        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers, dropout_data):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, dropout_data,
            upsample=True))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

import numpy as np
import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [sent len, batch size]

        embedded = self.dropout(self.embedding(x)) #embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))     

        return self.fc(hidden.squeeze(0))


class LSTMWithCDropout(nn.Module):

    def __init__(self, layer , input_size, output_size, gpu_id, weight_regularizer,
            dropout_regularizer, init_min, init_max):

        super().__init__()
        # Post drop out layer
        self.layer = layer #, batch_first=True)
        # Input dim for regularisation scaling
        self.input_dim = input_size
        # Regularisation hyper-parameters
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform_(self.p_logit, a=init_min, b=init_max)

        self.gpu_id = gpu_id

        self.use_dropout = True

    def forward(self, x):

        if self.use_dropout:
            return self.layer(self._concrete_dropout(x))
        #return self.layer(self._concrete_dropout(x))
        return self.layer(x)

    def regularisation(self):

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
        temp = 0.1
        self.p = nn.functional.sigmoid(self.p_logit)

        uniform_distribution = torch.distributions.uniform.Uniform(0, 1)

        unif_noise = uniform_distribution.sample(sample_shape=x.size())#.cuda(self.gpu_id)

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

    	#sum_of_square = 0
    	#for param in self.layer.parameters():
    #		sum_of_square += torch.sum(torch.pow(param, 2))

    #	return sum_of_square

        return torch.sum(torch.pow(self.layer.weight_ih_l0, 2)) + \
                torch.sum(torch.pow(self.layer.bias_ih_l0, 2))

        #return 0


class ConcreteLSTM (nn.Module):

    def __init__(self, vocab_size, output_dim, dropout ):

        super().__init__()

        embedding_dim = 100
        hidden_dim = 256

        self.embedding = nn.Embedding(vocab_size, embedding_dim)


        self.encoder_lstm1 = LSTMWithCDropout(nn.LSTM(embedding_dim, hidden_dim ), 
        					embedding_dim, hidden_dim, 0, 
            	    		dropout['wr'], dropout['dr'],
                    		dropout['init_min'], dropout['init_max'])

        self.decoder_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.embedding(x)
        output, (hidden, cell) = self.encoder_lstm1(x)
        return self.decoder_linear(hidden.squeeze(0))

    def regularisation_loss(self):

        reg_loss = self.encoder_lstm1.regularisation()
        return reg_loss


class HardLSTM(nn.Module):
    def __init__(self, vocab_size, output_dim ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, 100)
        self.rnn = nn.LSTM(100, 256)
        self.fc = nn.Linear(256, output_dim)
        
    def forward(self, x):

        embedded = self.embedding(x) 
        output, (hidden, cell) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))
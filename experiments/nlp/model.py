import numpy as np

import torch
from torch import nn

class LSTMWithCDropout(nn.Module):

    def __init__(self, input_size, output_size, gpu_id, weight_regularizer,
            dropout_regularizer, init_min, init_max):

        super(LSTMWithCDropout, self).__init__()
        # Post drop out layer
        self.layer = nn.LSTM(input_size, output_size, batch_first=True)
        # Input dim for regularisation scaling
        self.input_dim = input_size
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
        temp = 0.1
        self.p = nn.functional.sigmoid(self.p_logit)

        uniform_distribution = torch.distributions.uniform.Uniform(0, 1)

        unif_noise = uniform_distribution.sample(sample_shape=x.size()).cuda(self.gpu_id)

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
        return torch.sum(torch.pow(self.layer.weight_ih_l0, 2)) + \
                torch.sum(torch.pow(self.layer.bias_ih_l0, 2))


class Model(nn.Module):

    def __init__(self, graphemes_vocab_size, phonemes_vocab_size, seq_length,
            dropout):

        super(Model, self).__init__()

        embedding_size = 100
        lstm_thickness = 100

        self.mode = dropout['mode']

        self.seq_length = seq_length

        self.embedding = nn.Embedding(graphemes_vocab_size, embedding_size)

        if self.mode == 'dropout':
            self.encoder_dropout1 = nn.Dropout(dropout['prob'])
            self.encoder_lstm1 = nn.LSTM(embedding_size, lstm_thickness, batch_first=True)
        else:
            self.encoder_lstm1 = LSTMWithCDropout(embedding_size, lstm_thickness,
                    dropout['gpu_id'], dropout['wr'], dropout['dr'],
                    dropout['init_min'], dropout['init_max'])

        self.encoder_batchnorm2 = nn.BatchNorm1d(lstm_thickness)

        if self.mode == 'dropout':
            self.encoder_dropout2 = nn.Dropout(dropout['prob'])
            self.encoder_lstm2 = nn.LSTM(lstm_thickness, lstm_thickness, batch_first=True)
        else:
            self.encoder_lstm2 = LSTMWithCDropout(lstm_thickness, lstm_thickness,
                    dropout['gpu_id'], dropout['wr'], dropout['dr'],
                    dropout['init_min'], dropout['init_max'])

        self.encoder_batchnorm3 = nn.BatchNorm1d(lstm_thickness)

        if self.mode == 'dropout':
            self.encoder_dropout3 = nn.Dropout(dropout['prob'])
            self.encoder_lstm3 = nn.LSTM(lstm_thickness, lstm_thickness, batch_first=True)
        else:
            self.encoder_lstm3 = LSTMWithCDropout(lstm_thickness, lstm_thickness,
                    dropout['gpu_id'], dropout['wr'], dropout['dr'],
                    dropout['init_min'], dropout['init_max'])

        self.average_pooling = nn.AvgPool1d(seq_length)

        self.decoder_batchnorm1 = nn.BatchNorm1d(lstm_thickness)

        if self.mode == 'dropout':
            self.decoder_dropout1 = nn.Dropout(dropout['prob'])
            self.decoder_lstm1 = nn.LSTM(lstm_thickness, lstm_thickness, batch_first=True)
        else:
            self.decoder_lstm1 = LSTMWithCDropout(lstm_thickness, lstm_thickness,
                    dropout['gpu_id'], dropout['wr'], dropout['dr'],
                    dropout['init_min'], dropout['init_max'])

        self.decoder_batchnorm2 = nn.BatchNorm1d(lstm_thickness)

        if self.mode == 'dropout':
            self.decoder_dropout2 = nn.Dropout(dropout['prob'])
            self.decoder_lstm2 = nn.LSTM(lstm_thickness, lstm_thickness, batch_first=True)
        else:
            self.decoder_lstm2 = LSTMWithCDropout(lstm_thickness, lstm_thickness,
                    dropout['gpu_id'], dropout['wr'], dropout['dr'],
                    dropout['init_min'], dropout['init_max'])

        self.decoder_linear = nn.Linear(lstm_thickness, phonemes_vocab_size)

    def forward(self, x):

        # x of shape (batch, seq)

        x = self.embedding(x)

        if self.mode == 'dropout':
            x = self.encoder_dropout1(x)
        x = self.encoder_lstm1(x)[0]

        x = x.transpose(1, 2)
        x = self.encoder_batchnorm2(x)
        x = x.transpose(1, 2)
        if self.mode == 'dropout':
            x = self.encoder_dropout2(x)
        x = self.encoder_lstm2(x)[0]

        x = x.transpose(1, 2)
        x = self.encoder_batchnorm3(x)
        x = x.transpose(1, 2)
        if self.mode == 'dropout':
            x = self.encoder_dropout3(x)
        x = self.encoder_lstm3(x)[0]

        x = x.transpose(1, 2)
        x = self.average_pooling(x)
        x = torch.cat([x]*self.seq_length, 2)

        x = self.decoder_batchnorm1(x)
        x = x.transpose(1, 2)
        if self.mode == 'dropout':
            x = self.decoder_dropout1(x)
        x = self.decoder_lstm1(x)[0]

        x = x.transpose(1, 2)
        x = self.decoder_batchnorm2(x)
        x = x.transpose(1, 2)
        if self.mode == 'dropout':
            x = self.decoder_dropout2(x)
        x = self.decoder_lstm2(x)[0]

        x = self.decoder_linear(x)

        return nn.functional.log_softmax(x, 2)

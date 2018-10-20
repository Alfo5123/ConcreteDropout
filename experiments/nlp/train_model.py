import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import argparse
from pathlib import Path
import pickle

import torch
from torch import nn, optim
from torch.autograd import Variable

from model import Model
import utils

np.random.seed(42)


parser = argparse.ArgumentParser(description='NLP training')
parser.add_argument('--gpu_id', nargs=1, type=int)
parser.add_argument('--mode', nargs=1)
parser.add_argument('--model_id', nargs=1, type=int)
parser.add_argument('--dprob', nargs=1, type=float)
parser.add_argument('--dr', nargs=1, type=float)

parsed = parser.parse_args()

gpu_id = parsed.gpu_id[0]
mode = parsed.mode[0]
model_id = parsed.model_id[0]

if mode == 'dropout':
    drop_prob = parsed.dprob[0]
else:
    dr = parsed.dr[0]

Path('training_data/').mkdir(exist_ok=True)
Path('weights/').mkdir(exist_ok=True)


def get_regularization_loss(model):

    regularization_loss = 0.0

    def get_module_regularization_loss(module):

        nonlocal regularization_loss
        
        if module.__class__.__name__.endswith('LSTMWithCDropout'):
            regularization_loss = regularization_loss + module.regularisation()

    model.apply(get_module_regularization_loss)

    return regularization_loss

def set_dropout_state(model, value):

    def set_dropout_state_in_module(module):

        if module.__class__.__name__.endswith('LSTMWithCDropout'):
            module.use_dropout = value

    model.apply(set_dropout_state_in_module)


print('############# LOADING DATA ##############\n')

grapheme_codes, phoneme_codes, train_X, train_y, val_X, val_y, test_X, test_y = utils.load_data()

seq_length = train_X.shape[1]

print('############# ALLOCATING MODEL ##############\n')

epochs_count = 100
batch_size = 500
gpu_id = 0

if mode == 'dropout':
    dropout_data = {'mode': 'dropout', 'prob': drop_prob}
else:
    dropout_data = {'mode': 'concrete', 'gpu_id': gpu_id, 'wr': 1e-4, 'dr': dr,
                    'init_min': 0.05, 'init_max': 0.5}

model = Model(len(grapheme_codes) + 1, len(phoneme_codes) + 1, seq_length,
              dropout_data).cuda(gpu_id)

optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

criterion = nn.NLLLoss()

start = time.time()

training_data = []

print('############# TRAINING STARTS ##############\n')

def collect_dropout_probabilities(model):

    dropout_probabilities = []

    def add_dropout_prob(module):

        nonlocal dropout_probabilities

        if module.__class__.__name__.endswith('LSTMWithCDropout'):
            dropout_probabilities.append(float(module.p))

    model.apply(add_dropout_prob)

    return dropout_probabilities

for epoch in range(epochs_count):

    permutation = np.random.permutation(len(train_X))

    model.train()

    total_train_loss = 0.0
    train_mismatches_count = 0
    train_batches_count = 0
    
    for batch_start in range(0, len(train_X), batch_size):

        batch_end = min(len(train_X), batch_start + batch_size)

        if batch_start == batch_end:
            break

        batch_indices = permutation[batch_start:batch_end]

        batch_X = train_X[batch_indices]
        numpy_batch_y = train_y[batch_indices]

        batch_X = Variable(torch.LongTensor(batch_X), requires_grad=False).cuda(gpu_id)
        batch_y = Variable(torch.LongTensor(numpy_batch_y), requires_grad=False).cuda(gpu_id)

        output = model(batch_X)

        train_loss = criterion(output.transpose(1, 2), batch_y)

        output = output.detach().cpu().numpy().argmax(axis=2)

        train_mismatches_count += (output != numpy_batch_y).sum()
        total_train_loss += float(train_loss)
        train_batches_count += 1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    total_train_loss /= train_batches_count
    train_mismatches_ratio = train_mismatches_count/len(train_X)
 
    model.eval()

    val_train_loss = 0.0
    val_mismatches_count = 0
    val_batches_count = 0

    for batch_start in range(0, len(val_X), batch_size):

        batch_end = min(len(val_X), batch_start + batch_size)

        if batch_start == batch_end:
            break

        batch_X = val_X[batch_start:batch_end]
        numpy_batch_y = val_y[batch_start:batch_end]

        batch_X = Variable(torch.LongTensor(batch_X), requires_grad=False).cuda(gpu_id)
        batch_y = Variable(torch.LongTensor(numpy_batch_y), requires_grad=False).cuda(gpu_id)

        output = model(batch_X)

        val_loss = criterion(output.transpose(1, 2), batch_y)

        output = output.detach().cpu().numpy().argmax(axis=2)

        val_mismatches_count += (output != numpy_batch_y).sum()
        val_train_loss += float(val_loss)
        val_batches_count += 1

    val_train_loss /= val_batches_count
    val_mismatches_ratio = val_mismatches_count/len(val_X)

    time_lasted = (time.time() - start)/60
    
    print('Epoch {}'.format(epoch))
    print('{:.3f} minutes passed'.format(time_lasted))
    print('train-loss={0:.3f} train-score={1:.3f} val-loss={2:.3f} val-score={3:.3f}'.format(
        total_train_loss, train_mismatches_ratio, val_train_loss, val_mismatches_ratio
    ))

    training_data.append((time_lasted, total_train_loss, train_mismatches_ratio,
                          val_train_loss, val_mismatches_ratio, collect_dropout_probabilities(model)))
    
with open('training_data/training_data_{}.pk'.format(model_id), 'wb') as dump_file:
    pickle.dump(training_data, dump_file)

torch.save(model.cpu().state_dict(), 'weights/weights_{}.pth'.format(model_id))

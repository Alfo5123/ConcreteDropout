import os
import sys
import math
import string
import random
import shutil

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'


def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    #bs,h,w = preds.size()
    #n_pixels = bs*h*w

    class_ious = []

    for class_index in range(12):

        relevant = targets.eq(class_index)
        selected = preds.eq(class_index)

        relevant_and_selected_count = float(torch.min(relevant, selected).cpu().sum())
        relevant_or_selected_count = float(torch.max(relevant, selected).cpu().sum())

        class_iou = relevant_and_selected_count/(relevant_or_selected_count + 1e-8)

        class_ious.append(class_iou)

    return np.mean(class_ious)
        
    #incorrect = float(preds.ne(targets).cpu().sum())
    #err = incorrect/n_pixels
    #return round(err,5)

def _get_regularization_loss(model):

    regularization_loss = 0.0

    def get_module_regularization_loss(module):

        nonlocal regularization_loss
        
        if module.__class__.__name__.endswith('ConcreteDropout'):
            regularization_loss = regularization_loss + module.regularisation()

    model.apply(get_module_regularization_loss)

    return regularization_loss

def _set_dropout_state(model, value):

    def _set_dropout_state_in_module(module):

        if module.__class__.__name__.endswith('ConcreteDropout'):
            module.use_dropout = value

    model.apply(_set_dropout_state_in_module)

def train(model, trn_loader, optimizer, criterion, epoch, gpu_id, use_dropout):

    _set_dropout_state(model, use_dropout)
    
    model.train()
    trn_loss = 0
    reg_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        inputs = Variable(data[0].cuda(gpu_id))
        targets = Variable(data[1].cuda(gpu_id))

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        r_loss = _get_regularization_loss(model)
        (loss + r_loss).backward()
        optimizer.step()

        trn_loss += float(loss)
        reg_loss += float(r_loss)
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())

    trn_loss /= len(trn_loader)
    reg_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, reg_loss, trn_error

def test(model, test_loader, criterion, gpu_id, mc_samples_count, epoch=1):

    _set_dropout_state(model, mc_samples_count != 1)

    model.eval()
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        data = Variable(data.cuda(gpu_id), volatile=True)
        target = Variable(target.cuda(gpu_id))

        output = 0.0

        for sample_index in range(mc_samples_count):
            output = output + model(data).detach()

        output = output/mc_samples_count

        test_loss += criterion(output, target).data[0]
        pred = get_predictions(output)
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, gpu_id, mc_samples_count):

    _set_dropout_state(model, mc_samples_count != 1)

    #input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(gpu_id), volatile=True)
        #label = Variable(target.cuda(gpu_id))

        output = 0.0

        for sample_index in range(mc_samples_count):
            output = output + model(data).detach()

        output = output/mc_samples_count

        output = nn.functional.softmax(output, dim=1)

        #pred = get_predictions(output)
        predictions.append([target, output.cpu()])

    return predictions

def view_sample_predictions(model, loader, n, gpu_id):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(gpu_id), volatile=True)
    label = Variable(targets.cuda(gpu_id))
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])

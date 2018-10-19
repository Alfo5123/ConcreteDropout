import time
from pathlib import Path
import argparse
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils


parser = argparse.ArgumentParser(description='FC-DenseNet training')
parser.add_argument('--gpu_id', nargs=1, type=int)
parser.add_argument('--mode', nargs=1)
parser.add_argument('--model_id', nargs=1, type=int)

parsed = parser.parse_args()

gpu_id = parsed.gpu_id[0]
mode = parsed.mode[0]
model_id = parsed.model_id[0]


print('\n######### DATA PREPARATION #########\n')

CAMVID_PATH = Path('', 'SegNet-Tutorial/CamVid')
Path('training_data/').mkdir(exist_ok=True)
Path('weights/').mkdir(exist_ok=True)
train_batch_size = 2
val_batch_size = 2
train_image_size = 224
val_image_size = 224

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose([
    joint_transforms.JointRandomCrop(train_image_size),
    joint_transforms.JointRandomHorizontalFlip()
    ])
train_dset = camvid.CamVid(CAMVID_PATH, 'train',
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=train_batch_size, shuffle=True)

val_joint_transformer = transforms.Compose([
    joint_transforms.JointCenterCrop(val_image_size),
    ])

val_dset = camvid.CamVid(
    CAMVID_PATH, 'val', joint_transform=val_joint_transformer,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=val_batch_size, shuffle=False)

print("Train: %d" %len(train_loader.dataset.imgs))
print("Val: %d" %len(val_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())
print('')

print('\n######### MODEL INITIALIZATION #########\n')

LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 100
torch.cuda.manual_seed(0)

if mode == 'concrete':
    dropout_data = {
        'mode': 'concrete_dropout',
        'init_min': 0.01,
        'init_max': 0.05,
        'weight_reg': 1e-8,
        'dropout_reg': 1e0,
        'gpu_id': gpu_id
    }
elif mode == 'dropout':
    dropout_data = {'mode': 'dropout', 'prob': 0.2}
else:
    dropout_data = {'mode': 'dropout', 'prob': 0.0}

print(dropout_data)

model = tiramisu.FCDenseNet103(12, dropout_data).cuda(gpu_id)
model.apply(train_utils.weights_init)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda(gpu_id)).cuda(gpu_id)

print('\n######### TRAINING #########\n')

def collect_dropout_probabilities(model):

    dropout_probabilities = []

    def add_dropout_prob(module):

        nonlocal dropout_probabilities

        if module.__class__.__name__.endswith('ConcreteDropout'):
            dropout_probabilities.append(float(module.p))

    model.apply(add_dropout_prob)

    return dropout_probabilities

since = time.time()

training_data = {
    'mode': mode,
    'model_id': model_id,
    'dropout_data': dropout_data,
    'epoch_data': []
}

for epoch in range(1, N_EPOCHS + 1):

    ### Train ###
    trn_loss, reg_loss, trn_err = train_utils.train(
        model, train_loader, optimizer, criterion, epoch, gpu_id, True)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Reg - Loss: {:.4f}, IOU: {:.4f}'.format(
        epoch, trn_loss, reg_loss, trn_err))
    time_elapsed = time.time() - since
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### Test ###
    val_loss, val_err = train_utils.test(model, val_loader, criterion, gpu_id, 1,
            epoch=epoch)
    print('Val - Loss: {:.4f} | IOU: {:.4f}'.format(val_loss, val_err))
    time_elapsed = time.time() - since
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer,
                                     epoch, DECAY_EVERY_N_EPOCHS)

    training_data['epoch_data'].append(
            (trn_err, val_err, time_elapsed, collect_dropout_probabilities(model)))

with open('training_data/training_data_{}.pk'.format(model_id), 'wb') as dump_file:
    pickle.dump(training_data, dump_file)

torch.save(model.cpu().state_dict(), 'weights/weights_{}.pth'.format(model_id))

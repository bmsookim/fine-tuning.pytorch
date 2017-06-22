# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/pytorch-mammo
#
# Korea University, Data-Mining Lab
# Digital Mammogram Mass classification implementation
#
# Description : main.py
# The main code for training classification networks.
# ***********************************************************

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config as cf
import torchvision
import time
import copy
import os
import sys
import argparse

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--addlayer','-a',action='store_true', help='Add additional layer in fine-tuning')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Phase 1 : Data Upload
print('\n[Phase 1] : Data Preperation')

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ]),
}

data_dir = cf.aug_base
print("| Preparing %s dataset..." %(cf.data_base.split("/")[-1]))
dsets = {
    x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dset_loaders = {
    x : torch.utils.data.DataLoader(dsets[x], batch_size = cf.batch_size, shuffle=(x=='train'), num_workers=4)
    for x in ['train', 'val']
}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

use_gpu = torch.cuda.is_available()

# Phase 2 : Model setup
print('\n[Phase 2] : Model setup')

def getNetwork(args):
    if (args.net_type == 'vggnet'):
        net = VGG(args.finetune, args.depth)
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'resnet'):
        net = resnet(args.finetune, args.depth)
        file_name = 'resnet-%s' %(args.depth)
    else:
        print('Error : Network should be either [VGGNet / ResNet]')
        sys.exit(1)

    return net, file_name

# Training model
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=cf.num_epochs):
    since = time.time()

    best_model, best_acc = model, 0.0

    print('\n[Phase 3] : Training Model')
    print('| Training Epochs = %d' %num_epochs)
    print('| Initial Learning Rate = %f' %args.lr)
    print('| Optimizer = SGD')
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr = lr_scheduler(optimizer, epoch)
                print('\n=> Training Epoch #%d, LR=%f' %(epoch+1, lr))
                model.train(True)
            else:
                model.train(False)
                model.eval()

            running_loss, running_corrects, tot = 0.0, 0, 0

            for batch_idx, (inputs, labels) in enumerate(dset_loaders[phase]):
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # Forward Propagation
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # Backward Propagation
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Statistics
                running_loss += loss.data[0]
                running_corrects += preds.eq(labels.data).cpu().sum()
                tot += labels.size(0)

                if (phase == 'train'):
                    sys.stdout.write('\r')
                    sys.stdout.write('| Epoch [%2d/%2d] Iter[%3d/%3d]\t\tLoss %.4f\tAcc %.2f%%'
                            %(epoch+1, num_epochs, batch_idx+1,
                                (len(dsets[phase])//cf.batch_size)+1, loss.data[0], 100.*running_corrects/tot))
                    sys.stdout.flush()
                    sys.stdout.write('\r')

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc  = running_corrects / dset_sizes[phase]

            if (phase == 'val'):
                print('\n| Validation Epoch #%d\t\t\tLoss %.4f\tAcc %.2f%%'
                    %(epoch+1, loss.data[0], 100.*epoch_acc))

                if epoch_acc > best_acc:
                    print('| Saving Best model...\t\t\tTop1 %.2f%%' %(100.*epoch_acc))
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    state = {
                        'model': best_model,
                        'acc':   epoch_acc,
                        'epoch':epoch,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    save_point = './checkpoint/'
                    if not os.path.isdir(save_point):
                        os.mkdir(save_point)
                    torch.save(state, save_point+file_name+'.t7')

    time_elapsed = time.time() - since
    print('\nTraining completed in\t{:.0f} min {:.0f} sec'. format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc\t{:.2f}%'.format(best_acc*100))

    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=args.lr, lr_decay_epoch=10):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr

class MyResNet(nn.Module):
    def __init__(self, pretrained_model):
        super(MyResNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.bn = nn.BatchNorm2d(cf.feature_size)
        self.relu = nn.ReLU(inplace=True)
        self.last_layer = nn.Linear(cf.feature_size, len(dset_classes))

    def forward(self, x):
        return self.last_layer(self.pretrained_model(x))

model_ft, file_name = getNetwork(args)
if(args.finetune):
    print('| Fine-tuning pre-trained networks...')
    if(args.addlayer):
        print('| Add features of size %d' %cf.feature_size)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(dset_classes))
    else:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, cf.feature_size)
        model_ft = MyResNet(model_ft)

if use_gpu:
    model_ft = model_ft.cuda()

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=cf.num_epochs)

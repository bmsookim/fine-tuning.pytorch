# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/fine-tuning.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Fine tuning Implementation
#
# Description : extract_features.py
# The main code for extracting features of trained model.
# ***********************************************************

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
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
from PIL import Image
import pickle

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

data_dir = cf.test_dir
trainset_dir = cf.data_base.split("/")[-1] + os.sep
print("| Preparing %s dataset..." %(cf.test_dir.split("/")[-1]))

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

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print("| Loading checkpoint model for feature extraction...")
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
assert os.path.isdir('checkpoint/'+trainset_dir), 'Error: No model has been trained on the dataset!'
_, file_name = getNetwork(args)
checkpoint = torch.load('./checkpoint/'+trainset_dir+file_name+'.t7')
model = checkpoint['model']

print("| Consisting a feature extractor from the model...")
if(args.net_type == 'alexnet' or args.net_type == 'vggnet'):
    feature_map = list(checkpoint['model'].module.classifier.children())
    feature_map.pop()
    new_classifier = nn.Sequential(*feature_map)
    extractor = copy.deepcopy(checkpoint['model'])
    extractor.module.classifier = new_classifier
elif (args.net_type) == 'resnet'):
    feature_map = list(model.module.children())
    feature_map.pop()
    extractor = nn.Sequential(*feature_map)

if use_gpu:
    model.cuda()
    extractor.cuda()
    cudnn.benchmark = True

model.eval()
extractor.eval()

sample_input = Variable(torch.randn(1,3,224,224), volatile=True)
if use_gpu:
    sample_input = sample_input.cuda()

sample_output = extractor(sample_input)
featureSize = sample_output.size(1)
print("| Feature dimension = %d" %featureSize)

print("\n[Phase 3] : Feature & Score Extraction")

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean, cf.std)
])

if not os.path.isdir('vectors'):
    os.mkdir('vectors')

for subdir, dirs, files in os.walk(data_dir):
    for f in files:
        file_path = subdir + os.sep + f
        if (is_image(f)):
            vector_dict = {
                'file_path': "",
                'feature': [],
                'score': 0,
            }

            image = Image.open(file_path).convert('RGB')
            if test_transform is not None:
                image = test_transform(image)
            inputs = image
            inputs = Variable(inputs, volatile=True)
            if use_gpu:
                inputs = inputs.cuda()
            inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) # add batch dim in the front
            features = extractor(inputs).view(featureSize)

            outputs = model(inputs)
            softmax_res = softmax(outputs.data.cpu().numpy()[0])

            vector_dict['file_path'] = file_path
            vector_dict['feature'] = features
            vector_dict['score'] = softmax_res[1]

            vector_file = 'vectors' + os.sep + os.path.splitext(f)[0] + ".pickle"

            print(vector_file)
            print(vector_dict['feature'].size())
            print(vector_dict['score'])

            with open(vector_file, 'wb') as pkl:
                pickle.dump(vector_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)

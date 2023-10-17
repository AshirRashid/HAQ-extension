# Implementation for the FPEC-Opt in Compensated DNN paper

import os
import time
import math
import random
import shutil
import argparse
from copy import deepcopy
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
import models as customized_models

from lib.utils.utils import Logger, AverageMeter, accuracy
from lib.utils.data_utils import get_dataset
from progress.bar import Bar
from lib.utils.quantize_utils import QConv2d, QLinear, QModule


# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='data/imagenet', type=str)
parser.add_argument('--data_name', default='imagenet', type=str)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup_epoch', default=0, type=int, metavar='N',
                    help='manual warmup epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=512, type=int, metavar='N',
                    help='test batchsize (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='cos', type=str,
                    help='lr scheduler (exp/cos/step3/fixed)')
parser.add_argument('--schedule', type=int, nargs='+', default=[31, 61, 91],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
# Quantization
parser.add_argument('--linear_quantization', dest='linear_quantization', action='store_true',
                    help='quantize both weight and activation)')
parser.add_argument('--free_high_bit', default=True, type=bool,
                    help='free the high bit (>6)')
parser.add_argument('--half', action='store_true',
                    help='half')
parser.add_argument('--half_type', default='O1', type=str,
                    help='half type: O0/O1/O2/O3')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) + ' (default: resnet50)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu_id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
lr_current = state['lr']

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def test(val_loader, model, criterion, epoch=0, use_cuda=True):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    print(top1.avg)
    return losses.avg, top1.avg

def qlayers(dnn: torch.nn.Module) -> Iterator[QModule]:
    """Returns a generator yielding all modules in dnn of type QLinear or QConv2d (i.e. quantizeable layers)
    """
    return (module for _, module in dnn.named_modules() if type(module) in [QConv2d, QLinear])

def evaluate_accuracy(dnn: torch.nn.Module) -> float:
    _, accuracy = test(val_loader, dnn, criterion)
    return accuracy

def get_stats_for_layer(qlayer: torch.nn.Module) -> dict:
    """Get stats for each layer.
    The equivalent of getStatsForEachLayer in the Compensated DNN paper, but only for one layer
    TODO: fix this. get_stats_for_layer should also have access to the activations. This can only be done if it is used in the _quantize_activations
    """
    return {
        "weight_min": qlayer.weight().min().item(),
        "weight_max": qlayer.weight().max().item(),
        "value_dist": torch.historgram(input=qlayer.weight()),
    }

def identify_ib_and_fb_for_each_layer(non_quantized_dnn: torch.nn.Module) -> torch.nn.Module:
    """Sets the IB (integer bits) and FB (fractional bits) attributes of the model.
    This change quantizes the model since a fixed point representation is used to represent weights and activations.
    """
    for qlayer in qlayers(non_quantized_dnn):
        stats_dict = get_stats_for_layer(qlayer)
        qlayer.ib = torch.full(qlayer.weight.shape, 4)
        qlayer.fb = torch.full(qlayer.weight.shape, 4)
    return non_quantized_dnn

def set_default_ezb_thresh(c_dnn: torch.nn.Module) -> float:
    """ezb threshold to be determined using the emb tensors for each layer.
    The emb tensors for each layer can be accessed my iterating through c_dnn.modules()
    """
    return 0.

def increase_error_sparsity(ezb_thresh, c_dnn) -> torch.nn.Module:
    """Set the ezb for each layer to 1 or 0 depending on the ezb_thresh
    If the estimated error is greater than ezb_thresh, ezb is set to 1 (ignoring this error to save error compensation computation)
    The error is ignored by setting it to 0 so it is not taken into account
    """
    for qlayer in qlayers(c_dnn):
        qlayer.calculate_est()
        qlayer.cutoff_est()
    return c_dnn

def fpec_opt(non_quantized_dnn: torch.nn.Module, max_loss: float) -> torch.nn.Module:
    # To check for quantizable layers, iterate over model.modules() and check if the type is QConv2d or QLinear.
    # TODO: extend the implementation to also work for activations in addition to weights
    EZB_THRESH_INC = 0.2 # ezb threshold increment

    print("INITIAL ACCURACY, SET IB & FB")
    accuracy: float = evaluate_accuracy(non_quantized_dnn)
    # c_dnn: torch.nn.Module = identify_ib_and_fb_for_each_layer(non_quantized_dnn) # Compensated dnn
    for qlayer in qlayers(non_quantized_dnn):
        qlayer._initialize_compensated_dnn_attrs = True
    print("FORWARD PASS FOR _initialize_compensated_dnn_attrs")
    # print(id(next(qlayers(non_quantized_dnn))))
    inputs, _ = next(iter(val_loader))
    with torch.no_grad():
        non_quantized_dnn(inputs)   # forward pass to run the _initialize_compensated_dnn_attrs logic in _quantize function
                                    # initializes fb, ib & emb, ezb, edb for weights and activations
    print("INITIALIZED")
    for qlayer in qlayers(non_quantized_dnn):
        qlayer._initialize_compensated_dnn_attrs = False
        qlayer._use_compensated_dnn = True
        qlayer.calculate_est()

    c_dnn_temp: torch.nn.Module = deepcopy(non_quantized_dnn)
    print("EMB & FB WHILE")
    while (accuracy - evaluate_accuracy(c_dnn_temp)) < max_loss:
        c_dnn = deepcopy(c_dnn_temp)
        while (accuracy - evaluate_accuracy(c_dnn_temp)) < max_loss:
            c_dnn = deepcopy(c_dnn_temp)
            for qlayer in qlayers(non_quantized_dnn):
                qlayer.compensated_dnn_attrs.fb_weight -= 1
                qlayer.compensated_dnn_attrs.fb_activation -= 1
        for qlayer in qlayers(non_quantized_dnn):
            qlayer.compensated_dnn_attrs.emb_weight += 1
            qlayer.compensated_dnn_attrs.emb_activation += 1
    
    print("EZB WHILE")
    ezb_thresh: float = set_default_ezb_thresh(c_dnn)
    while (accuracy - evaluate_accuracy(c_dnn_temp))< max_loss:
        c_dnn = deepcopy(c_dnn_temp)
        ezb_thresh += EZB_THRESH_INC
        c_dnn_temp = increase_error_sparsity(ezb_thresh, c_dnn_temp)

    return c_dnn


if __name__ == '__main__':
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    train_loader, val_loader, n_class = get_dataset(dataset_name=args.data_name, batch_size=args.train_batch,
                                                    n_worker=args.workers, data_root=args.data)

    model = models.__dict__[args.arch](pretrained=args.pretrained)
    print("=> creating model '{}'".format(args.arch), ' pretrained is ', args.pretrained)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model).cuda()

    if args.evaluate:
        print('\nEvaluation only (Compensated DNN)')
        fpec_opt(model, 5.)

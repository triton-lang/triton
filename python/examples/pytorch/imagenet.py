from __future__ import print_function
import argparse
import os
import time, timeit

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models
import isaac.pytorch.models



def main():
    # Program options
    parser = argparse.ArgumentParser(description='ISAAC ImageNet Inference')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--workers', '-j', default=4, type=int, metavar='N',  help='number of workers for [default: 4]')
    parser.add_argument('--batch-size', '-b', default=32, type=int, metavar='N', help='mini-batch size [default: 16]')
    parser.add_argument('--calibration-batches', '-c', default=16, type=int, metavar='N', help='number of batches for calibration [default: 16]')
    args = parser.parse_args()

    # Fix random seeds (for reproducibility)
    np.random.seed(0)

    # Build data-loader
    val_dir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.Resize(256),  transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    image_folder = torchvision.datasets.ImageFolder(val_dir, transformations)
    val_loader = torch.utils.data.DataLoader(image_folder, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Build models
    print('Quantizing... ', end='', flush=True)
    #resnet_ref = torchvision.models.__dict__[args.arch](pretrained=True).cuda()
    #resnet_ref.eval()
    resnet_ref = isaac.pytorch.models.resnet(args.arch).cuda()
    resnet_sc = isaac.pytorch.models.resnet(args.arch).cuda()
    isaac.pytorch.quantize(resnet_sc, val_loader, args.calibration_batches)
    print('')

    # Benchmark
    print('Performance: ', end='', flush=True)
    input, target = next(iter(val_loader))
    with torch.no_grad():
        input = torch.autograd.Variable(input).cuda()
    y_ref = resnet_ref(input)
    y_sc = resnet_sc(input)
    t_sc = [x for x in timeit.repeat(lambda: (resnet_sc(input), torch.cuda.synchronize()), repeat=10, number=1)]
    t_ref = [x for x in timeit.repeat(lambda: (resnet_ref(input), torch.cuda.synchronize()), repeat=10, number=1)]
    print('{:.2f} Image/s (INT8) vs. {:.2f} Image/s (FP32)'.format(input.size()[0]/min(t_sc), input.size()[0]/min(t_ref)))

    # Accuracy
    criterion = nn.CrossEntropyLoss().cuda()
    validate(val_loader, resnet_sc, criterion)


def validate(val_loader, model, criterion, progress_frequency = 10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(float(loss.data), float(input.size(0)))
        top1.update(float(prec1), float(input.size(0)))
        top5.update(float(prec5), float(input.size(0)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % progress_frequency == 0 or i == len(val_loader) - 1:
            print('Accuracy [{0}/{1}]: {2:.4} ({3:.4})'.format(i, len(val_loader), top1.avg, top5.avg), end='\r', flush=True)

    print('')
    return top1.avg, top5.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

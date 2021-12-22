#!/usr/bin/env python
# coding: utf-8

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from trainer import Trainer

def initSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    initSeed(args.seed)

    cfg = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    print('config:', cfg)

    transform = Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5)
    ])
    
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)

    ## label transition matrix
    noise_type = "symmetry"
    noise_rate = 0.2
    num_classes = 10
    if noise_type == "symmetry":
        Q = torch.empty(size=(num_classes, num_classes))
        Q.fill_(noise_rate / (num_classes - 1))
        Q += (1.0 - noise_rate - Q[0,0]) * torch.eye(10)
    else:
        pass
    print(Q[0])

    noised_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    noised_trainset.targets = torch.multinomial(Q[trainset.targets], 1)
    
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('layer1', nn.Linear(784, 10))
    ]))

    print('start training')
    trainer = Trainer(model, trainset, testset, cfg)
    trainer.run()

    
if __name__ == '__main__':
    main()
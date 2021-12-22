#!/usr/bin/env python
# coding: utf-8

import argparse

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

from trainer import Trainer
import models

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
    parser.add_argument('--model_name', type=str)
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
    
    trainset = CIFAR10(root='../data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='../data', train=False, download=True, transform=transform)

    model = models.string2model(args.model_name)

    print('start training')
    trainer = Trainer(model, trainset, testset, cfg)
    trainer.run()

    
if __name__ == '__main__':
    main()
#!/usr/bin/env python
# coding: utf-8

import copy
import argparse

import torch
import torch.nn as nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

from trainer import Trainer
from cotrainer import CoTrainer

def initSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--mode', type=str, default='coteaching')
    parser.add_argument('--vis_idx', type=int, default=-1)
    args = parser.parse_args()

    initSeed(args.seed)

    transform = Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5)
    ])
    
    if args.dataset == 'mnist':
        trainset = MNIST(root='../data', train=True, download=True, transform=transform)
        testset = MNIST(root='../data', train=False, download=True, transform=transform)
    else:
        trainset = CIFAR10(root='../data', train=True, download=True, transform=transform)
        testset = CIFAR10(root='../data', train=False, download=True, transform=transform)

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

    noised_trainset = copy.copy(trainset)
    noised_trainset.targets = torch.multinomial(Q[trainset.targets], 1)

    class_id, num_per_class = torch.unique(trainset.targets, return_counts=True)
    print(f"train data per class:{dict(zip(class_id.tolist(), num_per_class.tolist()))}")
    class_id, num_per_class = torch.unique(testset.targets, return_counts=True)
    print(f"test data per class:{dict(zip(class_id.tolist(), num_per_class.tolist()))}")
    class_id, num_per_class = torch.unique(noised_trainset.targets, return_counts=True)
    print(f"noised train data per class:{dict(zip(class_id.tolist(), num_per_class.tolist()))}")
    
    model_f = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    model_g = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    cfg = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'noise_rate': noise_rate
    }
    print('config:', cfg)

    print('start training')
    if args.mode == 'standard-plain':
        trainer = Trainer(model_f, trainset, testset, cfg, args.vis_idx)
    elif args.mode == 'standard-noise':
        trainer = Trainer(model_f, noised_trainset, testset, cfg, args.vis_idx)
    else:
        trainer = CoTrainer(model_f, model_g, noised_trainset, testset, cfg, args.vis_idx)
    trainer.run()
    trainer.save_hist()

    
if __name__ == '__main__':
    main()
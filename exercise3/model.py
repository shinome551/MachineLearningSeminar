#!/usr/bin/env python
# coding: utf-8

import argparse

import torch
import torch.nn as nn


def string2model(name):
    if name == 'problem1':
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    elif name == 'problem2':
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(-1, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    elif name == 'problem3':
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.AvgPool2d(4).
            nn.Flatten(),
            nn.Linear(-1, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    elif name == 'problem4':
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 500, kernel_size=1),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    return model

   
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    model = string2model(args.name)

    inputs = torch.randn(1,3,32,32)
    try:
        outputs = model(inputs)
        print('OK!')
    except RuntimeError:
        print('INCORRECT...')

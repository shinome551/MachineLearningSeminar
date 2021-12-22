#!/usr/bin/env python
# coding: utf-8

import os
import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, trainset, testset, cfg, vis_idx=-1):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_epochs = cfg['num_epochs']
        self.model = model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=cfg['learning_rate'])
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=
                     lambda epoch: (cfg['num_epochs'] - epoch) / int(0.6 * cfg['num_epochs']) if epoch > int(0.4 * cfg['num_epochs']) else 1.0)
        batch_size = cfg['batch_size']
        self.train_num = len(trainset)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=2, 
                                pin_memory=True,
                                drop_last=True)
        self.test_num = len(testset)
        self.testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True,
                                drop_last=True)
        self.input_shape = trainset[0][0].shape
        self.vis_idx = vis_idx


    def train(self):
        self.model.train()
        trainloss = 0
        trainacc = 0
        for data in self.trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()
            trainloss += loss.item() * inputs.size()[0]
            _, predicted = torch.max(outputs.data, 1)
            trainacc += (predicted == labels).sum().item()

        trainloss = trainloss / self.train_num
        trainacc = 100 * trainacc / self.train_num
        return trainloss, trainacc


    def test(self):
        self.model.eval()
        testloss = 0
        testacc = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                testloss += loss.item() * inputs.size()[0]
                _, predicted = torch.max(outputs.data, 1)
                testacc += (predicted == labels).sum().item()

        testloss = testloss / self.test_num
        testacc = 100 * testacc / self.test_num
        return testloss, testacc


    def plot(self):
        plt.gcf().clear()
        weight_vis = True if self.vis_idx >= 0 else False

        x = range(1, len(self.train_hist['trainloss']) + 1)
        ax1 = self.fig.add_subplot(1 + int(weight_vis), 2, 1)
        ax1.plot(x, self.train_hist['trainloss'], label='train', marker='o')
        ax1.plot(x, self.train_hist['testloss'], label='test', marker='o')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel('epoch')
        ax1.set_xticks(x)
        ax1.set_ylabel('Loss')

        ax2 = self.fig.add_subplot(1 + int(weight_vis), 2, 2)
        ax2.plot(x, self.train_hist['trainacc'], label='train', marker='o')
        ax2.plot(x, self.train_hist['testacc'], label='test', marker='o')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlabel('epoch')
        ax2.set_xticks(x)
        ax2.set_ylabel('Accuracy')

        if weight_vis:
            weight = self.model[self.vis_idx].weight.data[:10]
            for i in range(10):
                ax = self.fig.add_subplot(2, 10, 11 + i, xticks=[], yticks=[])
                if self.model[self.vis_idx].__class__.__name__ == 'Linear':
                    img = weight[i].reshape(self.input_shape).mean(0)
                else:
                    img = weight[i].mean(0)
                img = (img - img.min()) / (img.max() - img.min())
                ax.imshow(img, cmap='gray', interpolation='none')

        plt.pause(1)


    def save_hist(self, save_dir='outputs'):
        with open(os.path.join(save_dir, 'standard.json'), mode='wt', encoding='utf-8') as f:
            json.dump(self.train_hist, f, ensure_ascii=False, indent=4)


    def run(self, save_dir='outputs'):
        self.train_hist = {}
        self.train_hist['trainloss'] = []
        self.train_hist['trainacc'] = []
        self.train_hist['testloss'] = []
        self.train_hist['testacc'] = []
        self.fig = plt.figure(figsize=(9, 6))
        self.plot()

        for epoch in range(self.num_epochs):
            trainloss, trainacc = self.train()
            testloss, testacc = self.test()
            print(f'epoch:{epoch+1}, trainloss:{trainloss:.3f}, testacc:{testacc:.1f}%')
            self.scheduler.step()
            self.train_hist['trainloss'].append(trainloss)
            self.train_hist['trainacc'].append(trainacc)
            self.train_hist['testloss'].append(testloss)
            self.train_hist['testacc'].append(testacc)
            self.plot()

        plt.savefig(os.path.join(save_dir, 'standard.png'))
        plt.show()

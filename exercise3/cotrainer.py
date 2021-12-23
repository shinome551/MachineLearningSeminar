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


class CoTrainer:
    def __init__(self, model_f, model_g, trainset, testset, cfg, vis_idx=-1):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_epochs = cfg['num_epochs']
        self.model_f = model_f.to(self.device)
        self.model_g = model_g.to(self.device)
        self.optimizer = Adam(list(self.model_f.parameters()) + list(self.model_g.parameters()), lr=cfg['learning_rate'])
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=
                     lambda epoch: (cfg['num_epochs'] - epoch) / int(0.6 * cfg['num_epochs']) if epoch > int(0.4 * cfg['num_epochs']) else 1.0)
        self.batch_size = cfg['batch_size']
        self.train_num = len(trainset)
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=2, 
                                pin_memory=True,
                                drop_last=True)
        self.test_num = len(testset)
        self.testloader = DataLoader(testset, batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True,
                                drop_last=True)
        self.input_shape = trainset[0][0].shape
        self.vis_idx = vis_idx
        self.noise_rate = cfg['noise_rate']
        self.mode = cfg['mode']


    def train(self, lamd):
        v_batch_size = int(lamd * self.batch_size) 
        trainloss_f = 0
        trainloss_g = 0
        trainacc_f = 0
        trainacc_g = 0
        iter = 0
        for (inputs, labels) in self.trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            ## model_f's forward
            outputs_f = self.model_f(inputs)
            loss_f = F.cross_entropy(outputs_f, labels, reduction='none')
            idx = torch.argsort(loss_f.data)[v_batch_size-1]
            weight_f = (loss_f.data <= loss_f.data[idx]).to(torch.float32) / v_batch_size
            with torch.no_grad():
                _, predicted = torch.max(outputs_f.data, 1)
                trainacc_f += (predicted == labels).sum().item()
            ## model_g's forward
            outputs_g = self.model_g(inputs)
            loss_g = F.cross_entropy(outputs_g, labels, reduction='none')
            idx = torch.argsort(loss_g.data)[v_batch_size-1]
            weight_g = (loss_g.data <= loss_g.data[idx]).to(torch.float32) / v_batch_size
            with torch.no_grad():
                _, predicted = torch.max(outputs_g.data, 1)
                trainacc_g += (predicted == labels).sum().item()
            ## backward
            loss_f.backward(weight_g)
            loss_g.backward(weight_f)
            self.optimizer.step()
            trainloss_f += (loss_f.data * weight_g).sum().item()
            trainloss_g += (loss_g.data * weight_f).sum().item()
            iter += 1

        trainloss = 0.5 * (trainloss_f + trainloss_g) / iter
        trainacc = 100 * 0.5 * (trainacc_f + trainacc_g) / self.train_num
        return trainloss, trainacc


    def test(self):
        testloss_f = 0
        testloss_g = 0
        testacc_f = 0
        testacc_g = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                ## model_f test
                outputs = self.model_f(inputs)
                loss = F.cross_entropy(outputs, labels)
                testloss_f += loss.item() * inputs.size()[0]
                _, predicted = torch.max(outputs.data, 1)
                testacc_f += (predicted == labels).sum().item()
                ## model_g test
                outputs = self.model_g(inputs)
                loss = F.cross_entropy(outputs, labels)
                testloss_g += loss.item() * inputs.size()[0]
                _, predicted = torch.max(outputs.data, 1)
                testacc_g += (predicted == labels).sum().item()

        testloss = 0.5 * (testloss_f + testloss_g) / self.test_num
        testacc = 100 * 0.5 * (testacc_f + testacc_g) / self.test_num
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
        #ax1.set_xticks(x)
        ax1.set_ylabel('Loss')

        ax2 = self.fig.add_subplot(1 + int(weight_vis), 2, 2)
        ax2.plot(x, self.train_hist['trainacc'], label='train', marker='o')
        ax2.plot(x, self.train_hist['testacc'], label='test', marker='o')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlabel('epoch')
        #ax2.set_xticks(x)
        ax2.set_ylabel('Accuracy')

        if weight_vis:
            weight = self.model_f[self.vis_idx].weight.data[:10]
            for i in range(10):
                ax = self.fig.add_subplot(2, 10, 11 + i, xticks=[], yticks=[])
                if self.model_f[self.vis_idx].__class__.__name__ == 'Linear':
                    img = weight[i].reshape(self.input_shape).mean(0)
                else:
                    img = weight[i].mean(0)
                img = (img - img.min()) / (img.max() - img.min())
                ax.imshow(img, cmap='gray', interpolation='none')

        plt.pause(1)


    def save_hist(self):
        with open(os.path.join('outputs', self.mode + '.json'), mode='wt', encoding='utf-8') as f:
            json.dump(self.train_hist, f, ensure_ascii=False, indent=4)


    def run(self):
        self.train_hist = {}
        self.train_hist['trainloss'] = []
        self.train_hist['testacc'] = []
        self.train_hist['testloss'] = []
        self.train_hist['trainacc'] = []
        self.fig = plt.figure(figsize=(12, 6))
        self.plot()

        E_k = 10
        for epoch in range(self.num_epochs):
            lamd = 1.0 - min(self.noise_rate * epoch / E_k, self.noise_rate)
            trainloss, trainacc = self.train(lamd)
            testloss, testacc = self.test()
            print(f'epoch:{epoch+1}, trainloss:{trainloss:.3f}, testacc:{testacc:.1f}%', end='\r')
            self.scheduler.step()
            self.train_hist['trainloss'].append(trainloss)
            self.train_hist['testloss'].append(testloss)
            self.train_hist['trainacc'].append(trainacc)
            self.train_hist['testacc'].append(testacc)
            self.plot()

        plt.savefig(os.path.join('outputs', self.mode + '.png'))
        plt.show()

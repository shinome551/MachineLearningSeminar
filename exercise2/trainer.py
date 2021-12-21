#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, trainset, testset, cfg, seed=42):
        self.initSeed(seed)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.num_epochs = cfg['num_epochs']
            
        self.model = model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=cfg['learning_rate'])

        batch_size = cfg['batch_size']
        self.trainloader = DataLoader(trainset, batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=2, 
                                pin_memory=True)
        self.testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True)


    def initSeed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

        trainloss = trainloss / len(self.trainloader.dataset)
        trainacc = 100 * trainacc / len(self.trainloader.dataset)
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

        testloss = testloss / len(self.testloader.dataset)
        testacc = 100 * testacc / len(self.testloader.dataset)
        return testloss, testacc


    def plot(self):
        plt.gcf().clear()

        x = range(1, len(self.train_hist['trainloss']) + 1)
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.plot(x, self.train_hist['trainloss'], label='train')
        ax1.plot(x, self.train_hist['testloss'], label='test')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('Loss')

        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.plot(x, self.train_hist['trainacc'], label='train')
        ax2.plot(x, self.train_hist['testacc'], label='test')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('Accuracy')

        plt.pause(1)


    def run(self):
        self.train_hist = {}
        self.train_hist['trainloss'] = []
        self.train_hist['trainacc'] = []
        self.train_hist['testloss'] = []
        self.train_hist['testacc'] = []
        self.fig = plt.figure(figsize=(9, 6))

        for epoch in range(self.num_epochs):
            trainloss, trainacc = self.train()
            testloss, testacc = self.test()
            print(f'epoch:{epoch+1}, trainloss:{trainloss:.3f}, testacc:{testacc:.1f}%')
            self.train_hist['trainloss'].append(trainloss)
            self.train_hist['trainacc'].append(trainacc)
            self.train_hist['testloss'].append(testloss)
            self.train_hist['testacc'].append(testacc)
            self.plot()


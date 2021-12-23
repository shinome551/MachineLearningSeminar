#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt

def main():
    x = torch.arange(-2, 2, 0.1)
    x_sample = x[torch.randperm(len(x))[:8]]

    def f(x, w):
        return w[1] * x + w[0]
    w_true = [1, 2]
    f_x = f(x_sample, w_true)
    y = f_x + 0.6 * torch.randn_like(f_x)

    w = torch.randn(len(w_true), requires_grad=True)
    lr = 0.1
    loss_list = []
    w_list = [w.data.tolist()]
    for i in range(50):
        f_w = f(x_sample, w)
        loss = torch.mean((y - f_w) ** 2)
        grad = torch.autograd.grad(loss, w)[0]
        with torch.no_grad():
            w += lr * grad
        w_list.append(w.data.tolist())
        loss_list.append(loss.item())
        print(f'iter:{i}, loss:{loss_list[-1]:.3f}, w0:{w[0].item()}, w1:{w[1].item()}', end='\r')
        if len(loss_list) > 1 and abs(loss_list[-1] - loss_list[-2]) < 0.001:
            print('\nconvergence!')
            break

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.grid()
    ax2.grid()
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax2.set_xlabel("w0")
    ax2.set_ylabel("w1")
    ax1.scatter(x_sample, y)
    ax2.scatter(w_true[0], w_true[1], color='black')
    for i, (w, loss) in enumerate(zip(w_list, loss_list)):
        ax1.plot(x, f(x,w), color='r', alpha=0.3+(0.7*i/len(w_list)))
        ax2.plot(w[0], w[1], color='r', marker='o', linestyle = 'None', alpha=0.3+(0.7*i/len(w_list)))
    
    plt.show()

if __name__ == '__main__':
    main()
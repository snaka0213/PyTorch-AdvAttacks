#!/usr/bin/env python3
import functools
import torch
import torch.nn as nn
from torch.autograd import Variable
from .attack import attack

def _pgd(model, X, y, epsilon, alpha, n_iters):
    out = model(X)
    _, predicted = torch.max(out.data, 1)
    acc = (predicted == y.data).float().sum().item() / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(n_iters):
        out_pgd = model(X_pgd)

        model.zero_grad()
        loss = nn.CrossEntropyLoss()(out_pgd, y)
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_pgd.data - X.data, min=-epsilon, max=epsilon)
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

    out_pgd = model(X_pgd)
    _, predicted_pgd = torch.max(out_pgd.data, 1)
    acc_pgd = (predicted_pgd == y.data).float().sum().item() / X.size(0)

    return acc, acc_pgd

def pgd(loader, model, epsilon=0.3, alpha=0.01, n_iters=40, use_cuda=False, verbose=False):
    _atk = functools.partial(_pgd, epsilon=epsilon, alpha=alpha, n_iters=n_iters)
    return attack(loader, model, atk=_atk, use_cuda=use_cuda, verbose=verbose)

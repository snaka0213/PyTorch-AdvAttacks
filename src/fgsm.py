#!/usr/bin/env python3
import functools
import torch
import torch.nn as nn
from .attack import attack

def _fgsm(model, X, y, epsilon):
    X.requires_grad = True
    out = model(X)
    _, predicted = torch.max(out.data, 1)
    acc = (predicted == y.data).float().sum().item() / X.size(0)

    model.zero_grad()
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()

    eta = epsilon*X.grad.data.sign()
    X_fgsm = X.data + eta

    out_fgsm = model(X_fgsm)
    _, predicted_fgsm = torch.max(out_fgsm.data, 1)
    acc_fgsm = (predicted_fgsm == y.data).float().sum().item() / X.size(0)

    return acc, acc_fgsm

def fgsm(loader, model, epsilon=0.1, use_cuda=False, verbose=False):
    _atk = functools.partial(_fgsm, epsilon=epsilon)
    return attack(loader, model, atk=_atk, use_cuda=use_cuda, verbose=verbose)

#!/usr/bin/env python3
import torch

def attack(loader, model, atk, use_cuda=False, verbose=False):
    total_acc, total_acc_atk = [], []
    device = "cuda" if use_cuda else "cpu"

    if verbose:
        print("Requiring no gradients for parameters.")
    for p in model.parameters():
        p.requires_grad = False

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        acc, acc_atk = atk(model, X, y)
        total_acc.append(acc)
        total_acc_atk.append(acc_atk)

        if verbose:
            print("batch [{}/{}], acc: {}, atk: {}"\
                .format(i+1, len(loader), acc, acc_atk))

    return total_acc, total_acc_atk

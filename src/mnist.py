#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.utils.data as td
import argparse

def argparser(
    batch_size=50, epochs=100, verbose=0, lr=1e-3,
    opt='adam', momentum=0.9, beta_1=0.9, beta_2=0.999,
    eps=1e-8, weight_decay=0):

    parser = argparse.ArgumentParser()

    # optimiser for training model
    parser.add_argument("--opt", default=opt)
    parser.add_argument("--momentum", default=momentum)
    parser.add_argument('--beta_1', type=float, default=beta_1)
    parser.add_argument('--beta_2', type=float, default=beta_2)
    parser.add_argument("--eps", default=eps)
    parser.add_argument("--weight_decay", default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--test_batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)

    # others
    parser.add_argument("--prefix", default="mnist")
    parser.add_argument("--verbose", type=int, default=verbose)

    args = parser.parse_args()
    if args.prefix is not None:
        banned = [
            'prefix', 'verbose', 'test_batch_size', 'lr',
            'momentum', 'beta_1', 'beta_2', 'eps', 'weight_decay',
        ]
        for arg in sorted(vars(args)):
            if arg not in banned and getattr(args,arg) is not None:
                args.prefix += '_' + arg + '_' +str(getattr(args, arg))

    return args

def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*7*7, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_loaders(batch_size: int):
    dataset_dir = os.path.dirname(os.path.abspath(__file__))+"/../data"
    mnist_train = datasets.MNIST(
        root=dataset_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    mnist_test = datasets.MNIST(
        root=dataset_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    train_loader = td.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_loader = td.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    return train_loader, test_loader

def train(loader, model, opt, epoch):
    model.train()
    train_loss, train_err = 0, 0

    for i, (X, y) in enumerate(loader):
        if y.dim() == 2:
            y = y.squeeze(1)

        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.max(1)[1] != y).float().sum() / X.size(0)

        train_loss += ce.item()
        train_err += err.item()

        opt.zero_grad() # initialize grad
        ce.backward() # back propagation
        opt.step() # update parameters

    torch.cuda.empty_cache()
    avg_train_loss = train_loss / len(loader)
    avg_train_err = train_err / len(loader)

    return avg_train_loss, avg_train_err

def test(loader, model, epoch):
    model.eval()
    val_loss, val_err = 0, 0

    for i, (X, y) in enumerate(loader):
        if y.dim() == 2:
            y = y.squeeze(1)

        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.max(1)[1] == y).float().sum() / X.size(0)

        val_loss += ce.item()
        val_err += err.item()

    torch.cuda.empty_cache()
    avg_val_loss = val_loss / len(loader)
    avg_val_err = val_err / len(loader)

    return avg_val_loss, avg_val_err

if __name__ == "__main__":
    args = argparser()

    # model
    model = mnist_model()

    # loaders
    train_loader, _ = mnist_loaders(args.batch_size)
    _, test_loader = mnist_loaders(args.test_batch_size)

    # optimizer
    if args.opt == 'adam':
        betas = (args.beta_1, args.beta_2)
        opt = optim.Adam(model.parameters(),
                        lr=args.lr,
                        betas=betas,
                        eps=args.eps,
                        weight_decay=args.weight_decay)

    elif args.opt == 'sgd':
        opt = optim.SGD(model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer")

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, _ = train(train_loader, model, opt, epoch)
        _, test_acc = test(test_loader, model, epoch)

        # print log
        print('Epoch [{}/{}], loss: {loss:.4f}, acc: {acc:.4f}'\
            .format(epoch+1, args.epochs, loss=train_loss, acc=test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "state_dict": model.state_dict(),
                "acc": best_acc,
                "epoch": epoch,
            }, args.prefix + "_best.pth")

        torch.save({
            "state_dict": model.state_dict(),
            "acc": test_acc,
            "epoch": epoch,
        }, args.prefix + "_checkpoint.pth")

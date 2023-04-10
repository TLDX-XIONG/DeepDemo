import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from Net import Classifier
import argparse


def pre_process(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize()
    ])

    train_dataset = datasets.MNIST(config['dataset_path'], train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(config['dataset_path'], train=False,download=True, transform=transform)

    # visulaization
    _, axes = plt.subplots(nrows=3, ncols=4, figsize=(8,6))
    axes = axes.flatten()
    for i in range(12):
        x, y = train_dataset[random.randint(0, len(train_dataset)-1)]
        axes[i].imshow(x.squeeze(), cmap='gray')
        axes[i].set_title(f'Labels: {y}')
        axes[i].set_axis_off()
    plt.savefig('preview_dataset/example.png', dpi=200)
    # plt.show()

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    return train_loader, test_loader, len(train_dataset), len(test_dataset)



def trainer(config, model, train_loader, test_loader, train_len, test_len):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    epochs = config['epochs']
    best_acc = 0.0
    start = 0
    if config['resume']:
        checkpoint = torch.load(config['last_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epochs = checkpoint['epochs']
        start = checkpoint['epoch']
        print(f'[Resume] Resume from Epoch {start}/{epochs} with Learning Rate {scheduler.get_last_lr()}')
    for epoch in range(start, epochs):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, dim=1)
            train_acc += (train_pred.detach() == y.detach()).sum().item()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                _, test_pred = torch.max(outputs, dim=1)
                test_acc += (test_pred.detach() == y.detach()).sum().item()
                test_loss += loss.item()
        print(f'[{epoch+1:03d}/{epochs:03d}] Train Acc: {train_acc/train_len:3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {test_acc/test_len:3.5f} loss: {test_loss/len(test_loader):3.5f}')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config['best_path'])
            torch.save({
                'epoch': epoch+1,
                'epochs': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, config['last_path'])
            print(f'saving model with acc {best_acc/test_len:.5f}')



if __name__ == '__main__':
    # config = {
    # 'batch_size': 256,
    # 'epochs': 100,
    # 'lr': 1e-4,
    # 'best_path': './ckpt/best.pth',
    # 'last_path': './ckpt/last.pth',
    # 'dataset_path': './dataset',
    # 'resume': True
    # }

    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size', type=int, default=256, help='batch size')
    parse.add_argument('--epochs', type=int, default=100, help='total epochs')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parse.add_argument('--best_path', type=str, default='./ckpt/best.pth', help='best model path')
    parse.add_argument('--last_path', type=str, default='./ckpt/last.pth', help='last mdoel path for resume')
    parse.add_argument('--dataset_path', type=str, default='dataset', help='path for dataset')
    parse.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    config = parse.parse_args().__dict__
    train_loader, test_loader, train_len, test_len = pre_process(config)
    model = Classifier()
    trainer(config, model, train_loader, test_loader, train_len, test_len)

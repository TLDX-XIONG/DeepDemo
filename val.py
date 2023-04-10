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

    test_dataset = datasets.MNIST(config['dataset_path'], train=False,download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    return test_loader, len(test_dataset)


def main(config, model, test_loader, test_len):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    test_loss = 0.0
    test_acc = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader):
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                _, test_pred = torch.max(outputs, dim=1)
                test_acc += (test_pred.detach() == y.detach()).sum().item()
                test_loss += loss.item()
        print(f'Val Acc: {test_acc/test_len:3.5f} loss: {test_loss/len(test_loader):3.5f}')
    
    _, axes = plt.subplots(nrows=3, ncols=4, figsize=(8,6))
    axes = axes.flatten()
    for i in range(12):
        index = random.randint(0, test_len)
        x, _ = test_loader.dataset[i]
        x = x.view(-1, *x.shape).to(device)
        outputs = model(x)
        pred = torch.argmax(outputs).cpu()
        axes[i].imshow(x.cpu().squeeze(), cmap='gray')
        axes[i].set_title(f'Pred: {pred}')
        axes[i].set_axis_off()
    plt.savefig('./pred_img/pred.png', dpi=200)
    # plt.show()





if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size', type=int, default=256, help='batch size')
    parse.add_argument('--weight_path', type=str, default='./ckpt/best.pth', help='weight path')
    parse.add_argument('--dataset_path', type=str, default='./dataset', help='path for dataset')
    config = parse.parse_args().__dict__
    test_loader, test_len = pre_process(config)
    model = Classifier()
    model.load_state_dict(torch.load(config['weight_path']))
    main(config, model, test_loader, test_len)
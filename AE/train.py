from __future__ import print_function
import AE_model
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch
import random
import numpy as np
import json
import playlist_batchmaker as pb
import split_data
import os

input_dim = 31202
noise_p = 0.5
extract_num = 100
aug_step = 1 #얼마가 최적일까?
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')
model_PATH = os.path.join(data_path, './AE_weight.pth')
batch_size = 256
epochs = 100
log_interval = 100
validation_ratio = 0.01
test_ratio = 0.01
random_seed = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dp = nn.Dropout(p=noise_p)
model = AE_model.AutoEncoder(layer_sizes = ((input_dim,500,500,500,1000),(1000,500,500,500,input_dim)), is_constrained=True, symmetric=True, dp_drop_prob=0.8).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-9)#l2_weight 추가
train_loader, valid_loader, test_loader = split_data.splited_loader(batch_size=batch_size, random_seed=random_seed, test_ratio=test_ratio, validation_ratio=validation_ratio)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    return BCE

def train(epoch, is_load = True):#Kakao AE
    if is_load:
        model.load_state_dict(torch.load(model_PATH))
    model.train()
    train_loss = 0
    for idx,data in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch = model(data['input_one_hot'])
        loss = loss_function(recon_batch, data['target_one_hot'])
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if aug_step > 0:
            # Dense refeed
            for _ in range(aug_step):
                inputs = recon_batch.detach()
                if noise_p > 0.0:
                    noised_inputs = dp(inputs)
                optimizer.zero_grad()
                recon_batch = model(noised_inputs)
                loss = loss_function(recon_batch, inputs)
                loss.backward()
                optimizer.step()

        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx, len(train_loader),
                100. * idx/len(train_loader),
                loss.item() / len(data)))   

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / batch_size))
    torch.save(model.state_dict(), model_PATH)

def test_accuracy():
  model.load_state_dict(torch.load(model_PATH))
  model.eval()
  with torch.no_grad():
    total_lostsong = 0
    correct = 0
    for data in test_loader:
        noise_img = data['input_one_hot']
        img = data['target_one_hot']
        output = model(noise_img)
        _, indices = torch.topk(output, extract_num, dim = 1)

        diff = img - noise_img

        total_lostsong += torch.sum(diff.view(-1))
        one_hot = torch.zeros(indices.size(0), input_dim).to(device)
        one_hot = one_hot.scatter(1, indices.cuda().data, 1)

        one_hot_filter = one_hot * diff
        correct += torch.sum(one_hot_filter.view(-1))
    print('accuracy: {}(%)'.format(correct / total_lostsong*100))

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch = epoch, is_load=False)
        test_accuracy() 
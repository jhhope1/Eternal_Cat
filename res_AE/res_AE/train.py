from __future__ import print_function
import res_AE_model
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau as lr_sc
from torch.nn import functional as F
import numpy as np
import json
import split_data
import os
import time


batch_size = 512
random_seed = 10
test_ratio = 0.01
noise_p = 0.5
train_loader, test_loader = split_data.splited_loader(batch_size=batch_size, random_seed=random_seed, test_ratio=test_ratio, noise_p = noise_p)

input_dim = 38459
output_dim = 20517
song_size = 12538
extract_num = 100
extract_song = 100
extract_tag = 10
aug_step = 0 #blobfusad
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')
model_PATH = os.path.join(data_path, './res_AE_weight.pth')
epochs = 100
log_interval = 10
learning_rate = 1e-3
D_ = 300

weight_decay = 0
layer_sizes = (input_dim,D_,D_,output_dim)
dropout_p = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes, dp_drop_prob = dropout_p, is_res=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)#l2_weight
scheduler = lr_sc(optimizer, mode = 'min', factor = 0.99, patience = 50, verbose = True)
dp = nn.Dropout(p=noise_p)

pos_weight = torch.tensor([-1.]).to(device)
neg_weight = torch.tensor([-0.1]).to(device)
def custom_loss_function(output, target):
    output = torch.clamp(output,min=1e-4,max=1-1e-4)
    loss =  pos_weight * (target * torch.log(output)) + neg_weight* ((1 - target) * torch.log(1 - output))
    return torch.sum(loss)

def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return BCE

def train(epoch, is_load = True):#Kakao AE
    if is_load and epoch == 1:
        model.load_state_dict(torch.load(model_PATH))
    model.train()
    train_loss = 0
    delation_load = 0
    delation_back = 0
    tot_st = time.time()
    for idx, (input_data, output_data) in enumerate(train_loader):
        optimizer.zero_grad()

        back_st = time.time()

        recon_batch = model(input_data)
        loss = custom_loss_function(recon_batch, output_data)

        loss.backward()
        train_loss += loss.item()
        scheduler.step(loss)
        optimizer.step()
        
        delation_back += time.time() - back_st
        
        if torch.isnan(loss):
            print("loss is nan!")
            return None

        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime elapsed: {:.6f}'.format(
                epoch, idx, len(train_loader),
                100. * idx/len(train_loader),
                loss.item() / input_data.shape[0], time.time() - tot_st))

    delation_load += time.time() - tot_st
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader)))
    torch.save(model.state_dict(), model_PATH)
    print('Backprop: ',delation_back, '(s), Total: ',delation_load,'(s)')
'''
def test_accuracy():
    model.load_state_dict(torch.load(model_PATH))
    model.eval()
    with torch.no_grad():
        total_lost = 0
        total_lostsong = 0
        total_losttag = 0
        correct = 0
        correct_song = 0
        correct_tag = 0
        for (input_data, output_data) in test_loader:
            noise_img = input_data
            img = output_data
            output = model(noise_img) #is this right?
            _, indices_tag = torch.topk(output.narrow(1,0,song_size), extract_song, dim = 1)
            _, indices_song = torch.topk(output.narrow(1,song_size,output_dim-song_size), extract_tag, dim = 1) 
            indices_song += torch.tensor(song_size).long()
            indices = torch.cat((indices_song, indices_tag) , dim = 1)

            diff = img - noise_img.narrow(1,0,output_dim)

            total_lost += torch.sum(diff.reshape(-1))
            total_lostsong += torch.sum(diff.narrow(1,0,song_size).reshape(-1))
            total_losttag += torch.sum(diff.narrow(1,song_size,output_dim-song_size).reshape(-1))

            one_hot = torch.zeros(indices_song.size(0), output_dim).to(device)
            one_hot = one_hot.scatter(1, indices.to(device).data, 1)

            one_hot_filter = one_hot * diff.to(device)
            correct_song += torch.sum(one_hot_filter.narrow(1,0,song_size).reshape(-1))
            correct_tag += torch.sum(one_hot_filter.narrow(1,song_size,output_dim-song_size).reshape(-1))
            correct += torch.sum(one_hot_filter.reshape(-1))

        accuracy = None
        if total_lostsong > 0:
            accuracy = correct / total_lost * 100.
            tag_accuracy = correct_tag / total_losttag * 100.
            song_accuracy = correct_song / total_lostsong * 100.
        print('::ACCURACY:: of Net \naccuracy: {}(%)\nsong_accuracy: {}(%)\ntag_accuracy: {}(%)'.format( accuracy, song_accuracy, tag_accuracy))'''
def test_accuracy():
    model.load_state_dict(torch.load(model_PATH))
    model.eval()
    with torch.no_grad():
        total_lost = 0
        total_lostsong = 0
        total_losttag = 0
        correct = 0
        correct_song = 0
        correct_tag = 0
        for data in test_loader:
            noise_img = data['meta_input_one_hot']
            img = data['target_one_hot']
            output = model(noise_img.to(device)) #is this right?
            _, indices_tag = torch.topk(output.narrow(1,0,song_size), extract_song, dim = 1)
            _, indices_song = torch.topk(output.narrow(1,song_size,output_dim-song_size), extract_tag, dim = 1) 
            indices_song += torch.tensor(song_size).long()
            indices = torch.cat((indices_song, indices_tag) , dim = 1)

            diff = img - noise_img.narrow(1,0,output_dim)

            total_lost += torch.sum(diff.reshape(-1))
            total_lostsong += torch.sum(diff.narrow(1,0,song_size).reshape(-1))
            total_losttag += torch.sum(diff.narrow(1,song_size,output_dim-song_size).reshape(-1))

            one_hot = torch.zeros(indices_song.size(0), output_dim).to(device)
            one_hot = one_hot.scatter(1, indices.to(device).data, 1)

            one_hot_filter = one_hot * diff.to(device)
            correct_song += torch.sum(one_hot_filter.narrow(1,0,song_size).reshape(-1))
            correct_tag += torch.sum(one_hot_filter.narrow(1,song_size,output_dim-song_size).reshape(-1))
            correct += torch.sum(one_hot_filter.reshape(-1))

        accuracy = None
        if total_lostsong > 0:
            accuracy = correct / total_lost * 100.
            tag_accuracy = correct_tag / total_losttag * 100.
            song_accuracy = correct_song / total_lostsong * 100.
        print('::ACCURACY:: of Net \naccuracy: {}(%)\nsong_accuracy: {}(%)\ntag_accuracy: {}(%)'.format( accuracy, song_accuracy, tag_accuracy))

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch = epoch, is_load= False)
        test_accuracy()
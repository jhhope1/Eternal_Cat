from __future__ import print_function
import res_AE_model
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import json
import split_data
import os
import time
from const import *
train_loader, valid_loader, test_loader = split_data.splited_loader(batch_size=batch_size, random_seed=random_seed, test_ratio=test_ratio, validation_ratio=validation_ratio)

model = res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes, dp_drop_prob = dropout_p, is_res=is_res).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)#l2_weight
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3 , mode = 'min',factor = 0.7, verbose = True)


def custom_loss_function(output, target):
    output = torch.clamp(output,min=1e-4,max=1-1e-4)
    loss =  pos_weight * (target * torch.log(output)) + neg_weight* ((1 - target) * torch.log(1 - output))
    return torch.mean(loss)
def custom_song_loss_function(output, target):
    output = output.narrow(1,0,song_size)
    target = target.narrow(1,0,song_size)
    output = torch.clamp(output,min=1e-4,max=1-1e-4)
    loss =  pos_weight * (target * torch.log(output)) + neg_weight* ((1 - target) * torch.log(1 - output))
    return torch.mean(loss)
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    return BCE

def train(epoch, is_load = True):#Kakao AE
    if is_load and epoch == 1:
        model.load_state_dict(torch.load(model_PATH))
    model.train()
    train_loss = 0
    st = time.time()
    for idx,data in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch = model(data['meta_input_one_hot'].to(device))
        loss = custom_song_loss_function(recon_batch, data['target_one_hot'].to(device))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        '''
        if aug_step > 0:
            # Dense refeed
            for _ in range(aug_step):
                noised_inputs = recon_batch.detach()
                if noise_p > 0.0:
                    noised_inputs = dp(noised_inputs)
                meta_noised_inputs = torch.cat([noised_inputs,data['meta_input_one_hot'].narrow(1,0,input_dim-output_dim)], dim = 1)
                optimizer.zero_grad()
                recon_batch = model(meta_noised_inputs.to(device))
                loss = loss_function(recon_batch, noised_inputs)
                loss.backward()
                optimizer.step()
        '''
        if torch.isnan(loss):
            print("loss is nan!")
            return None
        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx, len(train_loader),
                100. * idx/len(train_loader),
                loss.item() / len(data))) 
    scheduler.step(loss)
    ed = time.time()
    print("time = ", ed-st)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader)))
    torch.save(model.state_dict(), model_PATH)

def train_accuracy():
    model.eval()
    with torch.no_grad():
        total_lost = 0
        total_lostsong = 0
        total_losttag = 0
        correct = 0
        correct_song = 0
        correct_tag = 0
        for idx, data in enumerate(train_loader):
            if idx==10:
                break
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
        print('::train_ACCURACY:: of Net \naccuracy: {}(%)\nsong_accuracy: {}(%)\ntag_accuracy: {}(%)'.format( accuracy, song_accuracy, tag_accuracy))
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
        for idx, data in enumerate(test_loader):
            if idx==10:
                break
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
        print('::test_ACCURACY:: of Net \naccuracy: {}(%)\nsong_accuracy: {}(%)\ntag_accuracy: {}(%)'.format( accuracy, song_accuracy, tag_accuracy))

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch = epoch, is_load=False)
        train_accuracy()
        test_accuracy()
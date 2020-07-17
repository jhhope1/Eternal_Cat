from __future__ import print_function
import res_AE_model
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import json
import split_data_4_NN as split_data
import os

batch_size = 256
random_seed = 10
validation_ratio = 0.01
test_ratio = 0.01

input_dim = 101252
output_dim = 57229
song_size = 53921
noise_p = 0.5
extract_song = 100
extract_tag = 10
aug_step = 0 #blobfusad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

epochs = 100
log_interval = 100
learning_rate = 1e-3
weight_decay = 0
D_ = 300
dropout_p = 0.0

#train type of nn
type_nn = ['title_tag', 'song_meta'] #['song_meta_tag', 'title']
model_PATH = {name: os.path.join(data_path, 'res_AE_' + name) + '_weight.pth' for name in type_nn}
input_dim = {'title': 1000, 'title_tag': 4308, 'song_meta_tag': 100252, 'song_meta': 96944}
layer_sizes = {name: (input_dim[name],D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,output_dim) for name in type_nn}

train_loader = {}
valid_loader = {}
test_loader = {}
for id_nn in type_nn:
    train_loader[id_nn], valid_loader[id_nn], test_loader[id_nn] = split_data.splited_loader(id_nn = id_nn, batch_size=batch_size, random_seed=random_seed, test_ratio=test_ratio, validation_ratio=validation_ratio)

model = {name: res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes[name], dp_drop_prob = dropout_p, is_res=True).cuda() for name in type_nn}
model = {name: nn.DataParallel(model[name].cuda()) for name in type_nn}
optimizer = {name: optim.Adam(model[name].parameters(), lr=learning_rate, weight_decay=weight_decay) for name in type_nn}

dp = nn.Dropout(p=noise_p)

def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x.narrow(1,0,song_size), x.narrow(1,0,song_size), reduction='mean')
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    return BCE

def train(epoch, id_nn, is_load = True):#Kakao AE
    if is_load:
        model[id_nn].load_state_dict(torch.load(model_PATH[id_nn]))
    train_loss = 0
    for idx, data in enumerate(train_loader[id_nn]):
        model[id_nn].train()
        optimizer[id_nn].zero_grad()

        recon_batch = model[id_nn](data['meta_input_one_hot_' + id_nn].cuda()) #need to be modified
        loss = loss_function(recon_batch, data['target_one_hot'].cuda()) #you too
        loss.backward()
        train_loss += loss.item()
        optimizer[id_nn].step()

        if aug_step > 0:
            for _ in range(aug_step):
                noised_inputs = recon_batch.detach()
                if noise_p > 0.0:
                    noised_inputs = dp(noised_inputs)
                meta_noised_inputs = torch.cat([noised_inputs,data['meta_input_one_hot_' + id_nn].narrow(1,0,input_dim-output_dim)], dim = 1)
                optimizer[id_nn].zero_grad()
                recon_batch = model(meta_noised_inputs.cuda())
                loss = loss_function(recon_batch, noised_inputs.cuda())
                loss.backward()
                optimizer[id_nn].step()
        
        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)], Net={}\tLoss: {:.6f}'.format(
                epoch, idx, len(train_loader[id_nn]),
                100. * idx/len(train_loader[id_nn]),
                id_nn,
                loss.item() / len(data))) 
        ##
    print('====> Epoch: {}, Net={}, Average loss: {:.4f}'.format(
        epoch, id_nn, train_loss / len(train_loader[id_nn])))
    torch.save(model[id_nn].state_dict(), model_PATH[id_nn])

def test_accuracy(id_nn):
    model[id_nn].load_state_dict(torch.load(model_PATH[id_nn]))
    model[id_nn].eval()
    with torch.no_grad():
        total_lost = 0
        total_lostsong = 0
        total_losttag = 0
        correct = 0
        correct_song = 0
        correct_tag = 0
        for data in test_loader[id_nn]:
            noise_img = data['meta_input_one_hot_' + id_nn]
            img = data['target_one_hot']
            output = model[id_nn](noise_img.cuda()) #is this right?
            _, indices_tag = torch.topk(output.narrow(1,0,song_size), extract_song, dim = 1)
            _, indices_song = torch.topk(output.narrow(1,song_size,output_dim-song_size), extract_tag, dim = 1) 
            indices_song += torch.tensor(song_size).long()
            indices = torch.cat((indices_song, indices_tag) , dim = 1)
            diff = img
            if 'song' in id_nn:
                noise_input_extended = torch.zeros_like(img)
                colnum = data['noise_song_one_hot'].shape[1]
                noise_input_extended[:,:colnum] = data['noise_song_one_hot']
                diff -= noise_input_extended

            if 'tag' in id_nn:
                noise_input_extended = torch.zeros_like(img)
                col_st = img.shape[1] - data['noise_tag_one_hot'].shape[1]
                noise_input_extended[:, col_st:] = data['noise_tag_one_hot']
                diff -= noise_input_extended

            total_lost += torch.sum(diff.reshape(-1))
            total_lostsong += torch.sum(diff.narrow(1,0,song_size).reshape(-1))
            total_losttag += torch.sum(diff.narrow(1,song_size,output_dim-song_size).reshape(-1))

            one_hot = torch.zeros(indices_song.size(0), output_dim).cuda()
            one_hot = one_hot.scatter(1, indices.cuda().data, 1)

            one_hot_filter = one_hot * diff
            correct_song += torch.sum(one_hot_filter.narrow(1,0,song_size).reshape(-1))
            correct_tag += torch.sum(one_hot_filter.narrow(1,song_size,output_dim-song_size).reshape(-1))
            correct += torch.sum(one_hot_filter.reshape(-1))

        accuracy = None
        if total_lostsong > 0:
            accuracy = correct / total_lost * 100.
            tag_accuracy = correct_tag / total_losttag * 100.
            song_accuracy = correct_song / total_lostsong * 100.
        print('::ACCURACY:: of Net={} \naccuracy: {}(%)\nsong_accuracy: {}(%)\ntag_accuracy: {}(%)'.format(id_nn, accuracy, song_accuracy, tag_accuracy))

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        for id_nn in type_nn:
            train(epoch = epoch, id_nn = id_nn , is_load=True)
            test_accuracy(id_nn)
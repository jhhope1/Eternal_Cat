from __future__ import print_function
import res_AE_model
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import json
import split_data
import os

batch_size = 512
random_seed = 10
validation_ratio = 0.01
test_ratio = 0.01
noise_p = 0.5
train_loader, valid_loader, test_loader = split_data.splited_loader(batch_size=batch_size, random_seed=random_seed, test_ratio=test_ratio, validation_ratio=validation_ratio, noise_p = noise_p)

input_dim = 34139
output_dim = 20517
song_size = 12538
extract_num = 100
extract_song = 100
extract_tag = 10
aug_step = 0 #blobfusad
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')
model_PATH = os.path.join(data_path, './res_AE_weight.pth')
epochs = 1000
log_interval = 100
learning_rate = 5e-3
D_ = 1000

weight_decay = 0
layer_sizes = (input_dim,D_,D_,output_dim)
dropout_p = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes, dp_drop_prob = dropout_p, is_res=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)#l2_weight
steps = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

dp = nn.Dropout(p=noise_p)

pos_weight = torch.tensor([-1.]).to(device)
neg_weight = torch.tensor([-0.1]).to(device)
def custom_loss_function(output, target):
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
    for idx,data in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch = model(data['meta_input_one_hot'].to(device))
        loss = custom_loss_function(recon_batch, data['target_one_hot'].to(device))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if torch.isnan(loss) or loss.item() > 5:
            print("loss is nan!")
            return None

        if idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx, len(train_loader),
                100. * idx/len(train_loader),
                loss.item() / len(data)))   
    scheduler.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader)))
    torch.save(model.state_dict(), model_PATH)

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
        train(epoch = epoch, is_load=False)
        test_accuracy()
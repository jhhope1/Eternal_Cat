import torch
from torch.utils.data import DataLoader as dl
import sklearn.metrics
import torch.optim as optim
import torch.nn.functional as F
from dataloader import Dataset_train
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import torch.nn as nn
from model import gmf
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#device = "cpu"


epochs = 100
song_size = 53921
item_size = 57229
batch_size = 256
dl_params1 = {'batch_size': batch_size}
dl_params2 = {'batch_size': 256}
training_set = Dataset_train()
training_gen = dl(training_set, **dl_params1, sampler = SubsetRandomSampler(range(113000)))
test_gen = dl(training_set, **dl_params2, sampler = SubsetRandomSampler(range(113000,115071)))



model = gmf(emb_dim = 1024)
model.to(device)

def loss_function(recon_x, x):
    x = x.to(device)
    recon_x = recon_x.to(device)
    #loss_ = nn.MarginRankingLoss(margin=1.0, reduction = 'mean').forward(torch.sigmoid(x), x, torch.ones(item_size).to(device))
    loss_ = F.binary_cross_entropy(torch.sigmoid(recon_x), x, reduction='mean')
    return loss_

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-10)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.4, patience=15, verbose=True)

def F1(yhat, y):
    yhat = (torch.sigmoid(yhat) > 0.3).tolist()    
    y = (y > 0).tolist()
    return sklearn.metrics.f1_score(y, yhat, average = None)



for epoch in range(epochs):
    epoch_loss = 0
    torch.cuda.empty_cache()
    model.to(device)
    model.train()
    cnt = 0
    ones = torch.ones(item_size)
    for pairs, labels in training_gen:
        #print("bach!")
        #print(pairs)
        #print(pairs[:, 1])
        #print(pairs)
        optimizer.zero_grad()
        #print(labels)
        
        predicted_labels = model(pairs.to(device))
        #print(predicted_labels.squeeze().shape, labels.shape)
        #print(predicted_labels.squeeze(), labels)
        loss = loss_function(predicted_labels.squeeze().to(device),labels.to(device))
        loss.backward()
        optimizer.step()
        #print(loss.item())

        epoch_loss += loss.item()
        
        cnt += 1
        if cnt%50 == 0:
            print(loss.item())

    #if cnt % 100 == 0:
    model.eval()
    correct = 0
    total_lostsong = 0
    for pairs, img in test_gen:
        
        output = model(pairs.to(device))
        _, indices = torch.topk(output[:,:song_size], 100, dim = 1)
        total_lostsong += torch.sum(img.view(-1)).to(device)
        diff = img[:,:song_size].to(device)
        one_hot = torch.zeros(indices.size(0), song_size).to(device)
        one_hot = one_hot.scatter(1, indices.cuda().data, 1).to(device)

        one_hot_filter = one_hot * diff
        correct += torch.sum(one_hot_filter.view(-1))

    print('accuracy: {}(%)'.format(correct / total_lostsong))
    print(model.word_embeddings.weight.sum())
    model.train()
            

    print(epoch_loss)
    

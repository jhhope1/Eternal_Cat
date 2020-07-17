import torch
from torch.utils.data import DataLoader as dl
import sklearn.metrics
import torch.optim as optim
from dataloader import Dataset_train, Dataset_test
import torch.nn as nn
from model import gmf
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"


epochs = 100
item_size = 31202
batch_size = 512
dl_params = {'batch_size': batch_size, 'shuffle': True}
training_set = Dataset_train()
training_gen = dl(training_set, **dl_params)



model = gmf(emb_dim = 64)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 5*1e-3)

def F1(yhat, y):
    yhat = (torch.sigmoid(yhat) > 0.3).tolist()
    
    y = (y > 0).tolist()
    return sklearn.metrics.f1_score(y, yhat)


for epoch in range(epochs):
    epoch_loss = 0
    model.train()
    cnt = 0
    
    for pairs, labels in training_gen:
        #print("bach!")
        #print(pairs)
        #print(pairs[:, 1])
        optimizer.zero_grad()
        pairs = pairs.to(device)
        labels = labels.float().to(device)
        #print(labels)
        predicted_labels = model(pairs)
        #print(predicted_labels.squeeze().shape, labels.shape)
        #print(predicted_labels.squeeze(), labels)
        loss = criterion(predicted_labels.squeeze(), labels)
        loss.backward()
        optimizer.step()
        

        epoch_loss += loss.item()

        cnt += 1
        if cnt % 500 == 0:
            print(f'{cnt}% is done, loss : {loss.item()}')
            f1 = F1(predicted_labels, labels)
            #print(f'F1 ---> ::: epoch {epoch}: {f1}')

    print(epoch_loss)
    
        

    if epoch%1 == 0:
        model.eval()
        with open("playlist_train_idxs.json", 'r') as f1:
            playlists = json.load(f1)
            acc = 0
            for idx, playlist in enumerate(playlists):
                pairs = torch.cat([torch.ones(item_size).float().unsqueeze(1)*idx, torch.arange(0, item_size).float().unsqueeze(1)], dim = 1).long().to(device)
                #print(pairs)
                #pairs = torch.transpose(pairs, 0, 1)
                #print(pairs)
                predicted_labels = model(pairs.long()).squeeze()
                #print(predicted_labels)
                predicted_labels = torch.topk(predicted_labels, 100).indices.tolist()
                #print(predicted_labels)
                #break
                #f1 = F1(predicted_labels, labels)
                
                #print(f'F1 ---> ::: epoch {epoch}: {f1}')
                corr = 0
                for ind in predicted_labels:
                    if ind in playlist:
                        corr+= 1
                
                acc += corr/(len(playlist) + 1)
                if idx > 1000:
                    print(playlist)
                    predicted_labels = model(pairs.long()).squeeze()
                    print(predicted_labels)
                    print(torch.topk(predicted_labels, 100).indices.tolist())
                    print(f'avg acc : {acc/(idx + 1)}')
                    break

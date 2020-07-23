import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
song_size = 29296
item_size = 31202
item_size = 40000
tag_size = item_size - song_size
song_avg = 15
tag_avg = 3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

playlists_size = 115071


class gmf(nn.Module):
    def __init__(self, emb_dim, load_model = True):
        super(gmf, self).__init__()
        self.emb_dim = emb_dim
        
        self.playlist_embeddings = nn.Embedding(playlists_size, emb_dim).to(device)
        self.playlist_embeddings.weight.requires_grad = True
        nn.init.xavier_normal_(self.playlist_embeddings.weight)
        if load_model:
            self.playlist_embeddings.weight = torch.load('playlist_emb.dat')
        
        self.item_embeddings = nn.Embedding(item_size, emb_dim).to(device)
        self.item_embeddings.weight.requires_grad = True
        nn.init.xavier_normal_(self.item_embeddings.weight)
        if load_model:
            self.item_embeddings.weight = torch.load('item_emb.dat')
        
        self.fc1 = nn.Linear(emb_dim, emb_dim//2)
        self.fc2 = nn.Linear(emb_dim//2, 1)
        #self.batchnorm = nn.BatchNorm1d(emb_dim)
        
    
    def forward(self, x):
        batch_playlist_emb = self.playlist_embeddings(x[:,0])
        #print(batch_playlist_emb)
        batch_item_emb = self.item_embeddings(x[:,1])
        #print(batch_playlist_emb.shape, batch_item_emb.shape)
        x = torch.mul(batch_playlist_emb, batch_item_emb)
        y = torch.bmm(batch_playlist_emb.view(-1,1,self.emb_dim), batch_item_emb.view(-1,self.emb_dim,1)).squeeze(dim = 2)
        #print(y.shape)
        #print(x) 
        #x = self.batchnorm(x)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        #print(x.shape, y.shape)
        return y

        

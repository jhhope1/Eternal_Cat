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
playlists_size = 150000
class gmf(nn.Module):
    def __init__(self, emb_dim):
        super(gmf, self).__init__()
        self.playlist_embeddings =torch.empty(playlists_size, emb_dim).to(device)
        nn.init.xavier_uniform_(self.playlist_embeddings)
        self.item_embeddings = torch.empty(item_size, emb_dim).to(device)
        nn.init.xavier_uniform_(self.item_embeddings)
        self.fc1 = nn.Linear(emb_dim, emb_dim//2)
        self.fc2 = nn.Linear(emb_dim//2, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
        self.fc1.bias.data.fill_(0.)
        self.fc2.bias.data.fill_(0.)
        self.batchnorm = nn.BatchNorm1d(emb_dim)
        
    
    def forward(self, x):
        batch_playlist_emb = self.playlist_embeddings[x[:,0]]
        batch_item_emb = self.item_embeddings[x[:,1]]
        x = torch.mul(batch_playlist_emb, batch_item_emb)
        x = self.batchnorm(x)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
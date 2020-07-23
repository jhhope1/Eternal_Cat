import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
song_size = 29296
item_size = 31202
item_size = 57229 #song : 15, tag : 7
tag_size = item_size - song_size
song_avg = 15
tag_avg = 3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

word_size = 4979 + 1


class gmf(nn.Module):
    def __init__(self, emb_dim):
        super(gmf, self).__init__()
        
        self.layer_1 = nn.Linear(emb_dim,300)
        self.layer_2 = nn.Linear(300, 300)
        self.layer_30 = nn.Linear(300,300)
        self.layer_31 = nn.Linear(300,300)
        self.layer_32 = nn.Linear(300,300)
        self.layer_4 = nn.Linear(300, item_size)
        self.bn_1 = nn.BatchNorm1d(300)
        self.bn_2 = nn.BatchNorm1d(300)
        self.bn_30 = nn.BatchNorm1d(300)
        self.bn_31 = nn.BatchNorm1d(300)
        self.bn_32 = nn.BatchNorm1d(300)

        nn.init.xavier_normal(self.layer_1.weight)
        nn.init.xavier_normal(self.layer_4.weight)
        nn.init.normal(self.layer_1.bias)
        nn.init.normal(self.layer_4.bias)
        self.word_embeddings = nn.Embedding(word_size, emb_dim, padding_idx = word_size - 1)

        #self.word_embeddings.weight.data.xavier_uniform_(gain = 1.0)
        self.word_embeddings.weight.requires_grad = True
        nn.init.xavier_normal_(self.word_embeddings.weight)
        #self.word_embeddings.to(device)

        #self.song_embeddings = nn.Embedding(song_size + 1, emb_dim, padding_idx = song_size)
        
    
    def forward(self, x):
        x = self.word_embeddings(x).to(device).sum(dim = 1).squeeze()
        
        #print(x)
       
        #print(batch_item_emb)

        #print(x)
        #x = self.batchnorm(x)
        x = self.layer_1(x)
        x = self.bn_1(x)
        x = F.selu(x)
        x_1 = x

        x = self.layer_2(x)
        x = self.bn_2(x)
        x = F.selu(x)

        x = self.layer_30(x + x_1)
        x = self.bn_30(x)
        x = F.selu(x)
        x_2 = x

        x = self.layer_31(x)
        x = self.bn_31(x)
        x = F.selu(x)

        x = self.layer_32(x + x_2)
        x = self.bn_32(x)
        x = F.selu(x)

        x = x + x_1
        x = self.layer_4(x)
        
        return x

        

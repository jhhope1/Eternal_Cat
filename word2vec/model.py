import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
song_size = 29296
item_size = 31202
item_size = 31202
tag_size = item_size - song_size
song_avg = 15
tag_avg = 3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

word_size = 5099 + 1


class gmf(nn.Module):
    def __init__(self, emb_dim):
        super(gmf, self).__init__()
        
        self.layer_1 = nn.Linear(emb_dim, 20)
        self.layer_2 = nn.Linear(100, 100)
        self.layer_3 = nn.Linear(100,20)
        self.layer_4 = nn.Linear(20, item_size)
        self.batchnorm = nn.BatchNorm1d(emb_dim)
        nn.init.xavier_normal(self.layer_1.weight)
        nn.init.xavier_normal(self.layer_4.weight)
        nn.init.normal(self.layer_1.bias)
        nn.init.normal(self.layer_4.bias)
        self.word_embeddings = nn.Embedding(word_size, emb_dim)
        self.word_embeddings.weight.requires_grad = True
        
        self.word_embeddings.to(device)
        
    
    def forward(self, x):
        x = F.normalize(self.word_embeddings(x).to(device), p = 2, dim = 1).mean(dim = 1)
        #print(x)
       
        #print(batch_item_emb)

        #print(x)
        #x = self.batchnorm(x)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_4(x)
        
        return x

        

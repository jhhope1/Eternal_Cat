import torch
import numpy as np
import json
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

song_size = 29296
item_size = 31202
tag_size = item_size - song_size
song_avg = 30
tag_avg = 5
emb_dim = 8
word_size = 5099 + 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

class Dataset_train(torch.utils.data.Dataset):
    def __init__(self):
        self.items = []
        self.words = []
        with open("playlist_idxs.json", 'r') as f1:
            playlists = json.load(f1)
            for idx, playlist in enumerate(playlists):
                while len(playlist['wordset_idx']) < 50:
                    playlist['wordset_idx'].append(word_size- 1)
                self.words.append(torch.tensor(playlist['wordset_idx']))
                self.items.append(torch.tensor(playlist['items']).long())
                
        self.len = len(self.items)
        self.ones = torch.zeros(item_size)
        #words_ = [_word.to(device) for _word in self.words]
        

        #print(self.uipairs.shape, self.labels.shape)
    
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.words[index], torch.zeros(item_size).scatter_(0, self.items[index], 1)




        
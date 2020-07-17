import torch
import numpy as np
import json
import random

song_size = 29296
item_size = 31202
tag_size = item_size - song_size
song_avg = 30
tag_avg = 5

class Dataset_train(torch.utils.data.Dataset):
    def __init__(self):
        uipairs = []
        labels = []
        with open("playlist_train_idxs.json", 'r') as f1:
            playlists = json.load(f1)
            for idx, playlist in enumerate(playlists):
                
                loc_exi = set()
                for item in playlist:
                    uipairs.append((idx,item))
                    labels.append(1)
                    loc_exi.add(item)
                for item in range(song_avg):
                    rand = random.randint(0,song_size)
                    if rand not in loc_exi:
                        uipairs.append((idx, rand))
                        labels.append(0)
                for item in range(tag_avg):
                    rand = random.randint(song_size,item_size)
                    if rand not in loc_exi:
                        uipairs.append((idx, rand))
                        labels.append(0)
                if idx > 5000:
                    break
        self.uipairs = torch.tensor(uipairs)
        self.labels = torch.tensor(labels)
        self.len = len(labels)
        print(self.uipairs.shape, self.labels.shape)
    
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.uipairs[index], self.labels[index]

class Dataset_test(torch.utils.data.Dataset):
    def __init__(self):
        uipairs = []
        labels = []
        with open("playlist_test_idxs.json", 'r') as f1:
            playlists = json.load(f1)
            for idx, playlist in enumerate(playlists):
                
                for item in playlist:
                    uipairs.append((idx,item))
                    labels.append(1)
                    
        self.uipairs = torch.tensor(uipairs)
        self.labels = torch.tensor(labels)
        self.len = len(labels)
    
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        return self.uipairs[index], self.labels[index]



        
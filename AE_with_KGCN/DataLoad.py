from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import random
# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

input_dim = 31202
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

#file load
taking_song = set()
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data\\')
with open(DATA+"song_to_idx.json", 'r',encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
    for song, idx in song_to_idx.items():
        taking_song.add(int(song))
with open(DATA+"song_meta.json", 'r', encoding='utf-8') as f2:
    song_meta = json.load(f2)
with open(DATA+"idx_to_item.json", 'r', encoding='utf-8') as f3:
    idx_to_item = json.load(f3)
with open(DATA+"entity_to_idx.json", 'r', encoding='utf-8') as f4:
    entity_to_idx = json.load(f4)
with open(DATA+"relation_to_idx.json", 'r', encoding='utf-8') as f5:
    relation_to_idx = json.load(f5)
kg = {}
for line in open(DATA+"kg.txt", 'r', encoding='utf-8'):
    array = line.strip().split(' ')
    head_old = array[0]
    relation_old = array[1]
    tail_old = array[2]
    if song_to_idx[str(head_old)] not in kg:
        kg[song_to_idx[str(head_old)]] = [(relation_to_idx[relation_old],entity_to_idx[tail_old])]
    else:
        kg[song_to_idx[str(head_old)]].append((relation_to_idx[relation_old],entity_to_idx[tail_old]))

class Noise_p(object):
    def __init__(self, noise_p):
        self.noise_p = noise_p

    def __call__(self, sample):
        input_one_hot = sample['input_one_hot']
        non_zero_indices = sample['non_zero_indices']
        input_non_zero_indices = np.random.choice(non_zero_indices, replace=False,
                           size=int(non_zero_indices.size * (1-self.noise_p)))
        input_one_hot = np.zeros_like(input_one_hot)
        if input_non_zero_indices.size != 0:
            input_one_hot[input_non_zero_indices] = 1
        return {'input_one_hot': input_one_hot, 'target_one_hot' : sample['target_one_hot'], 'input_non_zero_indices' : input_non_zero_indices, 'target_non_zero_indices' : non_zero_indices}

class Noise_uniform(object):
    def __call__(self, sample):
        input_one_hot = sample['input_one_hot']
        non_zero_indices = sample['non_zero_indices']
        zero_indices = np.random.choice(non_zero_indices, replace=False,
                           size=int(non_zero_indices.size * random.uniform(0,1)))
        if zero_indices.size != 0:
            input_one_hot[zero_indices] = 0
        return {'input_one_hot': input_one_hot, 'target_one_hot': sample['target_one_hot']}


class PlaylistDataset(Dataset):
    """Playlist target dataset."""

    def __init__(self, transform = Noise_p(0.5)):
        with open(os.path.join(data_path, "train.json"), 'r', encoding='utf-8') as f1:
            self.training_set = json.load(f1)

        self.song_to_idx = {}
        self.tag_to_idx = {}
        with open(os.path.join(data_path, "song_to_idx.json"), 'r', encoding='utf-8') as f1:
            self.song_to_idx = json.load(f1)
        with open(os.path.join(data_path,"tag_to_idx.json"), 'r', encoding='utf-8') as f2:
            self.tag_to_idx = json.load(f2)
        
        self.transform = transform
    def __len__(self):
        return len(self.training_set)
    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        songs, tags = self.training_set[idx]['songs'], self.training_set[idx]['tags']
        input_one_hot = np.zeros(input_dim)

        non_zero_indices = []
        for song in songs:
            if self.song_to_idx.get(str(song)) != None:
                input_one_hot[self.song_to_idx[str(song)]] = 1
                non_zero_indices.append(self.song_to_idx[str(song)])
        for tag in tags:
            if self.tag_to_idx.get(tag) != None:
                input_one_hot[self.tag_to_idx[tag]] = 1
                non_zero_indices.append(self.tag_to_idx[tag])

        non_zero_indices = np.array(non_zero_indices)
        #playlist_vec: one hot vec of i'th playlist

        sample = {'input_one_hot' : input_one_hot, 'target_one_hot' : input_one_hot.copy(), 'non_zero_indices' : non_zero_indices}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        input_one_hot, target_one_hot = sample['input_one_hot'], sample['target_one_hot']
        if torch.cuda.is_available():
            return {'input_one_hot': torch.FloatTensor(input_one_hot).to(device),
                'target_one_hot': torch.FloatTensor(target_one_hot).to(device)}
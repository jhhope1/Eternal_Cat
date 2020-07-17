from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import random
# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

input_dim = 57229
tag_missing_ply_false = 0.5 #1745/(2628+1745)
tag_missing_ply_true = 0.3 #1745/(2628+1745)
plylst_missing = 0.81

DATA = "../data/"

song_to_idx = {}
tag_to_idx = {}
with open(DATA + "song_to_idx.json", 'r', encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
with open(DATA + "tag_to_idx.json", 'r', encoding='utf-8') as f2:
    tag_to_idx = json.load(f2)
with open(DATA + "res_song_to_entityidx.json", 'r', encoding='utf-8') as f3:
    song_to_entityidx = json.load(f3)
with open(DATA + "res_entity_to_idx.json", 'r', encoding='utf-8') as f4:
    entity_to_idx = json.load(f4)
with open(DATA + "res_letter_to_idx.json", 'r', encoding='utf-8') as f5:
    letter_to_idx = json.load(f5)

class Noise_p(object):#warning: do add_plylst_meta first! or change 'sample['include_plylst']' part
    def __init__(self, noise_p):
        self.noise_p = noise_p

    def __call__(self, sample):
        input_one_hot = sample['input_one_hot']
        input_song = sample['input_song']
        input_tag = sample['input_tag']

        noise_input_song = np.random.choice(input_song, replace=False,
                           size=int(input_song.size * (1-self.noise_p)))
        noise_input_tag = np.random.choice(input_tag, replace=False,
                           size=int(input_tag.size * (1-self.noise_p)))

        noise_input_one_hot = np.zeros_like(input_one_hot)
        
        if sample['include_plylst']:
            if random.random()>tag_missing_ply_true:
                for tag in noise_input_tag:
                    if tag_to_idx.get(tag) != None:
                        noise_input_one_hot[tag_to_idx[tag]] = 1
        else:
            for song in noise_input_song:
                if song_to_idx.get(str(song)) != None:
                    noise_input_one_hot[song_to_idx[str(song)]] = 1
            if random.random()>tag_missing_ply_false:
                for tag in noise_input_tag:
                    if tag_to_idx.get(tag) != None:
                        noise_input_one_hot[tag_to_idx[tag]] = 1 

        sample['input_one_hot'] = np.concatenate((noise_input_one_hot,sample['plylst_meta']))
        sample['noise_input_song'] = noise_input_song
        sample['noise_input_tag'] = noise_input_tag
        sample['target_song'] = input_song
        sample['target_tag'] = input_tag
        return sample

class Noise_uniform(object): #Deprecated
    def __call__(self, sample):
        input_one_hot = sample['input_one_hot']
        non_zero_indices = sample['non_zero_indices']
        zero_indices = np.random.choice(non_zero_indices, replace=False,
                           size=int(non_zero_indices.size * random.uniform(0,1)))
        if zero_indices.size != 0:
            input_one_hot[zero_indices] = 0
        return {'input_one_hot': input_one_hot, 'target_one_hot': sample['target_one_hot']}
class add_meta(object):
    def __call__(self,sample):
        noise_input_song = sample['noise_input_song']
        meta = np.zeros(len(entity_to_idx))
        for song in noise_input_song:
            if str(song) in song_to_entityidx:
                for idx in song_to_entityidx[str(song)]:
                    meta[idx] += 1
        sample['meta_input_one_hot'] = np.concatenate((sample['input_one_hot'] , meta))
        return sample
class add_plylst_meta(object):
    def __call__(self,sample):
        plylst_meta = np.zeros(len(letter_to_idx))
        if random.random()>plylst_missing:
            for plylst_title in sample['plylst_title']:
                for l in plylst_title:
                    if l in letter_to_idx:
                        plylst_meta[letter_to_idx[l]] += 1
            sample['include_plylst'] = True
        sample['plylst_meta'] = plylst_meta
        return sample

class PlaylistDataset(Dataset):
    """Playlist target dataset."""

    def __init__(self, transform = Noise_p(0.5)):

        with open(DATA + "train.json", 'r', encoding='utf-8') as f1:
            self.training_set = json.load(f1)

        self.song_to_idx = {}
        self.tag_to_idx = {}
        with open(DATA + "song_to_idx.json", 'r', encoding='utf-8') as f1:
            self.song_to_idx = json.load(f1)
        with open(DATA + "tag_to_idx.json", 'r', encoding='utf-8') as f2:
            self.tag_to_idx = json.load(f2)
        
        self.transform = transform
    def __len__(self):
        return len(self.training_set)
    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        songs, tags = self.training_set[idx]['songs'], self.training_set[idx]['tags']
        plylst_title = self.training_set[idx]['plylst_title']

        input_one_hot = np.zeros(input_dim)

        input_song = []
        input_tag = []
        for song in songs:
            input_song.append(song)  #not string
            if self.song_to_idx.get(str(song)) != None:
                input_one_hot[self.song_to_idx[str(song)]] = 1
        for tag in tags:
            input_tag.append(tag)
            if self.tag_to_idx.get(tag) != None:
                input_one_hot[self.tag_to_idx[tag]] = 1

        input_song = np.array(input_song)
        input_tag = np.array(input_tag)
        #playlist_vec: one hot vec of i'th playlist

        sample = {'input_one_hot' : input_one_hot, 'target_one_hot' : input_one_hot.copy(), 'input_song' : input_song,'input_tag' : input_tag, 'plylst_title' : plylst_title, 'include_plylst' : False}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        meta_input_one_hot = torch.from_numpy(sample['meta_input_one_hot']).type(torch.FloatTensor)
        target_one_hot = torch.from_numpy(sample['target_one_hot']).type(torch.FloatTensor)
        return {'meta_input_one_hot':meta_input_one_hot,'target_one_hot':target_one_hot}
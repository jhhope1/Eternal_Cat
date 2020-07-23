from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import random
from const import *
# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

tag_missing_ply_false = 0.5 #1745/(2628+1745)
tag_missing_ply_true = 0.3 #1745/(2628+1745)
plylst_missing = 0.81

with open(os.path.join(data_path,"res_song_to_entityidx.json"), 'r', encoding='utf-8') as f3:
    song_to_entityidx = json.load(f3)
with open(os.path.join(data_path,"res_songidx_to_entityidx.json"), 'r', encoding='utf-8') as f3:
    songidx_to_entityidx = json.load(f3)
with open(os.path.join(data_path,"res_entity_to_idx.json"), 'r', encoding='utf-8') as f4:
    entity_to_idx = json.load(f4)
with open(os.path.join(data_path,"res_letter_to_idx.json"), 'r', encoding='utf-8') as f5:
    letter_to_idx = json.load(f5)


class Noise_p(object):#warning: do add_plylst_meta first! or change 'sample['include_plylst']' part
    def __init__(self, noise_p):
        self.noise_p = noise_p

    def __call__(self, sample):
        input_one_hot = sample['input_one_hot']
        input_song = sample['input_song']
        input_tag_idx = sample['input_tag_idx']
        input_song_idx = sample['input_song_idx']

        noise_input_song = np.random.choice(input_song, replace=False,
                           size=int(input_song.size * (1-self.noise_p)))
        noise_input_song_idx = np.random.choice(input_song_idx, replace=False,
                           size=int(input_song_idx.size * (1-self.noise_p)))
        noise_input_tag_idx = np.random.choice(input_tag_idx, replace=False,
                           size=int(input_tag_idx.size * (1-self.noise_p)))

        noise_input_one_hot = np.zeros_like(input_one_hot)
        
        if sample['include_plylst']:
            noise_input_song = []
            noise_input_song_idx = []
            if random.random()>tag_missing_ply_true:
                noise_input_one_hot[noise_input_tag_idx] = 1
        else:
            noise_input_one_hot[noise_input_song_idx] = 1
            if random.random()>tag_missing_ply_false:
                noise_input_one_hot[noise_input_tag_idx] = 1 
            
        sample['input_one_hot'] = np.concatenate((noise_input_one_hot,sample['plylst_meta']))
        sample['noise_input_song'] = noise_input_song
        sample['noise_input_song_idx'] = noise_input_song_idx
        return sample

class add_meta(object):
    def __call__(self,sample):
        meta = np.zeros(len(entity_to_idx))
        for song in sample['noise_input_song']:
            meta[song_to_entityidx[str(song)]] += 1
        for song_idx in sample['noise_input_song_idx']:
            meta[songidx_to_entityidx[str(song_idx)]] += 1
        sample['meta_input_one_hot'] = np.concatenate((sample['input_one_hot'] , meta))
        return sample

class add_plylst_meta(object):
    def __call__(self,sample):
        plylst_meta = np.zeros(len(letter_to_idx))
        if random.random()>plylst_missing:
            plylst_meta[sample['plylst_title']] = 1
            sample['include_plylst'] = True
        sample['plylst_meta'] = plylst_meta
        return sample

class PlaylistDataset(Dataset):
    """Playlist target dataset."""

    def __init__(self, transform = Noise_p(0.5)):
        with open(os.path.join(data_path, "train_to_idx.json"), 'r', encoding='utf-8') as f1:
            train_to_idx = json.load(f1)

        self.training_idx_set = train_to_idx
        for data in self.training_idx_set:
            data['songs'] = np.array(data['songs']).astype(np.int32)
            #data['tags'] = np.array(data['tags'])
            data['tags_indices'] = np.array(data['tags_indices']).astype(np.int32)
            data['songs_indices'] = np.array(data['songs_indices']).astype(np.int32)
            data['plylst_title'] = np.array(data['plylst_title']).astype(np.int32)
        
        self.transform = transform
    def __len__(self):
        return len(self.training_idx_set)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        songs= self.training_idx_set[idx]['songs']
        songs_idx = self.training_idx_set[idx]['songs_indices']
        tags_idx = self.training_idx_set[idx]['tags_indices']
        plylst_title = self.training_idx_set[idx]['plylst_title']

        input_one_hot = np.zeros(output_dim)

        input_song = []

        input_one_hot[songs_idx] = 1
        input_one_hot[tags_idx] = 1

        input_song = np.array(songs)

        #playlist_vec: one hot vec of i'th playlist
        sample = {'input_one_hot' : input_one_hot, 'target_one_hot' : torch.from_numpy(input_one_hot).type(torch.FloatTensor), 'input_song' : input_song,'input_tag_idx' : tags_idx, 'input_song_idx' : songs_idx ,'plylst_title' : plylst_title, 'include_plylst' : False}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        meta_input_one_hot = torch.from_numpy(sample['meta_input_one_hot']).type(torch.FloatTensor)
        return {'meta_input_one_hot':meta_input_one_hot,'target_one_hot':sample['target_one_hot']}
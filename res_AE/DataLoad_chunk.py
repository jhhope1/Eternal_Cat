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

with open(os.path.join(data_path,"object_to_chunkidx.json"), 'r', encoding='utf-8') as f5:
    object_to_chunkidx = json.load(f5)
for obj in object_to_chunkidx:
    object_to_chunkidx[obj] = np.array(object_to_chunkidx[obj]).astype(np.int32)
with open(os.path.join(data_path,"tag_to_chunkidx.json"), 'r', encoding='utf-8') as f5:
    tag_to_chunkidx = json.load(f5)
for obj in tag_to_chunkidx:
    tag_to_chunkidx[obj] = np.array(tag_to_chunkidx[obj]).astype(np.int32)
with open(os.path.join(data_path,"title_to_chunkidx.json"), 'r', encoding='utf-8') as f5:
    title_to_chunkidx = json.load(f5)
for obj in title_to_chunkidx:
    title_to_chunkidx[obj] = np.array(title_to_chunkidx[obj]).astype(np.int32)
with open(os.path.join(data_path,"idx_to_song.json"), 'r', encoding='utf-8') as f:
    idx_to_song = json.load(f)
with open(os.path.join(data_path,"idx_to_tag.json"),'r', encoding='utf-8') as f:
    idx_to_tag = json.load(f)



class Noise_p(object):#warning: do add_plylst_meta first! or change 'sample['include_plylst']' part
    def __init__(self, noise_p):
        self.noise_p = noise_p

    def __call__(self, sample):
        input_one_hot = sample['input_one_hot']
        input_song = sample['input_song']
        input_tag_idx = sample['input_tag_idx']
        input_tag = sample['input_tag']
        input_song_idx = sample['input_song_idx']

        noise_input_song = np.random.choice(input_song, replace=False,
                           size=int(input_song.size * (1-self.noise_p)))
        noise_input_song_idx = np.random.choice(input_song_idx, replace=False,
                           size=int(input_song_idx.size * (1-self.noise_p)))
        noise_input_tag_idx = np.random.choice(input_tag_idx, replace=False,
                           size=int(input_tag_idx.size * (1-self.noise_p)))
        noise_input_tag = np.random.choice(input_tag, replace=False,
                           size=int(input_tag.size * (1-self.noise_p)))

        noise_input_one_hot = np.zeros_like(input_one_hot)
        
        if sample['include_plylst']:
            noise_input_song = []
            noise_input_song_idx = []
            if random.random()>tag_missing_ply_true:
                noise_input_one_hot[noise_input_tag_idx] = 1
            else:
                noise_input_tag_idx = []
        else:
            noise_input_one_hot[noise_input_song_idx] = 1
            if random.random()>tag_missing_ply_false:
                noise_input_one_hot[noise_input_tag_idx] = 1 
            else:
                noise_input_tag_idx = []
        
        sample['noise_input_tag'] = noise_input_tag
        sample['noise_input_tag_idx'] = noise_input_tag_idx
        sample['input_one_hot'] = noise_input_one_hot
        sample['noise_input_song'] = noise_input_song
        sample['noise_input_song_idx'] = noise_input_song_idx
        return sample

class add_meta(object):
    def __call__(self,sample):
        for song in sample['noise_input_song']:
            sample['chunk_one_hot'][object_to_chunkidx[str(song)]] += 1
        for song_idx in sample['noise_input_song_idx']:
            sample['chunk_one_hot'][object_to_chunkidx[str(idx_to_song[str(song_idx)])]] += 1

        for tag in sample['noise_input_tag']:
            sample['chunk_one_hot'][tag_to_chunkidx[tag]] += 1
        for tag_idx in sample['noise_input_tag_idx']:
            sample['chunk_one_hot'][tag_to_chunkidx[idx_to_tag[str(tag_idx)]]] += 1

        sample['meta_input_one_hot'] = np.concatenate((sample['input_one_hot'] , sample['chunk_one_hot']))
        return sample

class add_plylst_meta(object):
    def __call__(self,sample):
        chunk_one_hot = np.zeros(chunk_size)
        ######################################### should change plylst_title to real_playlist_title
        ######################################### should change train_to_idx_chunk
        if random.random()>plylst_missing:
            chunk_one_hot[title_to_chunkidx[sample['plylst_title']]] = 1
            sample['include_plylst'] = True
        sample['chunk_one_hot'] = chunk_one_hot
        return sample

class PlaylistDataset(Dataset):
    """Playlist target dataset."""

    def __init__(self, transform = Noise_p(0.5)):
        with open(os.path.join(data_path, "train_to_idx_chunk.json"), 'r', encoding='utf-8') as f1:
            train_to_idx_chunk = json.load(f1)

        self.training_idx_set = train_to_idx_chunk
        for data in self.training_idx_set:
            data['songs'] = np.array(data['songs'])
            data['tags'] = np.array(data['tags'])
            data['tags_indices'] = np.array(data['tags_indices']).astype(np.int32)
            data['songs_indices'] = np.array(data['songs_indices']).astype(np.int32)
        
        self.transform = transform
    def __len__(self):
        return len(self.training_idx_set)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #songs and tags are not intersect with song_idx and tag_idx
        songs= self.training_idx_set[idx]['songs']
        tags = self.training_idx_set[idx]['tags']
        songs_idx = self.training_idx_set[idx]['songs_indices']
        tags_idx = self.training_idx_set[idx]['tags_indices']
        plylst_title = self.training_idx_set[idx]['plylst_title']

        input_one_hot = np.zeros(output_dim)

        input_song = []

        input_one_hot[songs_idx] = 1
        input_one_hot[tags_idx] = 1

        input_song = np.array(songs)

        #playlist_vec: one hot vec of i'th playlist
        sample = {'input_one_hot' : input_one_hot, 'target_one_hot' : torch.from_numpy(input_one_hot).type(torch.FloatTensor), 'input_song' : input_song,'input_tag_idx' : tags_idx, 'input_song_idx' : songs_idx, 'input_tag':tags ,'plylst_title' : plylst_title, 'include_plylst' : False}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        meta_input_one_hot = torch.from_numpy(sample['meta_input_one_hot']).type(torch.FloatTensor)
        return {'meta_input_one_hot':meta_input_one_hot,'target_one_hot':sample['target_one_hot']}
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

input_dim = 61706
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')
type_nn = ['title', 'title_tag', 'song_meta_tag', 'song_meta']

song_to_idx = {}
tag_to_idx = {}
song_size = 0
tag_size = 0
entity_size = 0
l_num = 0

with open(os.path.join(data_path, "song_to_idx.json"), 'r', encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
    song_size = len(song_to_idx)

with open(os.path.join(data_path,"tag_to_idx.json"), 'r', encoding='utf-8') as f2:
    tag_to_idx = json.load(f2)
    tag_size = len(tag_to_idx)

with open(os.path.join(data_path,"res_song_to_entityidx.json"), 'r', encoding='utf-8') as f3:
    song_to_entityidx = json.load(f3)

with open(os.path.join(data_path,"res_entity_to_idx.json"), 'r', encoding='utf-8') as f4:
    entity_to_idx = json.load(f4)
    entity_size = len(entity_to_idx)

with open(os.path.join(data_path, 'tag_occur.json'), 'r', encoding='utf-8') as f5:
    title_to_tag = json.load(f5)


class Noise_p(object):#warning: do add_plylst_meta first! or change 'sample['include_plylst']' part
    def __init__(self, noise_p):
        self.noise_p = noise_p

    def __call__(self, sample):
        input_song = sample['input_song']
        input_tag = sample['input_tag']

        if 'song' in sample['id_nn']:
            noise_input_song = np.random.choice(input_song, replace=False,
                            size=int(input_song.size * (1-self.noise_p)))
            noise_song_one_hot = np.zeros(song_size)
            for song in noise_input_song:
                if song_to_idx.get(str(song)) != None:
                    noise_song_one_hot[song_to_idx[str(song)]] = 1
            sample['noise_song_one_hot'] = noise_song_one_hot
            sample['noise_input_song'] = noise_input_song

        #add tag
        if 'song_meta' != sample['id_nn']:
            noise_tag_one_hot = np.zeros(tag_size)
        if 'tag' in sample['id_nn']:
            noise_input_tag = np.random.choice(input_tag, replace=False,
                            size=int(input_tag.size * (1-self.noise_p)))
            for tag in noise_input_tag:
                if tag_to_idx.get(tag) != None:
                    noise_tag_one_hot[tag_to_idx[tag]-song_size] = 1 
            sample['noise_tag_one_hot'] = noise_tag_one_hot
            sample['noise_input_tag'] = noise_input_tag
        if 'title' == sample['id_nn']:
            sample['noise_tag_one_hot'] = noise_tag_one_hot

        sample['target_song'] = input_song
        sample['target_tag'] = input_tag
        return sample

class add_meta(object):
    def __call__(self,sample):
        if 'song' in sample['id_nn']:
            meta = np.zeros(len(entity_to_idx))
            for song in sample['noise_input_song']:
                if str(song) in song_to_entityidx:
                    for idx in song_to_entityidx[str(song)]:
                        meta[idx] += 1

        if 'title' not in sample['id_nn']:
            if 'tag' in sample['id_nn']:
                sample['meta_input_one_hot_song_meta_tag'] = np.concatenate((sample['noise_song_one_hot'], meta, sample['noise_tag_one_hot']))
            else:
                sample['meta_input_one_hot_song_meta'] = np.concatenate((sample['noise_song_one_hot'], meta))
        return sample

class add_plylst_meta(object):
    def __call__(self,sample):
        if "title" in sample['id_nn']:
            for tag_idx in sample['plylst_title_to_tag']:
                sample['noise_tag_one_hot'][tag_idx - song_size] += 1
        if sample['id_nn']=='title':
            sample['meta_input_one_hot_title'] = sample['noise_tag_one_hot']
        if sample['id_nn']=='title_tag':
            sample['meta_input_one_hot_title_tag'] = sample['noise_tag_one_hot']
        return sample

class PlaylistDataset(Dataset):
    """Playlist target dataset."""

    def __init__(self, id_nn , transform = Noise_p(0.5)):

        with open(os.path.join(data_path, "train.json"), 'r', encoding='utf-8') as f1:
            self.training_set = json.load(f1)

        self.song_to_idx = {}
        self.tag_to_idx = {}
        with open(os.path.join(data_path, "song_to_idx.json"), 'r', encoding='utf-8') as f1:
            self.song_to_idx = json.load(f1)
        with open(os.path.join(data_path,"tag_to_idx.json"), 'r', encoding='utf-8') as f2:
            self.tag_to_idx = json.load(f2)
        self.id_nn = id_nn
        if self.id_nn not in type_nn:
            print("Error :: id_nn is not in typenn")
            return None
        self.transform = transform
    def __len__(self):
        return len(self.training_set)
    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        songs, tags = self.training_set[idx]['songs'], self.training_set[idx]['tags']
        plylst_title = self.training_set[idx]['plylst_title']
        plylst_title_to_tag = title_to_tag[idx]

        target_one_hot = np.zeros(input_dim)
        input_song = []
        input_tag = []
        for song in songs:
            input_song.append(song)  #not string
            if self.song_to_idx.get(str(song)) != None:
                target_one_hot[self.song_to_idx[str(song)]] = 1
        for tag in tags:
            input_tag.append(tag)
            if self.tag_to_idx.get(tag) != None:
                target_one_hot[self.tag_to_idx[tag]] = 1

        input_song = np.array(input_song)
        input_tag = np.array(input_tag)
        #playlist_vec: one hot vec of i'th playlist
        sample = {'target_one_hot' : target_one_hot, 'input_song' : input_song,'input_tag' : input_tag, 'plylst_title_to_tag' : plylst_title_to_tag, 'id_nn' : self.id_nn}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        ret = {}
        ret['meta_input_one_hot_' + sample['id_nn']] = torch.from_numpy(sample['meta_input_one_hot_' + sample['id_nn']]).float().to(device)
        ret['target_one_hot'] = torch.from_numpy(sample['target_one_hot']).float().to(device)
        if 'song' in sample['id_nn']:
            ret['noise_song_one_hot'] = sample['noise_song_one_hot']
        if 'tag' in sample['id_nn']:
            ret['noise_tag_one_hot'] = sample['noise_tag_one_hot']
        return ret
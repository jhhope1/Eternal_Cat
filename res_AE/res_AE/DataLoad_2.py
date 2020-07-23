from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import random

import plylist_tensor_converter as ptc

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

input_dim = 38459
output_dim = 20517
song_size = 12538
meta_offset = 21517
entity_size = 38459 - 21517

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlaylistDataset(Dataset):
    """Playlist target dataset."""

    def __init__(self, noise_p):
        self.training_set_input = []
        self.training_set_output = []
        self.included_songs = []
        self.song_to_entity = ptc.build_song_to_entity()
        self.noise_p = noise_p

        with open(os.path.join(data_path, "train.json"), 'r', encoding='utf-8') as f1:
            train = json.load(f1)            
            for idx, playlist in enumerate(train):
                input_idx_bundle, included_song = ptc.plylist_without_meta(playlist)
                self.training_set_input.append(torch.tensor(input_idx_bundle).long().to(device))
                self.included_songs.append(torch.tensor(included_song).long().to(device))
                self.training_set_output.append(torch.tensor(ptc.plylist_song_tag_only(playlist)).long().to(device))

     
    def __len__(self):
        return len(self.training_set_input)

    def __getitem__(self, idx):
        mask = torch.cuda.FloatTensor(meta_offset).uniform_() > self.noise_p
        ret_inp = mask * torch.zeros(meta_offset,device=device).scatter_(0, self.training_set_input[idx], 1)
        entity_tensor = torch.zeros(entity_size, device=device)
        for song_idx in self.included_songs[idx]:
            if mask[song_idx]:
                entity_tensor[self.song_to_entity[song_idx]] += 1
        ret_inp = torch.cat([ret_inp, entity_tensor])
        ret_oup = torch.zeros(output_dim, device = device).scatter_(0, self.training_set_output[idx], 1)
        return ret_inp, ret_oup
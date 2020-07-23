from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import random


PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(data_path, "song_to_idx.json"), 'r', encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
with open(os.path.join(data_path,"tag_to_idx.json"), 'r', encoding='utf-8') as f2:
    tag_to_idx = json.load(f2)
with open(os.path.join(data_path,"res_song_to_entityidx.json"), 'r', encoding='utf-8') as f3:
    song_to_entityidx = json.load(f3)

with open(os.path.join(data_path,"res_entity_to_idx.json"), 'r', encoding='utf-8') as f4:
    entity_to_idx = json.load(f4)
    entity_size = len(entity_to_idx)

with open(os.path.join(data_path,"res_letter_to_idx.json"), 'r', encoding='utf-8') as f5:
    letter_to_idx = json.load(f5)

including_metadata = True
including_title = True

letter_set = set([letter for letter, val in letter_to_idx.items()])

tag_offset = len(song_to_idx)
title_offset = tag_offset + len(tag_to_idx)
meta_offset = title_offset + len(letter_set)

def plylist_to_sparse_idx(playlist, idx):
    idx_bundle = []
    for song in playlist["songs"]:
        if song_to_idx.get(str(song)) == None:
            continue
        idx_bundle.append(song_to_idx[str(song)])
        for meta_item in song_to_entityidx[str(song)]:
            idx_bundle.append(meta_item + meta_offset)
    for tag in playlist["tags"]:
        if tag_to_idx.get(str(tag)) == None:
            continue
        idx_bundle.append(tag_to_idx[tag])
    title_txt = playlist["plylst_title"]
    for letter in letter_set:
        if letter in title_txt:
            idx_bundle.append(letter_to_idx[letter] + title_offset)
    
    return idx_bundle

def plylist_song_tag_only(playlist):
    idx_bundle = []
    for song in playlist["songs"]:
        if song_to_idx.get(str(song)) == None:
            continue
        idx_bundle.append(song_to_idx[str(song)])
    for tag in playlist["tags"]:
        if tag_to_idx.get(str(tag)) == None:
            continue
        idx_bundle.append(tag_to_idx[tag])
    return idx_bundle

def plylist_without_meta(playlist):
    idx_bundle = []
    included_song = []

    for song in playlist["songs"]:
        if song_to_idx.get(str(song)) == None:
            continue
        idx_bundle.append(song_to_idx[str(song)])
        included_song.append(song_to_idx[str(song)])
        
    for tag in playlist["tags"]:
        if tag_to_idx.get(str(tag)) == None:
            continue
        idx_bundle.append(tag_to_idx[tag])
    title_txt = playlist["plylst_title"]
    for letter in letter_set:
        if letter in title_txt:
            idx_bundle.append(letter_to_idx[letter] + title_offset)
    
    return idx_bundle, included_song

def build_song_to_entity():
    ret = [torch.tensor(0) for _ in range(len(song_to_idx))]
    for song, entity_list in song_to_entityidx.items():
        if song_to_idx.get(song):
            ret[song_to_idx[song]] = torch.tensor([x for x in entity_list],device=device).long()
    return ret

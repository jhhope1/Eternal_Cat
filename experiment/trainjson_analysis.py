import os
import numpy as np
import json
from itertools import combinations
import torch
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data\\')
A = [1,1,1,1,1,2,3]
B = torch.tensor([0,0,0,0,0])

B[A] += 1
print (B)
l_num = 1000

with open(DATA + "train.json", 'r', encoding='utf-8') as f1:
    train = json.load(f1)

with open(DATA+"song_meta.json", "r", encoding = 'utf-8') as f2:
    song_meta = json.load(f2)

tag_set = set()
song_name_set = set()
album_name_set = set()
artist_name_set = set()

for playlist in train:
    for tag in playlist['tags']:
        tag_set.add(tag)
for song in song_meta:
    song_name_set.add(song['song_name'])
    album_name_set.add(song['album_name'])
    if song['album_name']!=None:
        for artist_name in song['album_name']:
            artist_name_set.add(artist_name)

l_set = set()
l_map = {}
song_len = {}
for playlist in train:
    title = playlist['plylst_title']
    if len(playlist['tags']) not in song_len:
        song_len[len(playlist['tags'])] = 1
    else:
        song_len[len(playlist['tags'])] += 1
    for l in title:
        l_set.add(l)
        if l in l_map:
            l_map[l]+=1
        else:
            l_map[l]=1
    for song in playlist['songs']:
        a = song_meta[song]

sorted_l = sorted(l_map.items(), reverse = True, key=lambda item: item[1])

letter_to_idx = {}
for i, k in enumerate(sorted_l):
    if k not in letter_to_idx:
        letter_to_idx[k[0]] = i

with open(DATA + "res_letter_to_idx.json", 'w', encoding='utf-8') as f3:
    json.dump(letter_to_idx,f3, ensure_ascii=False)

s = 0
for i in song_len:
    s+=song_len[i]
d = 0
for a, i in enumerate(song_len):
    if a==0:
        d+=i
    else:
        d-=i/(2**a)
A = 1
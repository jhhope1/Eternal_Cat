import os
import numpy as np
import json
from itertools import combinations 

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data\\')

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
    for artist_name_name in song['album_name']:
        album_name_set.add(album_name)


for playlist in train:
    title = playlist['plylst_title']
    substrings = [test_str[x:y] for x, y in combinations(range(len(test_str) + 1), r = 2)]
    for s in substrings:

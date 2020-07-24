#import pandas as pd
import numpy as np
import torch
import os
import json
from const import *

with open(os.path.join(data_path, 'train.json'), encoding='utf-8') as f1:
    train = json.load(f1)
with open(os.path.join(data_path, 'val.json'), encoding='utf-8') as f2:
    val = json.load(f2)
with open(os.path.join(data_path, 'song_meta.json'), encoding='utf-8') as f3:
    meta = json.load(f3)

total = train + val
song_to_newdt = {}
for song in meta:
    song_to_newdt[str(song['id'])] = int(song['issue_date'])
cnt = 0
for data in total:
    pre_date = data['updt_date']
    play_date = int(pre_date[0:4]+pre_date[5:7]+pre_date[8:10])
    for song in data['songs']:
        #should modify
        if song_to_newdt[str(song)]!=0 and song_to_newdt[str(song)]>play_date:
            song_to_newdt[str(song)] = 0
            cnt += 1
with open(os.path.join(data_path,'song_to_newdt.json'), 'w', encoding='utf-8') as f4:
    json.dump(song_to_newdt, f4)

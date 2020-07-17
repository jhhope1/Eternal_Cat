vector_size = 12393
song_size = 0
import json
import random
song_to_idx = {}
tag_to_idx = {}
idx_to_item = []
with open("song_to_idx.json", 'r', encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
    song_size = len(song_to_idx)
with open("tag_to_idx.json", 'r', encoding='utf-8') as f2:
    tag_to_idx = json.load(f2)
with open("idx_to_item.json", 'r', encoding='utf-8') as f3:
    idx_to_item = json.load(f3)



"""
with open("val.json", "r", encoding='utf-8') as f3, open("val_as_hot.json", "w", encoding='utf-8') as f4:
    val_list = json.load(f3)
    vector_list = []
    for playlist in val_list:
        playlist_vec= [0 for idx in range(vector_size)]
        for song in playlist["songs"]:
            if song_to_idx.get(song) != None:
                playlist_vec[song_to_idx[song]] = 1
        for tag in playlist["tags"]:
            if tag_to_idx.get(tag) != None:
                playlist_vec[tag_to_idx[tag]] = 1
        vector_list.append(playlist_vec)
    json.dump(vector_list, f4, ensure_ascii=False)
"""

import model_inference as mi

with open("train.json", "r", encoding = 'utf-8') as f3, open("result.json", "w", encoding = 'utf-8') as f4:
    ans_list = []
    val_list = json.load(f3)
    val_list.reverse()
    for playlist in val_list:
        if len(playlist['songs']) < 100:
            continue
        playlist_vec= [0 for idx in range(vector_size)]
        inf_ans = {}

        for song in playlist["songs"]:
            if song_to_idx.get(song) != None:
                playlist_vec[song_to_idx[song]] = 1
        for tag in playlist["tags"]:
            if tag_to_idx.get(tag) != None:
                playlist_vec[tag_to_idx[tag]] = 1
        fake_playlist = []
        laterchecklist = []
        alreadyin = []
        for idx in range(len(playlist_vec)):
            checked = playlist_vec[idx]
            if checked == 1:
                if random.random()%2 == 0:
                    checked = 0
                    laterchecklist.append(idx)
                else:
                    alreadyin.append(idx)
            fake_playlist.append(checked)

        inferenced_vec = mi.inference(fake_playlist)[0]
        inferenced_list = [(inferenced_vec[idx], idx) for idx in range(len(inferenced_vec))]
        inferenced_list = sorted(inferenced_list, reverse=True)
        tag_max = 10
        tag_ctr = 0
        song_max = 100
        song_ctr = 0
        loc_song = []
        loc_tag = []
        for val, idx in inferenced_list:
            if idx in range(song_size) and song_ctr < song_max:
                if idx in alreadyin:
                    continue
                else:
                    if idx in laterchecklist:
                        corrate += 1
                    song_ctr += 1
            if idx in range(song_size, vector_size) and tag_ctr < tag_max:
                if idx in alreadyin:
                    continue
                else:
                    if idx in laterchecklist:
                        corrate += 1
                    tag_ctr += 1
        
        print(len(corrate)/len(laterchecklist))

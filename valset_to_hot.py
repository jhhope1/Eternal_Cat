vector_size = 12393
song_size = 0
import json
song_to_idx = {}
tag_to_idx = {}
idx_to_item = []
with open("song_to_idx.json", 'r', encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
    song_size = len(song_to_idx)
with open("tag_to_idx.json", 'r', encoding='utf-8') as f2:
    tag_to_idx = json.load(f2)
with open("idx_to_item.json", 'r ', encoding='utf-8') as f3:
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

with open("val.json", "r", encoding = 'utf-8') as f3, open("result.json", "w", encoding = 'utf-8') as f4:
    ans_list = []
    val_list = json.load(f3)
    for playlist in val_list:
        playlist_vec= [0 for idx in range(vector_size)]
        inf_ans = {}
        for song in playlist["songs"]:
            if song_to_idx.get(song) != None:
                playlist_vec[song_to_idx[song]] = 1
        for tag in playlist["tags"]:
            if tag_to_idx.get(tag) != None:
                playlist_vec[tag_to_idx[tag]] = 1
        inferenced_vec = mi.inference(playlist_vec)
        inferenced_list = [(inferenced_vec[idx], idx) for idx in range(len(inferenced_vec))]
        inferenced_list = sorted(inferenced_list, reverse=True)
        tag_max = 10
        tag_ctr = 0
        song_max = 100
        song_ctr = 0
        loc_song = []
        loc_tag = []
        for val, idx in inferenced_list:
            if idx in range(song_size) and tag_ctr < tag_max:
                if idx_to_item[idx] in playlist['song']:
                    continue
                else:
                    loc_song.append(int(idx_to_item[idx]))
                    tag_ctr += 1
            if idx in range(song_size, vector_size) and song_ctr < song_max:
                if idx_to_item[idx] in playlist['tag']:
                    continue
                else:
                    loc_tag.append(idx_to_item[idx])
                    song_ctr += 1
                
        inf_ans['id'] = playlist['id']
        inf_ans['songs'] = loc_song
        inf_ans['tags'] = loc_tag
        ans_list.append(inf_ans)
    json.dump(ans_list, f4, ensure_ascii=False)


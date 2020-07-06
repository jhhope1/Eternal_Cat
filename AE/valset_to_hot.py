import json
import os
input_dim = 31202
song_size = 0
song_to_idx = {}
tag_to_idx = {}
idx_to_item = []

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

with open(os.path.join(data_path, "song_to_idx.json"), 'r', encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
    song_size = len(song_to_idx)
with open(os.path.join(data_path, "tag_to_idx.json"), 'r', encoding='utf-8') as f2:
    tag_to_idx = json.load(f2)
with open(os.path.join(data_path, "idx_to_item.json"), 'r', encoding='utf-8') as f3:
    idx_to_item = json.load(f3)

import model_inference as mi

with open(os.path.join(data_path, "val.json"), "r", encoding = 'utf-8') as f3, open(os.path.join(data_path, "results.json"), "w", encoding = 'utf-8') as f4:
    ans_list = []
    val_list = json.load(f3)
    cnt = 0
    for playlist in val_list:
        cnt+=1
        playlist_vec= [0 for idx in range(input_dim)]
        inf_ans = {}
        for song in playlist["songs"]:
            if song_to_idx.get(str(song)) != None:
                playlist_vec[song_to_idx[str(song)]] = 1
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
                if idx_to_item[idx] in playlist['songs']:
                    continue
                else:
                    loc_song.append(int(idx_to_item[idx]))
                    tag_ctr += 1
            if idx in range(song_size, input_dim) and song_ctr < song_max:
                if idx_to_item[idx] in playlist['tags']:
                    continue
                else:
                    loc_tag.append(idx_to_item[idx])
                    song_ctr += 1
            if tag_ctr==tag_max and song_ctr == song_max:
                break
        inf_ans['id'] = playlist['id']
        inf_ans['songs'] = loc_song
        inf_ans['tags'] = loc_tag
        ans_list.append(inf_ans)
        if cnt%1000==0:
            print(cnt)
    json.dump(ans_list, f4, ensure_ascii=False)


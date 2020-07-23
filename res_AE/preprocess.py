import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from res_AE import const
data_path = const.data_path

taking_song = set()

#바꿀 필요 있을듯? 많이 나온 태그 같은걸로?
with open(os.path.join(data_path,"song_to_idx.json"), 'r',encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
    for song, idx in song_to_idx.items():
        taking_song.add(int(song))

l_num = 1000

entity_to_idx = {}
song_to_entityidx_map = {}
songidx_to_entityidx_map = {}
def add_entity(song, entity):
    if entity not in entity_to_idx:
        return
    idx = entity_to_idx[entity]
    song_to_entityidx_map[song].append(idx)
    if str(song) in song_to_idx:
        songidx_to_entityidx_map[song_to_idx[str(song)]].append(idx)
def add_entity_idx(entity):
    if entity not in entity_to_idx:
        entity_to_idx[entity] = len(entity_to_idx)
with open(os.path.join(data_path, "song_meta.json"), 'r', encoding='utf-8') as f1:
    data = json.load(f1)
    for song_idx, song_data in enumerate(data):
        if song_idx not in taking_song:
            continue

        for genre in song_data["song_gn_gnr_basket"]:
            add_entity_idx(genre)
        for genre in song_data["song_gn_dtl_gnr_basket"]:
            add_entity_idx(genre)
        album = song_data["album_id"]
        add_entity_idx("al."+str(album))
        for artist in song_data["artist_id_basket"]:
            add_entity_idx("artist."+str(artist))
        add_entity_idx("year."+song_data["issue_date"][0:4])
        
        #add_entity_idx("month."+song_data["issue_date"][4:6])

with open(os.path.join(data_path, "song_meta.json"), 'r', encoding='utf-8') as f1:
    data = json.load(f1)
    song_to_entityidx_map = {i : [] for i in range(len(data))}
    songidx_to_entityidx_map = {i : [] for i in range(len(song_to_idx))}
    for song_idx, song_data in enumerate(data):
        for genre in song_data["song_gn_gnr_basket"]:
            add_entity(song_idx,genre)
        for genre in song_data["song_gn_dtl_gnr_basket"]:
            add_entity(song_idx,genre)
        album = song_data["album_id"]
        add_entity(song_idx,"al."+str(album))
        for artist in song_data["artist_id_basket"]:
            add_entity(song_idx,"artist."+str(artist))
        add_entity(song_idx,"year."+song_data["issue_date"][0:4])
        #add_entity(song_idx,"month."+song_data["issue_date"][4:6])


with open(os.path.join(data_path, "train.json"), 'r', encoding='utf-8') as f1:
    train = json.load(f1)

l_map = {}
song_len = {}
for playlist in train:
    title = playlist['plylst_title']
    if len(playlist['tags']) not in song_len:
        song_len[len(playlist['tags'])] = 1
    else:
        song_len[len(playlist['tags'])] += 1
    for l in title:
        if l in l_map:
            l_map[l]+=1
        else:
            l_map[l]=1

sorted_l = sorted(l_map.items(), reverse = True, key=lambda item: item[1])
letter_to_idx = {}

for i in range(l_num):
    letter_to_idx[sorted_l[i][0]] = i

with open(os.path.join(data_path,  "res_song_to_entityidx.json"), 'w') as f2:
    json.dump(song_to_entityidx_map,f2)

with open(os.path.join(data_path, "res_songidx_to_entityidx.json"), 'w') as f2:
    json.dump(songidx_to_entityidx_map,f2)

with open(os.path.join(data_path, "res_entity_to_idx.json"), 'w') as f3:
    json.dump(entity_to_idx,f3)

with open(os.path.join(data_path, "res_letter_to_idx.json"), 'w', encoding='utf-8') as f3:
    json.dump(letter_to_idx,f3,ensure_ascii=False)
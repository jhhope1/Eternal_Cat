import json
import os

taking_song = set()

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data\\')
WDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data\\')

#바꿀 필요 있을듯? 많이 나온 태그 같은걸로?
with open(DATA+"song_to_idx.json", 'r',encoding='utf-8') as f1:
    data = json.load(f1)
    for song, idx in data.items():
        taking_song.add(int(song))

l_num = 1000

entity_to_idx = {}
song_to_entityidx_map = {}
def add_entity(song, entity):
    if entity not in entity_to_idx:
        return
    idx = entity_to_idx[entity]
    song_to_entityidx_map[song].append(idx)
def add_entity_idx(entity):
    if entity not in entity_to_idx:
        entity_to_idx[entity] = len(entity_to_idx)
with open(DATA+"song_meta.json", 'r', encoding='utf-8') as f1:
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

with open(DATA+"song_meta.json", 'r', encoding='utf-8') as f1:
    data = json.load(f1)
    song_to_entityidx_map = {i : [] for i in range(len(data))}
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


with open(DATA + "train.json", 'r', encoding='utf-8') as f1:
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

with open(WDATA + "res_song_to_entityidx.json", 'w') as f2:
    json.dump(song_to_entityidx_map,f2)

with open(WDATA + "res_entity_to_idx.json", 'w') as f3:
    json.dump(entity_to_idx,f3)

with open(DATA + "res_letter_to_idx.json", 'w', encoding='utf-8') as f3:
    json.dump(letter_to_idx,f3,ensure_ascii=False)
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from res_AE import const
data_path = const.data_path

with open(os.path.join(data_path, "train.json"), "r", encoding="utf-8") as f1:
    train = json.load(f1)
with open(os.path.join(data_path, "song_to_idx.json"), 'r', encoding="utf-8") as f:
    song_to_idx = json.load(f)
with open(os.path.join(data_path, "tag_to_idx.json"), 'r', encoding="utf-8") as f:
    tag_to_idx = json.load(f)
with open(os.path.join(data_path, "res_letter_to_idx.json"), 'r', encoding="utf-8") as f:
    res_letter_to_idx = json.load(f)

for _, data in enumerate(train):
    song_indices = []
    tag_indices = []
    title_indices = []
    songs = []
    for song in data['songs']:
        if song_to_idx.get(str(song)) != None:
            song_indices.append(song_to_idx[str(song)])
        else:
            songs.append(str(song))
    for tag in data['tags']:
        if tag_to_idx.get(tag) != None:
            tag_indices.append(tag_to_idx[tag])
    data['songs_indices'] = song_indices
    data['songs'] = songs
    data['tags_indices'] = tag_indices

with open(os.path.join(data_path, "train_to_idx_chunk.json"), 'w', encoding='utf-8') as f2:
    json.dump(train, f2, ensure_ascii=False)
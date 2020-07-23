import json
import os

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

with open(os.path.join(data_path, "train.json"), "r", encoding="utf-8") as f1:
    train = json.load(f1)
with open(os.path.join(data_path, "song_to_idx"), 'r', encoding="utf-8") as f:
    song_to_idx = json.load(f)
with open(os.path.join(data_path, "tag_to_idx"), 'r', encoding="utf-8") as f:
    tag_to_idx = json.load(f)

for _, data in enumerate(train):
    song_indices = []
    tag_indices = []
    for song in data['songs']:
        if song_to_idx.get(str(song)) != None:
            song_indices.append(song_to_idx[str(song)])
    for tag in data['tags']:
        if tag_to_idx.get(tag) != None:
            tag_indices.append(tag_to_idx[tag])
    data['songs'] = song_indices
    data['tags'] = tag_indices

with open(os.path.join(data_path, "train_to_idx"), 'w', encoding='utf-8') as f2:
    json.dump(train, f2)
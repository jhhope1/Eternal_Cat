import json, os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from res_AE import const as C

data_path = C.data_path

with open(os.path.join(data_path, 'val.json'), 'r', encoding='utf-8') as f:
    val = json.load(f)

song_set = set()
tag_set = set()
for pl in val:
    for song in pl['songs']:
        song_set.add(song)
    for tag in pl['tags']:
        tag_set.add(tag)
print(len(song_set))
print(len(tag_set))

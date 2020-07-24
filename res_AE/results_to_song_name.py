import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from res_AE import const
data_path = const.data_path
with open(os.path.join(data_path,'song_meta.json'), encoding='utf-8') as f:
    song_meta = json.load(f)
with open(os.path.join(data_path,'results_test.json'), encoding='utf-8') as f:
    results = json.load(f)
for data in results:
    song_names = []
    for song_id in data['songs']:
        song_names.append('song_name : ')
        song_names.append(song_meta[song_id]['song_name'])
        if 'artist_name_basket' in song_meta[song_id]:
            song_names.append("artist_name")
            song_names.append(song_meta[song_id]['artist_name_basket'])
    print("song names = ", song_names)
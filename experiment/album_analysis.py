import os, sys, json
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from res_AE import const as C


with open(os.path.join(C.data_path, 'song_meta.json'), encoding='utf-8') as f:
    song_meta = json.load(f)

with open(os.path.join(C.data_path, 'train.json'), encoding='utf-8') as f:
    train = json.load(f)

song_to_album = {}
for song_info in song_meta:
    song_to_album[song_info['id']] = song_info['album_id']

album_num_dstb = {}
def add_key(key: int):
    if album_num_dstb.get(key):
        album_num_dstb[key] += 1
    else:
        album_num_dstb[key] = 1

def take_avg():
    w_sum = 0
    w_num = 0
    for key, val in album_num_dstb.items():
        w_sum += key * val
        w_num += val
    return w_sum / w_num

for playlist in train:
    albums = set()
    for song in playlist['songs']:
        albums.add(song_to_album[song])
    
    add_key(len(albums))

print('# of playlists with single albums: ', album_num_dstb.get(1))
print('# of playlists: ', len(train))
print('Average # of albums: {:.4f}'.format(take_avg()))
print(album_num_dstb)
import json
import os
taking_song = set()
taking_item = set()

string = ''

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data\\')
WDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data\\music_kakao\\')

with open(DATA+"song_to_idx.json", 'r',encoding='utf-8') as f1:
    data = json.load(f1)
    for song, idx in data.items():
        taking_song.add(int(song))


with open(DATA+"song_meta.json", 'r', encoding='utf-8') as f1:
    data = json.load(f1)
    for idx, song in enumerate(data):
        if idx not in taking_song:
            continue
        for genre in song["song_gn_gnr_basket"]:
            string += str(idx) + ' has.genre ' + genre + '\n'
            taking_item.add(genre)
        album = song["album_id"]
        string += str(idx) + ' is.album ' + "al."+str(album) + '\n'
        taking_item.add("al." + str(album))
        for artist in song["artist_id_basket"]:
            string += str(idx) + ' is.artist ' + "artist."+str(artist) + '\n'
            taking_item.add("artist." + str(artist))

with open(WDATA + "kg.txt", 'w') as f2:
    f2.write(string)

string = ''
taking_song
for idx, item in enumerate(taking_song):
    string += str(item) + '\t' + str(idx) + '\n'

with open(WDATA + "item2idx.txt", 'w') as f2:
    f2.write(string)
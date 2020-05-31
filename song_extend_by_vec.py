import numpy as np
import json

import concurrent.futures
from time import time
import itertools
    
result = []
with open('song_vec.json', 'r', encoding='utf-8') as f1:
        song_vec1 = json.load(f1)

with open('tag_vec.json', 'r', encoding='utf-8') as f2:
    tag_vec1 = json.load(f2)

song_vec = {song : np.asarray(vec) for song, vec in song_vec1.items()}
song_exi = {song : True for song, vec in song_vec1.items()}
tags_vec = {song : np.asarray(vec) for song, vec in tag_vec1.items()}
tags_exi = {song : True for song, vec in tag_vec1.items()}


with open('val.json', 'r', encoding='utf-8') as f3:
    target_data = json.load(f3)



LapEig = 10
j = 0

def find_and_append(data):

    playlist_vec = np.zeros(LapEig)
    i = 0
    for song in data['songs']:
        if song_exi.get(song) == None:
            continue
        else:
            i += 1
            playlist_vec = playlist_vec + song_vec[song]
    if i != 0:
        playlist_vec = playlist_vec/i
    tag_vec = np.zeros(LapEig)
    i = 0
    for tag in data['tags']:
        if tags_exi.get(tag) == None:
            continue
        else:
            tag_vec = tag_vec + tags_vec[tag]
    if i != 0:
        tag_vec = tag_vec/i
    
    playlist_vec = (playlist_vec + tag_vec)/2
    if len(data['tags']) == 0:
        playlist_vec = playlist_vec * 2

    if len(data['songs']) == 0:
        playlist_vec = playlist_vec * 2
    
    song_possible_list = []
    tag_possible_list = []
    
    for every_song, vec in song_vec.items():
        song_possible_list.append((np.linalg.norm(vec - playlist_vec), every_song))
    for every_tag, vec in tags_vec.items():
        tag_possible_list.append((np.linalg.norm(vec - playlist_vec), every_tag))
    
    song_possible_list.sort()
    tag_possible_list.sort()
    tdata = {'id':data['id']}
    tdata['songs'] = [int(song[1]) for song in song_possible_list[0:100]]
    tdata['tags'] = [tag[1] for tag in tag_possible_list[0:10]]
    print(data['id'], "th is done,")
    return tdata

def main():
    executor = concurrent.futures.ProcessPoolExecutor(20)
    futures = [executor.submit(find_and_append, item) for item in target_data[0:50]]
    concurrent.futures.wait(futures)
    result = [fu.result() for fu in futures]


    f4 = open('evaluated_set.json', 'w', encoding='utf-8')
    json.dump(result, f4, ensure_ascii=False)

if __name__ == '__main__':
    main()

    

    
        

            

    
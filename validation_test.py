import numpy as np
import json

import concurrent.futures
from time import time
import itertools
    
result = []
with open('song_vec', 'r', encoding='utf-8') as f1:
        song_vec1 = json.load(f1)

with open('tag_vec', 'r', encoding='utf-8') as f2:
    tag_vec1 = json.load(f2)

song_vec = {song : np.asarray(vec) for song, vec in song_vec1.items()}
song_exi = {song : True for song, vec in song_vec1.items()}
tags_vec = {song : np.asarray(vec) for song, vec in tag_vec1.items()}
tags_exi = {song : True for song, vec in tag_vec1.items()}


test_set = []
answers_set = []

with open("testset.txt", 'r', encoding='utf-8') as f3:
    while True:
        txt = f3.readline()
        if txt == '':
            break
        loc_tag = txt.strip().split(',')
        loc_tag.pop()
        
        loc_song = f3.readline().strip().split(',')
        loc_song.pop()
        if len(loc_song) < 70:
            continue
        test_set.append(loc_song[0:20])
        answers_set.append((loc_tag, loc_song[20:]))


LapEig = 4
j = 0
 

def find_and_append(data, song_extend_len, tag_extend_len):

    playlist_vec = np.zeros(LapEig)
    i = 0
    for song in data:
        if song_exi.get(song) == None:
            continue
        else:
            i += 1
            playlist_vec = playlist_vec + song_vec[song]
    if i != 0:
        playlist_vec = playlist_vec/i
    
    
    song_possible_list = []
    tag_possible_list = []
    
    for every_song, vec in song_vec.items():
        song_possible_list.append((np.linalg.norm(vec - playlist_vec), every_song))
    for every_tag, vec in tags_vec.items():
        tag_possible_list.append((np.linalg.norm(vec - playlist_vec), every_tag))
    
    song_possible_list.sort()
    tag_possible_list.sort()
    tdata = {'songs': [], 'tags': []}
    for real, song in song_possible_list:
        if song in data:
            continue
        tdata['songs'].append(song)

    for real, tag in tag_possible_list:
        tdata['tags'].append(tag)
    
    return tdata




for i in range(len(test_set)):
    truth_answer = answers_set[i]
    input_data = test_set[i]
    answer = find_and_append(input_data, len(truth_answer[0]), len(truth_answer[1]))
    right_song_ctr = 0
    right_tag_ctr = 0
    for song in answer['songs']:
        
        if song in truth_answer[1]:
            right_song_ctr += 1

    for tag in answer['tags']:
        if tag in truth_answer[0]:
            right_tag_ctr += 1
    
    print(i, "th playlist: ", right_song_ctr/(len(truth_answer[1])), right_tag_ctr/(len(truth_answer[0])))


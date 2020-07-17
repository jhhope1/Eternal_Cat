import json
training = []
with open("train.json", 'r', encoding='utf-8') as f1:
    training_set = json.load(f1)
    for data in training_set:
        training.append((data['songs'], data['tags']))

song_to_idx = {}
tag_to_idx = {}

with open("song_to_idx.json", 'r', encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
with open("tag_to_idx.json", 'r', en    coding='utf-8') as f2:
    tag_to_idx = json.load(f2)


def batch_of(range_): ##(0,256) (234, 1234)
    vector_list = []
    for i in range(range_[0], range_[1]):
        songs, tags = training[i]
        playlist_vec = [0 for i in range(12393)]
        for song in songs:
            if song_to_idx.get(song) != None:
                playlist_vec[song_to_idx[song]] = 1
        for tag in tags:
            if tag_to_idx.get(tag) != None:
                playlist_vec[tag_to_idx[tag]] = 1
        vector_list.append(playlist_vec)
    return playlist_vec

print(batch_of((0,1)))
        
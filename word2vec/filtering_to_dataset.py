import json
import random

song_freqs = {}
tag_freqs = {}
bounded_freq_song = 30
bounded_freq_tag = 15
vector_lists = []
vector_trainlists = []
idx_to_item = []
song_genre_map = {}
genre_to_idx = {}
dim = 31202

with open("data/train.json", "r", encoding="utf-8") as f1:
    data = json.load(f1)
    vector_size = 0
    song_to_idx = {}
    tag_to_idx = {}

    for playlist in data:
        for tag in playlist['tags']:
            if tag_freqs.get(tag) == None:
                tag_freqs[tag] = 1
            else:
                tag_freqs[tag] += 1
        for song in playlist['songs']:
            if song_freqs.get(song) == None:
                song_freqs[song] = 1
            else:
                song_freqs[song] += 1
        
    for song, freq in song_freqs.items():
        if freq > bounded_freq_song:
            song_to_idx[int(song)] = vector_size
            song_genre_map[int(song)] = []
            idx_to_item.append(int(song))
            vector_size += 1
    
    

    print(vector_size)
    for tag, freq in tag_freqs.items():
        if freq > bounded_freq_tag:
            tag_to_idx[tag] = vector_size
            idx_to_item.append(tag)
            vector_size += 1
    print(vector_size, " is size of this vector")
    with open("data/song_meta.json", "r", encoding = 'utf-8') as f2:
        songmap = json.load(f2)
        for song, idx in song_to_idx.items():
            for genre in songmap[song]["song_gn_gnr_basket"]:
                song_genre_map[song].append(genre)
            #for genre in songmap[song]["song_gn_dtl_gnr_basket"]:
                #song_genre_map[song].append(genre)
    '''
    with open("genre_gn_all.json", "r", encoding="utf-8") as f3:
        genremap = json.load(f3)
        for gn, _ in genremap.items():
            genre_to_idx[gn] = vector_size
            idx_to_item.append(gn)
            vector_size += 1
    print(vector_size, " is size of this vector")'''
    wordset = {}
    with open("wordset.json", 'r', encoding = 'utf-8') as f2:
        wordset = json.load(f2)
    

    for idx2, playlist in enumerate(data):
        tmp_list = []
        bows = []
        for song in playlist['songs']:
            if song_to_idx.get(song) != None:
                tmp_list.append(song_to_idx[song])
                

        for tag in playlist['tags']:
            if tag_to_idx.get(tag) != None:
                tmp_list.append(tag_to_idx[tag])
        
        title = playlist['plylst_title']
        wordset_idx = []
        for idx, word in enumerate(wordset):
            if word in title:
                wordset_idx.append(idx)
        
        if idx2%10000 == 0:
            print(idx2)
        
        vector_lists.append({"items": tmp_list,"wordset_idx":wordset_idx})
        
    
    print(len(vector_lists))


with open("playlist_idxs.json", 'w') as f2:
    json.dump(vector_lists, f2)

with open("tag_to_idx.json", "w", encoding = 'utf-8') as f3:
    json.dump(tag_to_idx, f3, ensure_ascii=False)

with open("song_to_idx.json", "w", encoding = 'utf-8') as f3:
    json.dump(song_to_idx, f3, ensure_ascii=False)

with open("idx_to_item.json", "w", encoding = 'utf-8') as f3:
    json.dump(idx_to_item, f3, ensure_ascii=False)

with open("gerne_to_idx.json", "w", encoding = 'utf-8') as f3:
    json.dump(genre_to_idx, f3, ensure_ascii=False)

with open("song_genre_map.json", "w", encoding = 'utf-8') as f3:
    json.dump(song_genre_map, f3, ensure_ascii = False)
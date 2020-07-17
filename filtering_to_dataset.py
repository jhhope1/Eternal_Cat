import json
song_freqs = {}
tag_freqs = {}
bounded_freq_song = 30
bounded_freq_tag = 15
vector_lists = []
idx_to_item = []
song_genre_map = {}
genre_to_idx = {}


with open("train.json", "r", encoding = 'utf-8') as f1:
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
            song_to_idx[song] = vector_size
            song_genre_map[song] = []
            idx_to_item.append(song)
            vector_size += 1
    
    

    print(vector_size)
    for tag, freq in tag_freqs.items():
        if freq > bounded_freq_tag:
            tag_to_idx[tag] = vector_size
            idx_to_item.append(tag)
            vector_size += 1
    print(vector_size, " is size of this vector")
    with open("song_meta.json", "r", encoding = 'utf-8') as f2:
        songmap = json.load(f2)
        for song, idx in song_to_idx.items():
            for genre in songmap[song]["song_gn_gnr_basket"]:
                song_genre_map[song].append(genre)
            #for genre in songmap[song]["song_gn_dtl_gnr_basket"]:
                #song_genre_map[song].append(genre)
    
    with open("genre_gn_all.json", "r", encoding="utf-8") as f3:
        genremap = json.load(f3)
        for gn, _ in genremap.items():
            genre_to_idx[gn] = vector_size
            idx_to_item.append(gn)
            vector_size += 1
    print(vector_size, " is size of this vector")


    playlist_limit = 2000
    cnt = 0
    for playlist in data:
        cnt += 1
        if cnt > playlist_limit:
            break
        playlist_vector = [0 for i in range(vector_size)]
        for song in playlist['songs']:
            if song_freqs[song] > bounded_freq_song:
                playlist_vector[song_to_idx[song]] = 1
        for tag in playlist['tags']:
            if tag_freqs[tag] > bounded_freq_tag:
                playlist_vector[tag_to_idx[tag]] = 1
        vector_lists.append(playlist_vector)

    
with open("playlist_hot_vector.json", 'w') as f2:
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
f1 = open("train.txt", 'r', encoding='utf-8')
f2 = open("test.txt", 'r', encoding='utf-8')
import json
import math

song_to_playlist = {}
song_likelihood = {}
song_genre = {}
song_dtl_genre = {}
song_tagdata = {}


playlist = []

with open("song_meta.json", encoding='utf-8') as f3:
    data = json.load(f3)
    i = 0
    for a in data:
        i += 1
        song_genre[str(a['id'])] = a['song_gn_gnr_basket']
        song_dtl_genre[str(a['id'])] = a['song_gn_dtl_gnr_basket']
        if i%100000 == 0:
            print(song_genre[str(a['id'])])

### preprocess:
idx = 0
while True:
    txt = f1.readline()
    if txt == '':
        break
    tags = txt.strip().split(',')
    tags.pop()
    loclist = f1.readline().strip().split(',')
    f1.readline()
    
    playlist.append(loclist)
    
    for i in loclist:
        if i == '':
            continue

        if song_to_playlist.get(i) == None:
            song_to_playlist[i] = [idx]
            song_likelihood[i] = 0
            if len(tags) == 0:
                song_tagdata[i] = {}
                continue
            song_tagdata[i] = {tag : 1 for tag in tags}

        else:
            song_to_playlist[i].append(idx)
            if len(tags) == 0:
                continue

            for tag in tags:
                if song_tagdata[i].get(tag) != None:
                    song_tagdata[i][tag] += 1
                else:
                    song_tagdata[i][tag] = 1

    idx += 1


song_tag_pq = {}

for idx, tagdata in song_tagdata.items():
    tag_pqs = [(ctr, tags) for tags , ctr in tagdata.items()]
    tag_pqs.sort(reverse=True)
    if len(tag_pqs) > 10:
        song_tag_pq[idx] = [tags for ctr, tags in tag_pqs[0:10]]
    else:
        song_tag_pq[idx] = [tags for ctr, tags in tag_pqs]


def genre_nearness(song1, song2):
    ##곡 자체에 있는 장르들에 대해, 장르가 태그들에 의해 얼마나 서로 가까운지도 판단하자! 즉, 곡들에는 장르에 의한 태그, 플레이리스트에 의한 태그 이렇게 두개가 있다!
    genresim = 1
    for a in song_genre[song1]:
        if a in song_genre[song2]:
            genresim += 1
    tagsim = 1
    for tag in song_tag_pq[song1]:
        if tag in song_tag_pq[song2]:
            tagsim += 1

    
    return tagsim*genresim

    # or just replace this part with node2vec, idont know.



def jaccord(song):
    
    for idx in song_to_playlist[song]:
        lenss = math.log2(len(playlist[idx])) + 1
        for songs in playlist[idx]:
            if song_likelihood.get(songs) == None:
                continue
            if songs == song:
                continue
            song_likelihood[songs] += genre_nearness(song, songs)

while True:
    for idx, item in song_likelihood.items():
        song_likelihood[idx] = 1

    loclist = f2.readline().strip().split(',')
    print(loclist)
    if len(loclist) == 0:
        break

    L = len(loclist)//5
    if L < 10:
        continue
    data_list = loclist[0:L]
    ##print(data_list)
    empty_song = 0
    for song in data_list:
        if song_to_playlist.get(song) == None:
            empty_song += 1
            continue
        jaccord(song)

    #print(song_tag_pq[data_list[0]])

    print(empty_song/len(data_list))
    song_priority = [(ctr, song) for song, ctr in song_likelihood.items()]
    song_priority.sort(reverse=True)
    extended = []
    ubs = len(loclist) - L
    #print(song_priority[0:ubs])
    for ctr, song in song_priority:
        
        if song not in data_list:
            extended.append(song)

        if len(extended) == ubs:
            break
        
    counts = 0
    ctrs = 0
    
    for song in extended:
        ctrs += 1
        if song in loclist:
            counts += 1
    
    

    print(counts/ubs)

    



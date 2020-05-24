f1 = open("train.txt", 'r', encoding='utf-8')
f2 = open("test.txt", 'r', encoding='utf-8')

song_to_playlist = {}
song_likelihood = {}

playlist = []

### preprocess:
idx = 0
while True:
    txt = f1.readline()
    if txt == '':
        break
    loclist = f1.readline().strip().split(',')
    f1.readline()
    
    playlist.append(loclist)
    
    for i in loclist:
        if song_to_playlist.get(i) == None:
            song_to_playlist[i] = [idx]
            song_likelihood[i] = 0

        else:
            song_to_playlist[i].append(idx)

    idx += 1


def jaccord(song):
    if song_to_playlist.get(song) == None:
        return
    for idx in song_to_playlist[song]:
        for songs in playlist[idx]:
            if song_likelihood.get(songs) == None:
                continue
            song_likelihood[songs] += 1

while True:
    for idx, item in song_likelihood.items():
        song_likelihood[idx] = 0
    
    loclist = f2.readline().strip().split(',')
    if len(loclist) == 0:
        break
    L = len(loclist)//5
    data_list = loclist[0:L]
    for song in data_list:
        jaccord(song)

    song_priority = [(ctr, song) for song, ctr in song_likelihood.items()]
    song_priority.sort(reverse=True)
    extended = []
    ubs = len(loclist)
    for idx, song in song_priority:
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

        if ctrs == ubs:
            break
    
    print(extended)
    print(loclist)
    

    print(counts/len(loclist))

    



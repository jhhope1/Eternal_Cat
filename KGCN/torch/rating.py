import json

string = ''
F = open('rating.txt', 'w', encoding='utf-8',)
    

with open('val.json', encoding = 'utf-8') as f:
    data = json.load(f)
    for playlist in data:
        ID = playlist['id']
        for song in playlist['songs']:
            F.write( str(ID) + '\t' + str(song) + '\t1\n')

with open('train.json', encoding = 'utf-8') as f:
    data = json.load(f)
    for playlist in data:
        ID = playlist['id']
        for song in playlist['songs']:
            F.write(str(ID) + '\t' + str(song) + '\t1\n')



import json
import random

i = 0

with open("train.json", encoding = 'utf-8') as json_file, open('train.txt','w', encoding='utf-8') as FILE, open('testset.txt', 'w', encoding='utf-8') as FILE2:
    data = json.load(json_file)

    for a in data:
        i += 1
        # if i%10 == 0:
        #     st = ''
        #     for s in a['songs']:
        #         st += str(s) + ','
        #     st += '\n'
        #     FILE.write(st)
        #     st = ''
        #     for s in a['tags']:
        #         st += s + ','
        #     st += '\n'
        #     FILE2.write(st)
        
        st = ''
        for s in a['tags']:
            st += s + ','
        st += '\n'
        for s in a['songs']:
            st += str(s) + ','
        st += '\n'
        st += str(a['like_cnt']) + '\n'

        if i%10 == 0:
            FILE2.write(st)
        else:
            FILE.write(st)

            

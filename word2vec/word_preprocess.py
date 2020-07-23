import json
import re
from collections import Counter
from typing import *
words = {}
bound = 50
candi = set()
candi_ = {}
def get_variation(word):
    vari = []
    for idx1 in range(1,len(word)):
        vari.append(word[:idx1])
    for idx1 in range(len(word)-1):
        vari.append(word[idx1:])
    #print(vari)
    return vari

with open("data/train.json", 'r', encoding='utf-8') as f1, open('data/val.json', 'r', encoding='utf-8') as f2:

    data = json.load(f1)
    data.extend(json.load(f2))
    for playlist in data:
        text = playlist['plylst_title']
        #text = re.sub('[0-9]+', '', title)
        text = re.sub('[ㄱ-ㅎ]+', '', text)
        text = re.sub('[-=+,#/\?:^$•▶.♬@*★_\"※~&%ㆍ·!』\\‘’|\(\)\[\]\<\>`\'…》]', '', text)
        tlist = text.split()
        #print(tlist)
        #tlist = playlist['plylst_title'].replace().replace(pat=r'[^\w\s]', repl=r'', regex=True).replace(pat=r'[ ]{2,}', repl=r' ', regex=True).replace(pat=r'[\u3000]+', repl=r'', regex=True).split(' ')
        for word_ in tlist:

            if len(word_) > 10:
                continue
            for word in get_variation(word_):
                if words.get(word) is not None:
                    words[word] += 1
                else:
                    words[word] = 1
            



    for word, cnt in words.items():
        if cnt > bound and len(word) < 8:
            candi.add(word)

    candi_ = candi.copy()
    
    
    print(candi)
    candi = {word : idx for idx, word in enumerate(candi)}

with open("wordset.json", 'w', encoding='utf-8') as f1:
    json.dump(candi, f1, ensure_ascii=False)

print(len(candi))
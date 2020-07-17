import json
import re
from collections import Counter
from typing import *
words = {}
bound = 10
candi = set()
candi_ = {}
def get_variation(word):
    vari = []
    for idx1 in range(len(word)-1):
        for idx2 in range(idx1, len(word)-1):
            vari.append(word[idx1:idx2])
    return vari

with open("data/train.json", 'r', encoding='utf-8') as f1, open('data/val.json', 'r', encoding='utf-8') as f2:

    data = json.load(f1)
    data.extend(json.load(f2))
    for playlist in data:
        tlist = re.sub('\W+',' ', playlist['plylst_title']).split()
        #tlist = playlist['plylst_title'].replace().replace(pat=r'[^\w\s]', repl=r'', regex=True).replace(pat=r'[ ]{2,}', repl=r' ', regex=True).replace(pat=r'[\u3000]+', repl=r'', regex=True).split(' ')
        for word_ in tlist:
            for word in get_variation(word_):
                if words.get(word) is not None:
                    words[word] += 1
                else:
                    words[word] = 1
            

    for word, cnt in words.items():
        if cnt > bound and len(word) < 11 and len(word) > 1:
            candi.add(word)

    candi_ = candi.copy()
    for word in candi_:
        for word_ in get_variation(word):
            if word_ in candi and word_ != word:
                try:
                    candi.remove(word_)
                except KeyError:
                    continue
    
    print(candi)
    candi = {word : idx for idx, word in enumerate(candi)}

with open("wordset.json", 'w', encoding='utf-8') as f1:
    json.dump(candi, f1, ensure_ascii=False)

print(len(candi))
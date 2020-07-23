import json
import os
import re

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

with open(os.path.join(data_path, "train.json"), 'r', encoding='utf-8') as f1:
    train_data = json.load(f1)

with open(os.path.join(data_path, "tag_to_idx.json"), 'r', encoding='utf-8') as f2:
    tag_to_idx_temp = json.load(f2)
    tag_to_idx = {}
    for tag, val in tag_to_idx_temp.items():
        tag_to_idx[tag.lower()] = val

title_keylist = [pl['plylst_title'].lower() for pl in train_data]

#Enumerate substrings
tag_occur = [[] for _ in range(len(title_keylist))]
alphanumeric = 'qwertyuiopasdfghjklzxcvbnm1234567890'
for i, title in enumerate(title_keylist):
    title_trimmed = re.sub('[ .,?!#_$%^&*}{)(]', '', title)
    n = len(title_trimmed)
    s = 0
    while s < n:
        for e in range(n, s, -1):
            title_sub = title_trimmed[s:e]
            if tag_to_idx.get(title_sub):
                if e == s + 1 and title_sub in alphanumeric:
                    break
                if e == s + 1 and s > 0 and e < n:
                    for exc_char in ' ][}{/.,?! #-_&)(+=@$❤':
                        if exc_char + title_sub in title or title_sub + exc_char in title:
                            tag_occur[i].append(tag_to_idx[title_sub])
                            break
                    break 
                tag_occur[i].append(tag_to_idx[title_sub])
                s = e - 1
                break
        s += 1

with open(os.path.join(data_path, 'tag_occur.json'), 'w', encoding='utf-8') as f3:
    json.dump(tag_occur, f3, ensure_ascii=False)

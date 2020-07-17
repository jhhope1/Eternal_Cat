import json
with open("data/train.json", 'r', encoding='utf-8') as f1, open('data/val.json', 'r', encoding='utf-8') as f2:
    data = json.load(f1)
    for ply in data:
        if '아아아아' in ply['plylst_title']:
            print('!!')
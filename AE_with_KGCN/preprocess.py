import os
import json
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data\\')
entity_map = {}
relation_map = {}
for line in open(DATA+"kg.txt", 'r', encoding='utf-8'):
    array = line.strip().split(' ')
    head_old = array[0]
    relation_old = array[1]
    tail_old = array[2]
    if tail_old not in entity_map:
        entity_map[tail_old] = len(entity_map)
    if relation_old not in relation_map:
        relation_map[relation_old] = len(relation_map)
print(len(entity_map), len(relation_map))
with open(DATA+'entity_to_idx.json', 'w', encoding='utf-8') as f2:
    json.dump(entity_map, f2)
with open(DATA+'relation_to_idx.json', 'w', encoding='utf-8') as f3:
    json.dump(relation_map,f3)

import json
import os, sys
import re

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from res_AE import const
data_path = const.data_path

with open(os.path.join(data_path, "train.json"), 'r', encoding='utf-8') as f1:
    train_data = json.load(f1)
with open(os.path.join(data_path, "val.json"), 'r', encoding='utf-8') as f:
    train_data += json.load(f)
with open(os.path.join(data_path, "test.json"), 'r', encoding='utf-8') as f:
    train_data += json.load(f)

with open(os.path.join(data_path, "song_meta.json"), 'r', encoding='utf-8') as f3:
    song_meta = json.load(f3)

with open(os.path.join(data_path, 'genre_gn_all.json'), 'r', encoding='utf-8') as f:
    genre_to_name = json.load(f)

garbage = u'.,?!#_$%^&*}{( \t][-;:)/:‘’\"\'+~:˸'

def is_korean(s):
    for c in s:
        if (0xac00 <= ord(c) <= 0xd7af) or (0xf900 <= ord(c) <= 0xfaff): pass
        else: return False
    return True

def is_kanji_or_kana(s):
    for cjk in s:
        if (0x4e00 <= ord(cjk) <= 0x9fff) or (0x3041 <= ord(cjk) <= 0x3096) or (0x30a0 <= ord(cjk) <= 0x30ff): pass
        else: return False
    return True

class Chunker(object):
    def __init__(self):
        super(Chunker, self).__init__()
        self.chunk = dict()
        self.chunk_idx = dict() #trimmed chunk idxmap
    def __len__(self):
        return len(self.chunk_idx)
    def add_key(self, key):
        if self.chunk.get(key):
            self.chunk[key] += 1
        else:
            self.chunk[key] = 1
    def do_filter(self, threshold = 10):
        self.chunk_idx = dict()
        for key, val in self.chunk.items():
            if val >= threshold:
                self.chunk_idx[key] = len(self.chunk_idx)

    def gen_chunks(self, target_string):
        name_trimmed = ''.join([c for c in target_string.lower() if c.isalpha()])
        for chunk_len in range(3, 5):
            for i in range(max(0, len(name_trimmed) - chunk_len)):
                chunk = name_trimmed[i:i+chunk_len]
                self.add_key(chunk)
        for korean_chunk_len in range(1, 3):
            for i in range(max(0, len(name_trimmed) - korean_chunk_len)):
                chunk = name_trimmed[i:i+korean_chunk_len]
                if is_korean(chunk):
                    self.add_key(chunk)
        for cjk in name_trimmed:
            if is_kanji_or_kana(cjk):
                self.add_key(cjk)
        
    def get_basket(self, target_string):
        name_trimmed = ''.join([c for c in target_string.lower() if c.isalpha()])
        ret_idxset = set()

        for chunk_len in range(3, 5):
            for i in range(max(0, len(name_trimmed) - chunk_len + 1)):
                chunk = name_trimmed[i:i+chunk_len]
                if self.chunk_idx.get(chunk):
                    ret_idxset.add(self.chunk_idx[chunk])
        for korean_chunk_len in range(1, 3):
            for i in range(max(0, len(name_trimmed) - korean_chunk_len + 1)):
                chunk = name_trimmed[i:i+korean_chunk_len]
                if is_korean(chunk):
                    if self.chunk_idx.get(chunk):
                        ret_idxset.add(self.chunk_idx[chunk])
        for cjk in name_trimmed:
            if is_kanji_or_kana(cjk):
                if self.chunk_idx.get(cjk):
                    ret_idxset.add(self.chunk_idx[cjk])
        return ret_idxset

song_chunker = Chunker()


for song in song_meta:
    song_chunker.gen_chunks(song['song_name'])
    if song.get('album_name'):
        song_chunker.gen_chunks(song['album_name'])
    for genre in song['song_gn_gnr_basket']:
        if not genre_to_name.get(genre):
            continue
        song_chunker.gen_chunks(genre_to_name[genre])
    for genre in song['song_gn_dtl_gnr_basket']:  
        if not genre_to_name.get(genre):
            continue  
        song_chunker.gen_chunks(genre_to_name[genre])

for pl in train_data:
    for tag in pl['tags']:
        song_chunker.gen_chunks(tag)
    song_chunker.gen_chunks(pl['plylst_title'])

song_chunker.do_filter(threshold = 100)
print(len(song_chunker))

song_to_chunkidx = dict()
tag_to_chunkidx = dict()
title_to_chunkidx = dict()

for song in song_meta:
    chunkset = song_chunker.get_basket(song['song_name'])
    if song.get('album_name'):
        chunkset = chunkset.union(song_chunker.get_basket(song['album_name']))
    for genre in song['song_gn_gnr_basket']:
        if not genre_to_name.get(genre):
            continue
        chunkset = chunkset.union(song_chunker.get_basket(genre_to_name[genre]))
    for genre in song['song_gn_dtl_gnr_basket']:   
        if not genre_to_name.get(genre):
            continue 
        chunkset = chunkset.union(song_chunker.get_basket(genre_to_name[genre]))
    song_to_chunkidx[song['id']] = list(chunkset)

for pl in train_data:
    for tag in pl['tags']:
        if not tag_to_chunkidx.get(tag):
            tag_to_chunkidx[tag] = list(song_chunker.get_basket(tag))
    
    if not title_to_chunkidx.get(pl['plylst_title']):
        title_to_chunkidx[pl['plylst_title']] = list(song_chunker.get_basket(pl['plylst_title']))

with open(os.path.join(data_path,'object_to_chunkidx.json'), 'w', encoding='utf-8') as f:
    json.dump(song_to_chunkidx, f, ensure_ascii=False)

with open(os.path.join(data_path, 'tag_to_chunkidx.json'), 'w', encoding='utf-8') as f:
    json.dump(tag_to_chunkidx, f, ensure_ascii=False)

with open(os.path.join(data_path, 'title_to_chunkidx.json'), 'w', encoding='utf-8') as f:
    json.dump(title_to_chunkidx, f, ensure_ascii=False)
import json
import os
import torch
import time #just for benchmark

#input_dim = 100058
output_dim = 57229
song_size = 0
tag_size = 0
entity_size = 0
song_to_idx = {}
tag_to_idx = {}
idx_to_item = []

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 512 #adjust up to your memory limit
use_meta = True #toggle if you don't want to use meta
use_ply_meta = True

with open(os.path.join(data_path, "song_to_idx.json"), 'r', encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
    song_size = len(song_to_idx)

song_to_idx_keyset = set(song_to_idx.keys())

with open(os.path.join(data_path, "tag_to_idx.json"), 'r', encoding='utf-8') as f2:
    tag_to_idx = json.load(f2)
    tag_size = len(tag_to_idx)

tag_to_idx_keyset = set(tag_to_idx.keys())

with open(os.path.join(data_path, "idx_to_item.json"), 'r', encoding='utf-8') as f3:
    idx_to_item = json.load(f3)


with open(os.path.join(data_path,"res_song_to_entityidx.json"), 'r', encoding='utf-8') as f3:
    song_to_entityidx = json.load(f3)

with open(os.path.join(data_path,"res_entity_to_idx.json"), 'r', encoding='utf-8') as f3:
    entity_to_idx = json.load(f3)
    entity_size = len(entity_to_idx)


with open(os.path.join(data_path,"res_letter_to_idx.json"), 'r', encoding='utf-8') as f5:
    letter_to_idx = json.load(f5)
    l_num = len(letter_to_idx)

letter_to_idx_keyset = set(letter_to_idx.keys())
song_to_entityidx_key_set = set(song_to_entityidx.keys())

import model_inference as mi

onehot_len = output_dim + entity_size if use_meta else output_dim
onehot_len = onehot_len + l_num if use_ply_meta else onehot_len
song_add = 100
tag_add = 10

eps = torch.tensor(0.1).to(device)
def zero_one_normalize(x: torch.Tensor): #[eps,1] normalize
    return x + eps

val_file = os.path.join(data_path, "val.json")
res_file = os.path.join(data_path, "results.json")
open_utf8 = lambda name, mode: open(name, mode, encoding='utf-8')

with open_utf8(val_file, 'r') as f3, open_utf8(res_file, 'w') as f4:
    time_start = time.time()

    ans_list = []
    val_list = json.load(f3)
    print('length of validation set: ', len(val_list))
    for st in range(0, len(val_list), batch_size):
        batch_len = min(batch_size, len(val_list) - st)
        ed = st + batch_len
        print('loading batch... [', st, ',', ed, ')')
        input_one_hot = torch.zeros(batch_len, onehot_len)
        mask = torch.ones(batch_len, output_dim) #selector for topk

        for j in range(st, ed):
            #playlist = val_list[j]
            #song extraction and tensorization
            noise_input_song = []
            song_set = set(val_list[j]['songs'])
            plylst_title_list = list(val_list[j]['plylst_title'])
            noise_input_song = list(song_set)
            song_set = {str(song) for song in song_set} #convert to string
            song_list_refined = list(song_set.intersection(song_to_idx_keyset))
            song_idxlist = [song_to_idx[sname] for sname in song_list_refined]
            input_one_hot[j - st][song_idxlist] = 1
            mask[j - st][song_idxlist] = 0

            if use_ply_meta:
                l_list_idxlist = [output_dim + letter_to_idx[lname] for lname in plylst_title_list if lname in letter_to_idx]
                input_one_hot[j - st][l_list_idxlist] += 1

            if use_meta:
                song_key_list_refined = list(song_set.intersection(song_to_entityidx_key_set))
                entity_idxlist = []
                for sname in song_key_list_refined:
                    entity_idxlist += [output_dim + l_num + entity for entity in song_to_entityidx[sname]] 
                #replacing concatenation; depends on output_dim
                input_one_hot[j - st][entity_idxlist] += 1
            
            #tag extraction and tensorization
            tag_set = set(val_list[j]['tags'])
            tag_list_refined = list(tag_set.intersection(tag_to_idx_keyset))
            tag_idxlist = [tag_to_idx[tname] for tname in tag_list_refined]
            input_one_hot[j - st][tag_idxlist] = 1
            mask[j - st][tag_idxlist] = 0
        #Inference start
        inferenced = mi.inference(input_one_hot.to(device))
        infer_normalized = zero_one_normalize(inferenced)
        infer_masked = infer_normalized * mask.to(device)

        song_masked = infer_masked.narrow(dim = 1, start = 0, length = song_size).to(device)
        tag_masked  = infer_masked.narrow(dim = 1, start = song_size, length = tag_size).to(device)

        song_indices = torch.topk(song_masked, song_add, dim = 1)[1].to('cpu').numpy()
        tag_indices  = torch.topk(tag_masked, tag_add, dim = 1)[1].to('cpu').numpy()
        tag_indices = tag_indices + song_size #correcting to the indices for idx_to_item

        for j in range(st, ed):
            loc_song = song_indices[j - st]
            loc_tag =  tag_indices[j - st]
            loc_song = [int(idx_to_item[idx]) for idx in loc_song]
            loc_tag = [idx_to_item[idx] for idx in loc_tag]
            inf_ans = {}
            inf_ans['id'] = val_list[j]['id']
            inf_ans['songs'] = loc_song
            inf_ans['tags'] = loc_tag
            ans_list.append(inf_ans)
            if j % 10000 == 0:
                print('no. ', j)
                print(inf_ans['songs'])
                print(inf_ans['tags'])
    json.dump(obj=ans_list, fp=f4, ensure_ascii=False)
    print(time.time() - time_start, 'seconds took')

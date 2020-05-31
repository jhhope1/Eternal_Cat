import torch
import numpy as np
import pandas as pd
import json
import vecs_validity
from vecs_validity import Net
cuda0 = torch.device('cuda:0')
model_PATH = './vecs_validity_net.pth'

vec_num = 20
batch_size = 1000
dim_vec = 10
song_num = 707989
song_list = []

net = Net()
net.load_state_dict(torch.load(model_PATH))
net.eval()
net.to(device = torch.device(cuda0))


test_PATH = 'train.json'        #change to 'test.json'

S = set()
with open(test_PATH, 'r') as f:
    train = pd.read_json('train.json', typ = 'frame')
    test_loc_list = train['songs']
    for L in test_loc_list:
        for song in L:
            S.add(str(song))
    for i in S:
        song_list.append(i)
del S

with open('NN\data.json') as json_file:
    data = json.load(json_file)
    song_vec = data['song_vecs']

valid_test_num = 0

batch_size = 100
mini_batch = np.empty((batch_size, vec_num, dim_vec))
mini_batch[0:150%batch_size+1,:,:]

def f(A):
    default_song_vec = np.array([song_vec[idx] for idx in A])
    re = []
    batch_size = 100000
    mini_batch = np.empty((batch_size, vec_num, dim_vec))
    for i, song in enumerate(song_list):
        NA = np.array(default_song_vec)
        NA = np.vstack((NA,np.array(song_vec[str(song)])))
        mini_batch[i%batch_size] = NA
        if i % batch_size == batch_size-1:
            print(i)
            with torch.no_grad():
                re.append(net(torch.from_numpy(mini_batch).type(torch.cuda.FloatTensor)).cpu().numpy().flatten().tolist())
            continue
        if i == len(song_list)-1:
            with torch.no_grad():
                re.append(net(torch.from_numpy(mini_batch[0:i%batch_size+1,:,:]).type(torch.cuda.FloatTensor)).cpu().numpy().flatten().tolist())
    output = []
    for L in re:
        for a in L:
            output.append(a)
    output = np.array(output)
    indices = output.argsort()[-(vec_num+100+1):][::-1]
    predict = []

    for i in indices:
        if str(song_list[i]) in A:
            continue
        predict.append(str(song_list[i]))
        if(len(predict)==100):
            break
    return predict

for songs in test_loc_list:
    if len(songs)>vec_num:
        valid_test_num += 1
        for i in range(len(songs)):
            songs[i] = str(songs[i])
        extracted_list = songs[0:vec_num-1]
        predict = f(extracted_list)
        correct = 0
        for pre in predict:
            if pre in songs:
                correct += 1
        print("corrct, total, probability = ",correct, len(songs)-20, correct/(len(songs)-20))
        print(len(predict),len(songs))
        print("songs = ",songs)
        print("predict = ",predict)

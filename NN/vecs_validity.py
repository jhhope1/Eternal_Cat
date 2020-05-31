import torch
import torchvision
import torch.utils.data as Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import pandas as pd
import json
cuda0 = torch.device('cuda:0')
model_PATH = './vecs_validity_net.pth'

vec_num = 20
batch_size = 1000
dim_vec = 10
song_num = 707989

def gen_random_playlist():
    return np.array([[random.randint(0,song_num) for n in range(random.randint(0,100))] for m in range(1000)])
def gen_random_song_vec():
    return np.random.randn(song_num,dim_vec).tolist()

def load_playlist():
    train = pd.read_json('train.json', typ = 'frame')
    plylst_song = train['songs']
    return plylst_song.tolist()

def load_song_vec():
    with open('song_vec.json','r') as f:
        song_vec_dict = json.load(f)
    return song_vec_dict

def load_playlist_song_vec():
    data_path = 'NN\data.json'
    playlist = []
    song_vec = []
    if os.path.isfile(data_path) and os.path.getsize(data_path) > 0: #load
        with open(data_path) as json_file:
            data = json.load(json_file)
            song_vec = data['song_vecs']
            playlist = data['playlist']
    else:                                                            #generate
        data = {}

        #song_vec = gen_random_song_vec()
        #playlist = gen_random_playlist()

        song_vec = load_song_vec()
        playlist = load_playlist()

        data['song_vecs'] = song_vec
        data['playlist'] = playlist

        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return [playlist, song_vec]

def rand_invalid_list():
    #samp = np.random.randint(300,size = vec_num)
    A = []
    while len(A)<vec_num:
        while True:
            s = str(random.randint(1,song_num))
            if s in song_vec:
                A.append(song_vec[s])
                break
    return [A,0]

def rand_valid_list():
    while True:
        PL = random.choice(playlist)
        if len(PL)>=vec_num:
            samp = random.sample(PL, k=vec_num)
            A = []
            for key in samp:
                A.append(song_vec[str(key)])
            return [A,1]


def gen_batch():
    gen_list = [rand_valid_list,rand_invalid_list]
    A = [np.empty((batch_size,vec_num,dim_vec)),np.empty((batch_size,1))]
    for i in range(batch_size):
        B = random.choice(gen_list)()
        A[0][i] = B[0]
        A[1][i] = B[1]
    return A



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_vec*vec_num, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 840)
        self.bn2 = nn.BatchNorm1d(840)
        self.fc3 = nn.Linear(840,1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1,dim_vec*vec_num)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(epoch_num):
    net = Net()
    net.to(cuda0)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(epoch_num):
        running_loss = 0.0
        for i in range(int(len(playlist)/batch_size)):
            data = gen_batch()
            inputs, labels = data
            inputs = torch.from_numpy(inputs).to(device = cuda0).type(torch.cuda.FloatTensor)
            labels = torch.from_numpy(labels).to(device = cuda0).type(torch.cuda.FloatTensor)
            
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch<10:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss))
        if epoch%10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss))

    print('Finished Training')
    torch.save(net.state_dict(), model_PATH)



def test():
    net = Net()
    net.load_state_dict(torch.load(model_PATH))
    net.eval()
    net.to(device = torch.device('cpu'))
    correct = 0
    total = 0
    testN = 10
    with torch.no_grad():
        for i in range(testN):
            data = gen_batch()
            images, labels = data

            images = torch.from_numpy(images).to(device = 'cpu').type(torch.FloatTensor)
            labels = torch.from_numpy(labels).to(device = 'cpu').type(torch.FloatTensor)

            outputs = net(images)
            predicted = outputs.data.round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %d %%' % (
        100 * correct / total))

if __name__=='__main__':
    playlist , song_vec = load_playlist_song_vec()
    train(10)
    test()
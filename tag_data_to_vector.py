f1 = open("train.txt", 'r', encoding='utf-8')

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding
import numpy as np
import json

tag_weighted_graph = {}
tag_exi_checker = {}

i = 0
song_nameset = {}
with open("song_meta.json", encoding='utf-8') as f3:
    data = json.load(f3)
    i = 0
    for a in data:
        i += 1
        song_nameset[str(a['id'])] = a['song_name']
        if i%100000 == 0:
            print(a['song_name'])

while True:
    txt = f1.readline()
    if txt == '':
        break
    tags = txt.strip().split(',')
    tags.pop()
    f1.readline()
    f1.readline()
    for a in tags:
        if tag_exi_checker.get(a) == None:
            tag_exi_checker[a] = 1
            tag_weighted_graph[a] = {}

            for b in tags:
                if tag_weighted_graph[a].get(b) == None:
                    tag_weighted_graph[a][b] = 1
                else:
                    tag_weighted_graph[a][b] += 1
        else:
            tag_exi_checker[a] += 1
            for b in tags:
                if tag_weighted_graph[a].get(b) == None:
                    tag_weighted_graph[a][b] = 1
                else:
                    tag_weighted_graph[a][b] += 1


print(len(tag_exi_checker))

tag_ctr_data = [(ctr, tag) for tag, ctr in tag_exi_checker.items()]
tag_ctr_data.sort(reverse = True)
SZ = 1000
tag_pq= [tag for ctr, tag in tag_ctr_data[0:SZ]]

tag_pq_exi = {tag_pq[idx]: True for idx in range(SZ)}

# for ctr, tag in tag_ctr_data[0:SZ]:
#     #print(tag, ctr)

aff_mat = [[0]*SZ for i in range(SZ)]

for i in range(SZ):
    tag1 = tag_pq[i]
    if tag_weighted_graph.get(tag1) == None:
        aff_mat[i][j] = 0
        continue

    for j in range(SZ):
        tag2 = tag_pq[j]
        
        if tag_weighted_graph[tag1].get(tag2) == None:
            aff_mat[i][j] = 0
            continue

        aff_mat[i][j] = 1 - 1/(tag_weighted_graph[tag1][tag2] + 1)


lapEig = 4

embedding = SpectralEmbedding(n_components = lapEig, affinity = 'precomputed')

X_transformed = np.array(embedding.fit_transform(aff_mat))
print(X_transformed[0:100])

tag_vec = {tag_pq[idx] : X_transformed[idx] for idx in range(len(tag_pq))} ### tag_vec["띵곡"] = [1.2, ...]  := numpy array

#### song to vector

f1 = open("train.txt", 'r', encoding='utf-8')

song_vec = {}
song_ctr = {}
i = 0
while True:
    i += 1
    txt = f1.readline()
    if txt == '':
        break
    tags = txt.strip().split(',')
    tags.pop()

    song_list = f1.readline().strip().split(',')
    song_list.pop()
    f1.readline()
    for tag in tags:
        if tag_pq_exi.get(tag) == None:
            continue
        t_vec = tag_vec[tag]
        for song in song_list:

            if song_ctr.get(song) == None:
                song_vec[song] = t_vec
                song_ctr[song] = 1 
            else:
                song_vec[song] = song_vec[song] + t_vec
                song_ctr[song] += 1

f1.close()

i = 0
with open('song_vec.json', 'w', encoding='utf-8') as f5:
    song_vec_as_list = {}
    for song, vec in song_vec.items():
        vec_list = vec.tolist()
        song_vec_as_list[song] = vec_list
    json.dump( song_vec_as_list, f5, ensure_ascii=False)

with open('tag_vec.json', 'w', encoding='utf-8') as f6:
    tag_vec_as_list = {}
    for tag, vec in tag_vec.items():
        tag_vec = vec.tolist()
        tag_vec_as_list[tag] = tag_vec

    json.dump(tag_vec_as_list, f6, ensure_ascii=False)



print("song to vec is done... normalizing...")



song_name = []
print(X_transformed.shape)
song_vec_size = len(song_vec)
song_vec_list = []
for song in song_vec:
    song_vec[song] /= song_ctr[song]
    if i%1000 == 0:
        song_vec_list.append(song_vec[song])
        song_name.append(song_nameset[song])
    i += 1
    if i%10000 == 0:
        print(i," th song is done... ","out of ",song_vec_size)
print(np.array(song_vec_list).reshape(-1,lapEig))
print(np.array(song_vec_list))
song_vec_list_vec = np.asarray(song_vec_list).reshape(-1,lapEig)
X_transformed = np.concatenate([X_transformed, song_vec_list_vec],axis = 0)
print(X_transformed.shape)

del song_vec_list_vec
del song_vec_list
del song_vec





print("song vec normalizing & X concatenation & namecall is done ... tSNEing...")


from sklearn.manifold import TSNE

Y= TSNE(n_components = 2).fit_transform(X_transformed)

#Y = X_transformed

#print(Y)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = 'font.TTF'
fontprop = fm.FontProperties(fname=path, size=10)
fontprop_main = fm.FontProperties(fname=path, size=15)

y_min, y_max = np.min(Y, axis = 0), np.max(Y, axis = 0)

Y = (Y - y_min)/(y_max - y_min)

for i in range(Y.shape[0]):
    if i < 80:
        plt.text(Y[i,0], Y[i,1], tag_pq[i], FontProperties = fontprop_main, color = (0,0,0))
        continue
    if i >= SZ:
        plt.text(Y[i,0], Y[i,1], song_name[i-SZ], FontProperties = fontprop, color = (1,0,0))
        print(i,"th song is done....")
        continue
    if i%10 == 0:
        plt.text(Y[i,0], Y[i,1], tag_pq[i], FontProperties = fontprop, color = (0,0,1))
    


plt.title("tSNE dim : " + str(2) + "  LapEig dim : " + str(lapEig), size = 20)
plt.axis('off')
plt.show()
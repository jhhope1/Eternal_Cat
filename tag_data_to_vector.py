f1 = open("train.txt", 'r', encoding='utf-8')

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding

tag_weighted_graph = {}
tag_exi_checker = {}


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

import numpy as np

lapEig = 5

embedding = SpectralEmbedding(n_components = lapEig, affinity = 'precomputed')

X_transformed = embedding.fit_transform(aff_mat)

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
    if i < 100:
        plt.text(Y[i,0], Y[i,1], tag_pq[i], FontProperties = fontprop_main, color = (0,0,0))
        continue
    if i%10 == 0:
        plt.text(Y[i,0], Y[i,1], tag_pq[i], FontProperties = fontprop, color = (0,0,1))



plt.title("tSNE dim : " + str(2) + "LapEig dim : " + str(lapEig), size = 20)
plt.axis('off')
plt.show()



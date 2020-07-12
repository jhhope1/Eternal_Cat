import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import os
import json
import numpy as np
from aggregators import ConcatAggregator
from sklearn.metrics import f1_score, roc_auc_score
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# music_kakao

#file load
taking_song = set()
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data\\')
with open(DATA+"song_to_idx.json", 'r',encoding='utf-8') as f1:
    song_to_idx = json.load(f1)
    for song, idx in song_to_idx.items():
        taking_song.add(int(song))
with open(DATA+"song_meta.json", 'r', encoding='utf-8') as f2:
    song_meta = json.load(f2)
with open(DATA+"idx_to_item.json", 'r', encoding='utf-8') as f3:
    idx_to_item = json.load(f3)
with open(DATA+"entity_to_idx.json", 'r', encoding='utf-8') as f4:
    entity_to_idx = json.load(f4)
with open(DATA+"relation_to_idx.json", 'r', encoding='utf-8') as f5:
    relation_to_idx = json.load(f5)
kg = {}
for line in open(DATA+"kg.txt", 'r', encoding='utf-8'):
    array = line.strip().split(' ')
    head_old = array[0]
    relation_old = array[1]
    tail_old = array[2]
    if song_to_idx[str(head_old)] not in kg:
        kg[song_to_idx[str(head_old)]] = [(relation_to_idx[relation_old],entity_to_idx[tail_old])]
    else:
        kg[song_to_idx[str(head_old)]].append((relation_to_idx[relation_old],entity_to_idx[tail_old]))


def activation(input, kind):
  #print("Activation: {}".format(kind))
  if kind == 'selu':
    return F.selu(input)
  elif kind == 'relu':
    return F.relu(input)
  elif kind == 'relu6':
    return F.relu6(input)
  elif kind == 'sigmoid':
    return torch.sigmoid(input)
  elif kind == 'tanh':
    return F.tanh(input)
  elif kind == 'elu':
    return F.elu(input)
  elif kind == 'lrelu':
    return F.leaky_relu(input)
  elif kind == 'swish':
    return input*F.sigmoid(input)
  elif kind == 'none':
    return input
  else:
    raise ValueError('Unknown non-linearity type')

class AE_KGCN(nn.Module):
    def __init__(self, layer_sizes, dim, n_iter = 1,  n_entity = 25274, n_relation = 3,n_neighbor = 4, nl_type='selu', is_constrained=True, dp_drop_prob=0.0, last_layer_activations='none'):
        super(AE_KGCN, self).__init__()
        #KGCN hyper parameters
        self.n_relation = n_relation
        self.n_entity = n_entity
        self.n_iter = n_iter
        self.dim = dim
        self.aggregator_class = ConcatAggregator
        self.n_neighbor = n_neighbor ## should change if we use n_iter > 1
        # KGCN parameters
        self.n_item = len(song_to_idx)
        self.item_emb_matrix = nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.dim)
        self.entity_emb_matrix = nn.Embedding(num_embeddings=n_entity, embedding_dim=self.dim)
        self.relation_emb_matrix = nn.Embedding(num_embeddings=n_relation, embedding_dim=self.dim)
        self.entity_vectors = []
        self.relation_vectors = []

        #step for iter_n = 1
        self.get_neighbors()

        self.fc = nn.ModuleList([nn.Linear(2*dim, dim)
                                 for _ in range(self.n_iter)])

        # KGCN layer init
        nn.init.xavier_normal_(self.item_emb_matrix.weight)
        nn.init.xavier_normal_(self.entity_emb_matrix.weight)
        nn.init.xavier_normal_(self.relation_emb_matrix.weight)
        for L in self.fc:
            nn.init.xavier_normal_(L.weight)
            L.bias.data.fill_(0.)

        # AE hyper_parameters
        self.layer_sizes = layer_sizes
        self._dp_drop_prob = dp_drop_prob
        self._last_layer_activations = last_layer_activations
        self.is_constrained = is_constrained
        if dp_drop_prob > 0:
            self.drop = nn.Dropout(dp_drop_prob)
        self._last = len(layer_sizes[0]) - 2
        self._nl_type = nl_type

        # encode_decode_parameters: Poor to read...
        self.encode_w = nn.ParameterList(
            [nn.Parameter(torch.rand(layer_sizes[0][i + 1], layer_sizes[0][i])) for i in range(len(layer_sizes[0]) - 1)])
        for ind, w in enumerate(self.encode_w):
            weight_init.xavier_uniform_(w)
        self.encode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(layer_sizes[0][i + 1])) for i in range(len(layer_sizes[0]) - 1)])
        if not is_constrained:
            self.decode_w = nn.ParameterList(
                [nn.Parameter(torch.rand(layer_sizes[1][i + 1], layer_sizes[1][i])) for i in range(len(layer_sizes[1]) - 1)])
            for ind, w in enumerate(self.decode_w):
                weight_init.xavier_uniform(w)
        self.decode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(layer_sizes[1][i + 1])) for i in range(len(layer_sizes[1]) - 1)])
        
        self.encode_to_u = nn.Linear(layer_sizes[0][-1],self.dim)
        weight_init.xavier_uniform(self.encode_to_u.weight)

        self.encode_bn = nn.ModuleList([torch.nn.BatchNorm1d(
            layer_sizes[0][i+1]) for i in range(len(layer_sizes[0])-2)])
        self.decode_bn = nn.ModuleList([torch.nn.BatchNorm1d(
            layer_sizes[1][i+1]) for i in range(len(layer_sizes[1])-2)])

    def get_neighbors(self):
        self.entity_vectors = nn.ParameterList([self.item_emb_matrix.weight])
        self.relation_vectors = nn.ParameterList([])

        relation = []
        entity = []
        for i in range(self.n_item):
            r_list = []
            e_list = []
            for (r, e) in kg[i]:
                r_list.append(r)
                e_list.append(e)
            relation.append(r_list)
            entity.append(e_list)
        relation_vectors_1 = []
        entity_vectors_1 = []
        for idx, r_list in enumerate(relation):
            if self.n_neighbor > len(r_list):
                sampled_indices = np.random.choice(list(range(len(r_list))), size=self.n_neighbor, replace=True)
            else:
                sampled_indices = np.random.choice(list(range(len(r_list))), size=self.n_neighbor, replace=False)
            r_vector = self.relation_emb_matrix(torch.from_numpy(np.array(r_list)[sampled_indices]).type(torch.LongTensor))
            e_vector = self.entity_emb_matrix(torch.from_numpy(np.array(r_list)[sampled_indices]).type(torch.LongTensor))
            relation_vectors_1.append(r_vector)
            entity_vectors_1.append(e_vector)
        self.relation_vectors.append(nn.Parameter(torch.cat(relation_vectors_1)))
        self.entity_vectors.append(nn.Parameter(torch.cat(entity_vectors_1)))

    def aggregate(self, batch_entity_vectors, batch_relation_vectors, user_latent):
        #code for iter = 1
        #entities, relatinos: all entities, relations
        self.aggregators = []

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(
                    self.batch_size, self.dim, self.fc[i], act=torch.tanh)
            else:
                aggregator = self.aggregator_class(
                    self.batch_size, self.dim, self.fc[i])
            self.aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = (-1, self.n_neighbor, self.dim)#same for all batch
                vector = aggregator(self_vectors=batch_entity_vectors[hop],
                                    neighbor_vectors=torch.reshape(
                                    batch_entity_vectors[hop+1], shape),
                                    neighbor_relations=torch.reshape(
                                    batch_relation_vectors[hop], shape),
                                    user_embeddings=user_latent # change to encode vector
                                    )
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = torch.reshape(entity_vectors[0], (self.batch_size,-1, self.dim))

        return res

    def encode(self, x):
        for ind, w in enumerate(self.encode_w):
            x = activation(input=F.linear(input=x, weight=w,
                                          bias=self.encode_b[ind]), kind=self._nl_type)
        if ind != self._last:
            x = self.encode_bn[ind](x)
        if self._dp_drop_prob > 0:  # apply dropout only on code layer
            x = self.drop(x)
        return x
    def encode2u(self, x):
        return self.encode_to_u(x)
    def decode(self, z):
        # constrained autoencode re-uses weights from encoder
        if self.is_constrained:
            for ind, w in enumerate(list(reversed(self.encode_w))):
                if ind != self._last:
                    z = activation(input=F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[ind]),
                                # last layer or decoder should not apply non linearities
                                kind=self._nl_type)
                    z = self.decode_bn[ind](z)
                else:
                    z = activation(input=F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[ind]),
                                # last layer or decoder should not apply non linearities
                                kind=self._last_layer_activations)
                    # if self._dp_drop_prob > 0 and ind!=self._last: # and no dp on last layer
                    #  z = self.drop(z)
        else:
            for ind, w in enumerate(self.decode_w):
                if ind != self._last:
                    z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
                                # last layer or decoder should not apply non linearities
                                kind=self._nl_type)
                    z = self.decode_bn[ind](z)
                else:
                    z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
                                # last layer or decoder should not apply non linearities
                                kind=self._last_layer_activations)
        return z

    def forward(self, x):
        KGCN = True
        if KGCN:        
            if len(x.shape) == 2:
                self.batch_size = x.shape[0]
            else:
                self.batch_size = 1
            
            batch_entity_vectors = []
            batch_relation_vectors = []
            for i in range(self.n_iter+1):
                batch_entity_vectors.append(self.entity_vectors[i].to(device)) #same for all batch
                if i==self.n_iter:
                    break
                batch_relation_vectors.append(self.relation_vectors[i].to(device))#same for all batch
            
            encode = self.encode(x)
            user_latent = self.encode2u(encode)
            res = self.aggregate(batch_entity_vectors, batch_relation_vectors, user_latent)
            ret = torch.bmm(res,torch.reshape(user_latent,(self.batch_size,self.dim,1))).reshape(self.batch_size,-1)
            #g: innerproduct
            return activation(self.decode(encode)+F.pad(ret, (0, self.layer_sizes[0][0]-self.n_item), "constant", 0) ,'sigmoid')
        else:
            return self.decode(self.encode(x))
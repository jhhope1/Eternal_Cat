import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = torch.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = torch.mean(user_embeddings * neighbor_relations, axis = -1)
            user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = torch.unsqueeze(user_relation_scores_normalized, dim=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = torch.mean(neighbor_vectors, axis=2)

        return neighbors_aggregated



class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=F.selu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)

        self.fc1 = nn.Linear(self.dim * 2, self.dim)
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        # [batch_size, -1, dim * 2]
        output = torch.cat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = torch.reshape(output, [-1, self.dim * 2])
        dout = torch.nn.Dropout(self.dropout)
        output = dout(output)
        #output = torch.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = self.fc1(output)
        #output = torch.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = torch.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

rep_dim = 17
batch_size = 5
neighbor_n = 8
secret_dim = 23
#[batch, -1, n_neigh, input_dim]
#c = ConcatAggregator(batch_size, rep_dim, name = 'asdf')

#c._call(torch.zeros(batch_size, secret_dim, rep_dim), torch.rand(batch_size, secret_dim, neighbor_n, rep_dim), torch.rand(batch_size, secret_dim, neighbor_n, rep_dim), torch.rand(batch_size, rep_dim))
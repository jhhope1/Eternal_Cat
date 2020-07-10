import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import ConcatAggregator
from sklearn.metrics import f1_score, roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KGCN(nn.Module):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation):
        super(KGCN, self).__init__()
        self._parse_args(args, adj_entity, adj_relation)
        #self._build_train()
        self.user_emb_matrix = nn.Embedding(num_embeddings=n_user, embedding_dim=self.dim)
        self.entity_emb_matrix = nn.Embedding(num_embeddings=n_entity, embedding_dim=self.dim)
        self.relation_emb_matrix = nn.Embedding(num_embeddings=n_relation, embedding_dim=self.dim)
        self.fc = nn.ModuleList([nn.Linear(2*args.dim ,args.dim) for _ in range(args.n_iter)])
    
    def _parse_args(self, args, adj_entity, adj_relation):
        #[entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.aggregator_class = ConcatAggregator


    
    def get_neighbors(self, seeds):
        seeds = torch.unsqueeze(seeds, -1).to(device)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            #entities[i] = torch.squeeze(entities[i])
            #neighbor_entities = torch.index_select(self.adj_entity, 0, entities[i]).view(self.batch_size, -1)
            #neighbor_relations = torch.index_select(self.adj_relation, 0, entities[i]).view(self.batch_size, -1)
            neighbor_entities = self.adj_entity[entities[i]].view(self.batch_size, -1).to(device)
            neighbor_relations = self.adj_relation[entities[i]].view(self.batch_size, -1).to(device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations
    def aggregate(self, entities, relations):
        self.aggregators = []
        #entities = [torch.squeeze(e) for e in entities]
        #entity_vectors = [torch.index_select(self.entity_emb_matrix, 0, i) for i in entities]
        entity_vectors = [self.entity_emb_matrix(i) for i in entities]
        #relation_vectors = [torch.index_select(self.relation_emb_matrix, 0, i) for i in relations]
        relation_vectors = [self.relation_emb_matrix(i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, self.fc[i], act = torch.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim, self.fc[i])
            self.aggregators.append(aggregator)
        
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = (self.batch_size, -1, self.n_neighbor, self.dim)
                vector = aggregator(self_vectors = entity_vectors[hop],
                                neighbor_vectors=torch.reshape(entity_vectors[hop+1], shape),
                                neighbor_relations=torch.reshape(relation_vectors[hop], shape),
                                user_embeddings=self.user_embeddings
                )
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
    
        res = torch.reshape(entity_vectors[0], (self.batch_size, self.dim))

        return res

    def forward(self, user_indices, item_indices):
        self.user_embeddings = self.user_emb_matrix(user_indices.long())
        #self.user_embeddings = torch.index_select(self.user_emb_matrix, 0, user_indices)

        entities, relations = self.get_neighbors(item_indices)
        res = self.aggregate(entities, relations)
        ret = torch.sum(res * self.user_embeddings, dim=-1)
        return torch.sigmoid(ret)
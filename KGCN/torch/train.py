import torch
import torch.nn.functional as F
from torch import nn, optim
from model import KGCN
from data_loader import load_data
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_PATH = './AE_weight.pth'

#neighbor_sample_size = 8
#dim = 16
#n_iter = 1
#lr = 5e-4
#l2_reg = 1e-4
#batch_size = 128

def base_loss_fn(y_hat, y):
    base_loss = F.binary_cross_entropy_with_logits(y.float(), y_hat, reduction='mean')
    return base_loss

def train(args, show_loss = True, show_topk = False):
    n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation = load_data(args)
    print('Users: %d, Items: %d, Entities: %d, Relations: %d'%(n_user, n_item, n_entity, n_relation))
    adj_entity = torch.from_numpy(adj_entity)
    adj_relation = torch.from_numpy(adj_relation)
    model = KGCN(args = args, n_user = n_user, n_entity = n_entity,
    n_relation = n_relation, adj_entity = adj_entity, adj_relation = adj_relation).to(device)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.l2_weight)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        train_loss = 0
        cnt = 0
        for i in range(0, train_data.shape[0], args.batch_size):
            optimizer.zero_grad()
            if i + args.batch_size > train_data.shape[0]:
                break
            k = i + args.batch_size
            train_input_user = torch.from_numpy(train_data[i:k, 0]).to(device)
            train_input_item = torch.from_numpy(train_data[i:k, 1]).to(device)
            train_input_label = torch.from_numpy(train_data[i:k, 2]).to(device)

            train_output = model(train_input_user, train_input_item)
            base_loss = base_loss_fn(train_output, train_input_label)
            
            l2_loss = torch.sum(model.user_emb_matrix.weight**2) + torch.sum(model.entity_emb_matrix.weight**2) + torch.sum(model.relation_emb_matrix.weight**2)

            for aggregator in model.aggregators:
                l2_loss = l2_loss + torch.sum(aggregator.fc_.weight**2)
            loss = base_loss + model.l2_weight * l2_loss

            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            cnt += args.batch_size
        
        print('====> Epoch: {} Total Loss: {:.4f}'.format(epoch, train_loss / cnt))
        torch.save(model.state_dict(), model_PATH)

        #test accuracy evaluation for each epoch
        model.eval()
        with torch.no_grad():
            #model.load_state_dict(torch.load(model_PATH))
            #model.eval()
            f1_tot = 0
            acc_tot = 0
            auc_tot = 0
            cnt = 0
            for i in range(0, test_data.shape[0], args.batch_size):
                if i + args.batch_size > test_data.shape[0]:
                    break
                k = i + args.batch_size
                test_input_user = torch.from_numpy(test_data[i:k, 0]).to(device)
                test_input_item = torch.from_numpy(test_data[i:k, 1]).to(device)
                test_input_label = torch.from_numpy(test_data[i:k, 2]).to(device)

                test_output = model(test_input_user, test_input_item)

                auc_tot += roc_auc_score(y_true=test_input_label.cpu(), y_score=test_output.cpu())
                test_output[test_output >= 0.5] = 1
                test_output[test_output < 0.5] = 0
                f1_tot += f1_score(y_true=test_input_label.cpu(), y_pred=test_output.cpu())
                acc_tot += torch.mean((test_output == test_input_label).float())
                cnt += 1

            print('Average ROC_AUC score: ', auc_tot / cnt)
            print('Average F1 score: ', f1_tot / cnt)
            print('Average accuracy: ', acc_tot.item() / cnt)

            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')


def topk_inference(model, user, train_record_user, test_record_user, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    test_item_list = list(item_set - train_record)
    item_score_map = dict()
    start = 0
    while start + batch_size <= len(test_item_list):
        scores = model(torch.tensor([user] * batch_size).to(device), torch.tensor(test_item_list[start:start + batch_size]).to(device)).cpu().tolist()
        items = test_item_list[start:start + batch_size]
        for item, score in zip(items, scores):
            item_score_map[item] = score
        start += batch_size

    # padding the last incomplete minibatch if exists
    if start < len(test_item_list):
        scores = model(torch.tensor([user] * batch_size).to(device), torch.tensor(test_item_list[start:] + [test_item_list[-1]] * (
                            batch_size - len(test_item_list) + start)).to(device)).cpu().tolist()
        items = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)

        for item, score in zip(items, scores):
            item_score_map[item] = score

    item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
    item_sorted = [i[0] for i in item_score_pair_sorted]
    return item_sorted[:k]

def topk_eval(model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            scores = model(torch.tensor([user] * batch_size).to(device), torch.tensor(test_item_list[start:start + batch_size]).to(device)).cpu().tolist()
            items = test_item_list[start:start + batch_size]
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            scores = model(torch.tensor([user] * batch_size).to(device), torch.tensor(test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)).to(device)).cpu().tolist()
            items = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)

            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall

def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


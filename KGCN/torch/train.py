import torch
import torch.nn.functional as F
from torch import nn, optim
from model import KGCN
from data_loader import load_data
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

device = torch.device("cpu")# change plz
model_PATH = './AE_weight.pth'

#neighbor_sample_size = 8
#dim = 16
#n_iter = 1
#lr = 5e-4
#l2_reg = 1e-4
#batch_size = 128

def loss_fn(y_hat, y):
    base_loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction='sum')
    return base_loss

def train(args, show_loss = True, show_topk = False):
    n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation = load_data(args)
    print('Users: %d, Items: %d, Entities: %d, Relations: %d'%(n_user, n_item, n_entity, n_relation))
    adj_entity = torch.from_numpy(adj_entity)
    adj_relation = torch.from_numpy(adj_relation)
    model = KGCN(args = args, n_user = n_user, n_entity = n_entity,
    n_relation = n_relation, adj_entity = adj_entity, adj_relation = adj_relation).to(device)
    
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
            train_input_user = torch.from_numpy(train_data[i:k, 0])
            train_input_item = torch.from_numpy(train_data[i:k, 1])
            train_input_label = torch.from_numpy(train_data[i:k, 2])

            train_output = model(train_input_user, train_input_item)
            loss = loss_fn(train_output, train_input_label)
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
                test_input_user = torch.from_numpy(test_data[i:k, 0])
                test_input_item = torch.from_numpy(test_data[i:k, 1])
                test_input_label = torch.from_numpy(test_data[i:k, 2])

                test_output = model(test_input_user, test_input_item)

                auc_tot += roc_auc_score(y_true=test_input_label, y_score=test_output)
                test_output[test_output >= 0.5] = 1
                test_output[test_output < 0.5] = 0
                f1_tot += f1_score(y_true=test_input_label, y_pred=test_output)
                acc_tot += torch.mean((test_output == test_input_label).float())
                cnt += 1

            print('Average ROC_AUC score: ', auc_tot / cnt)
            print('Average F1 score: ', f1_tot / cnt)
            print('Average accuracy: ', acc_tot.item() / cnt)





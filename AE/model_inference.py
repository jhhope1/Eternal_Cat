import sys
import torch
from torch.nn import functional as F
from torch import nn, optim
import AE_model
import os
input_dim = 77658
model_PATH = './AE_weight.pth'

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

device = torch.device("cuda")
model = AE_model.AutoEncoder(layer_sizes = ((input_dim,1024,512,1024,512),(512,1024,512,1024,input_dim)), is_constrained=True, symmetric=True, dp_drop_prob=0.7).to(device)

model.load_state_dict(torch.load(os.path.join(data_path,model_PATH)))
model.eval()
def inference(one_hot):
    with torch.no_grad():
        recon = model(torch.FloatTensor([one_hot]).to(device))
        return recon[0].to('cpu').tolist()


import sys
import torch
from torch.nn import functional as F
from torch import nn, optim
import res_AE_model
import os
input_dim = 100058
output_dim = 57229
model_PATH = './res_AE_weight.pth'

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

device = torch.device("cpu")

layer_sizes = (input_dim,300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,output_dim)
model = res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes).to(device)
model.load_state_dict(torch.load(os.path.join(data_path,model_PATH)))
model.eval()
def inference(one_hot):
    with torch.no_grad():
        recon = model(torch.FloatTensor([one_hot]).to(device))
        return recon[0].to('cpu').tolist()


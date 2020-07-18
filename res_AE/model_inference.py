import sys
import torch
from torch.nn import functional as F
from torch import nn, optim
import res_AE_model
import os
input_dim = 101252
output_dim = 61706
model_PATH = './res_AE_weight.pth'

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D_ = 400 #layer dim

layer_sizes = (input_dim,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,output_dim)
model = res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes).to(device)
model.load_state_dict(torch.load(os.path.join(data_path,model_PATH), map_location = device))
model.eval()
model.to(device)
def inference(one_hot: torch.Tensor): #one_hot: torch.FloatTensor
    with torch.no_grad():
        recon = model(one_hot).to(device)
        return recon


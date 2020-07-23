import sys
import torch
from torch.nn import functional as F
from torch import nn, optim
import res_AE_model
import os
from const import *

model = res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes,dp_drop_prob=0, is_res = is_res).to(device)
model.load_state_dict(torch.load(os.path.join(data_path,model_PATH), map_location = device))
model.eval()
model.to(device)
def inference(one_hot: torch.Tensor): #one_hot: torch.FloatTensor
    with torch.no_grad():
        recon = model(one_hot).to(device)
        return recon


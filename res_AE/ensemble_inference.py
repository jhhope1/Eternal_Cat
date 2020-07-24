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

import const2
model2 = res_AE_model.res_AutoEncoder(layer_sizes = const2.layer_sizes, dp_drop_prob=0, is_res = const2.is_res).to(device)
model2.load_state_dict(torch.load(os.path.join(data_path,const2.model_PATH), map_location = device))
model2.eval()
model2.to(device)

import const3
model3 = res_AE_model.res_AutoEncoder(layer_sizes = const3.layer_sizes, dp_drop_prob=0, is_res = const3.is_res).to(device)
model3.load_state_dict(torch.load(os.path.join(data_path,const3.model_PATH), map_location = device))
model3.eval()
model3.to(device)

import const4
model4 = res_AE_model.res_AutoEncoder(layer_sizes = const4.layer_sizes, dp_drop_prob=0, is_res = const4.is_res).to(device)
model4.load_state_dict(torch.load(os.path.join(data_path,const4.model_PATH), map_location = device))
model4.eval()
model4.to(device)

def ensemble(recons, op_type = 'sum'):
    ret = recons[0]
    for x in recons[1:]:
        if op_type == 'sum':
            ret = ret + x
        if op_type == 'max':
            ret = torch.max(ret, x)
        if op_type == 'min':
            ret = torch.min(ret, x)
    return ret

def inference(one_hot: torch.Tensor, op_type='sum'): #one_hot: torch.FloatTensor
    with torch.no_grad():
        recon = model(one_hot).to(device)
        recon2 = model2(one_hot).to(device)
        recon3 = model3(one_hot).to(device)
        recon4 = model4(one_hot).to(device)
        recons = [recon, recon2, recon3, recon4]

        return ensemble(recons, op_type = op_type)


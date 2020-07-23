import sys
import torch
from torch.nn import functional as F
from torch import nn, optim
import res_AE_model
import os

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')
D_ = 300 #layer dim

output_dim = 61706
type_nn = ['title', 'title_tag', 'song_meta_tag', 'song_meta']
model_PATH = {name: os.path.join(data_path, 'res_AE_' + name) + '_weight.pth' for name in type_nn}
#(3308 3308 100252 96944) if titletag
#(1000 4308 100252 96944) if not
input_dim = {'title': 7785, 'title_tag': 7785, 'song_meta_tag': 104729, 'song_meta': 96944}
layer_sizes = {name: (input_dim[name],D_,D_,D_,output_dim) for name in type_nn}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = {name: res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes[name], is_res=False).to(device) for name in type_nn}

#layer_sizes = (input_dim,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,D_,output_dim)
#model = res_AE_model.res_AutoEncoder(layer_sizes = layer_sizes).to(device)
for id_nn in type_nn:
    model[id_nn].load_state_dict(torch.load(os.path.join(data_path,model_PATH[id_nn]), map_location = device))
    model[id_nn].eval()
    model[id_nn].to(device)

def inference(one_hot: torch.Tensor, id_nn: str): #one_hot: torch.FloatTensor
    with torch.no_grad():
        recon = model[id_nn](one_hot).to(device)
        return recon


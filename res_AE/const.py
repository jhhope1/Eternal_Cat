import os
import torch
batch_size = 512
random_seed = 10
validation_ratio = 0.01
test_ratio = 0.01
input_dim =  38459
output_dim =  20517
song_size =  17937
noise_p = 0.5
extract_song = 100
extract_tag = 10
aug_step = 0 #blobfusad
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')
model_PATH = os.path.join(data_path, './res_AE_weight.pth')
epochs = 1000
log_interval = 100
learning_rate = 3e-4
D_ = 300
weight_decay = 1e-10
layer_sizes = (input_dim,D_,D_,D_,output_dim)
dropout_p = 0.0
is_res = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps = 10
pos_weight = torch.tensor([-1.]).to(device)
neg_weight = torch.tensor([-0.3]).to(device)

def show_vars():
    print('input_dim = ', input_dim)
    print('output_dim = ', output_dim)
    print('song_size = ', song_size)
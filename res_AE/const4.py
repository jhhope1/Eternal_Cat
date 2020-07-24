import os
import torch
batch_size = 256
random_seed = 10
validation_ratio = 0.01
test_ratio = 0.01
input_dim =  137580
output_dim =  79303
song_size =  75078
noise_p = 0.5
extract_song = 100
extract_tag = 10
aug_step = 0 #blobfusad
epochs = 1000
log_interval = 100
learning_rate = 1e-3
D_ = 1000
weight_decay = 1e-10
layer_sizes = (input_dim,D_,D_,D_,output_dim)
dropout_p = 0.0
is_res = True
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(PARENT_PATH, 'data')
try:
    import google.colab
except ModuleNotFoundError:
    data_path = os.path.join(PARENT_PATH, 'data')
else:
    data_path = os.path.join(PARENT_PATH, 'drive/Shared drives/Eternal_Cat/data')   
 
model_PATH = os.path.join(data_path, './res_AE_weight'+str(layer_sizes)+'res'+str(is_res)+'dp'+str(dropout_p)+'.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps = 10
pos_weight = torch.tensor([-1.]).to(device)
neg_weight = torch.tensor([-0.3]).to(device)

def show_vars():
    print('input_dim = ', input_dim)
    print('output_dim = ', output_dim)
    print('song_size = ', song_size)
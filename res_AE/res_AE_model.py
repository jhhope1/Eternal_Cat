import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

def activation(input, kind):
  #print("Activation: {}".format(kind))
  if kind == 'selu':
    return F.selu(input)
  elif kind == 'relu':
    return F.relu(input)
  elif kind == 'relu6':
    return F.relu6(input)
  elif kind == 'sigmoid':
    return torch.sigmoid(input)
  elif kind == 'tanh':
    return F.tanh(input)
  elif kind == 'elu':
    return F.elu(input)
  elif kind == 'lrelu':
    return F.leaky_relu(input)
  elif kind == 'swish':
    return input*F.sigmoid(input)
  elif kind == 'none':
    return input
  else:
    raise ValueError('Unknown non-linearity type')

class res_AutoEncoder(nn.Module):
  def __init__(self, layer_sizes, nl_type='selu', dp_drop_prob=0.0, last_layer_activations='sigmoid'):
    super(res_AutoEncoder, self).__init__()
    self.layer_sizes = layer_sizes
    self._dp_drop_prob = dp_drop_prob
    self._last_layer_activations = last_layer_activations

    if dp_drop_prob > 0:
      self.drop = nn.Dropout(dp_drop_prob)

    self._last = len(layer_sizes) - 2
    self._nl_type = nl_type

    self.MLP_list = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
    self.res_list = nn.ModuleList([nn.Linear(layer_sizes[i*2], layer_sizes[i*2+2]) for i in range(int((len(layer_sizes)-1)/2))])
    
    for L in self.MLP_list:
      weight_init.xavier_uniform_(L.weight)
      L.bias.data.fill_(0.)
    for L in self.res_list:
      weight_init.xavier_uniform_(L.weight)
      L.bias.data.fill_(0.)

    self.MLP_bn = nn.ModuleList([torch.nn.BatchNorm1d(layer_sizes[i+1]) for i in range(len(layer_sizes)-2)])
    self.dp = nn.Dropout(self._dp_drop_prob)

  def forward(self, x):
    MLP_output = x
    for idx, L in enumerate(self.MLP_list):
      if idx == len(self.MLP_list)-1:
        x = L(x)
        break
      
      if idx % 2 == 0:
        x = activation(L(x),self._nl_type)
        if idx != (len(self.MLP_list)-1):
          x = self.MLP_bn[idx](x)
      else:
        x = activation(L(x),self._nl_type)
        if x.shape[1]==MLP_output.shape[1]:
          x = x + MLP_output #chk if idx-1 = len(MLP_output)-2
        MLP_output = x
        x = self.dp(x)
      '''if idx == int(len(self.layer_sizes)/2):
        x = self.dp(x)'''
    return activation(x, self._last_layer_activations)
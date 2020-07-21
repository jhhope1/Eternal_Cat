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

class MMCF(nn.Module):
  def __init__(self, layer_sizes, output_dim, nl_type='selu', dp_drop_prob=0.0, last_layer_activations='sigmoid'):
    super(MMCF, self).__init__()
    self.output_dim = output_dim
    self.layer_sizes = layer_sizes
    self._dp_drop_prob = dp_drop_prob
    self._last_layer_activations = last_layer_activations

    self._last = len(layer_sizes) - 2
    self._nl_type = nl_type

    self.encode_weight_list = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes)-1)])
    self.encode_bias_list = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1])) for i in range(len(layer_sizes)-1)])
    self.decode_bias_list = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i-1])) for i in range(len(layer_sizes)-1, 0, -1)])
    for idx, W in enumerate(self.encode_weight_list):
      weight_init.xavier_uniform_(W)
      self.encode_bias_list[idx].data.fill_(0.)
      self.decode_bias_list[idx].data.fill_(0.)

    self.encode_bn = nn.ModuleList([torch.nn.BatchNorm1d(layer_sizes[i+1]) for i in range(len(layer_sizes)-2)])
    self.decode_bn = nn.ModuleList([torch.nn.BatchNorm1d(layer_sizes[i]) for i in range(len(layer_sizes)-2, 0, -1)])
    if self._dp_drop_prob>0:
      self.dp = nn.Dropout(self._dp_drop_prob)
  def encode(self, x):
    for idx, W in enumerate(self.encode_weight_list):
      x = nn.functional.linear(x, torch.transpose(W,0,1), self.encode_bias_list[idx])
      '''if idx != (len(self.MLP_encode_list-1)):
        x = self.MLP_encode_bn(x)'''
      # why not..?
    return torch.sigmoid(x) # ???why...
  def decode(self, x):
    for idx in range(len(self.encode_weight_list)):
      W = self.encode_weight_list[len(self.encode_weight_list)-1-idx]
      x = torch.nn.functional.linear(x, W, self.decode_bias_list[idx])
      '''if idx != (len(self.MLP_decode_list)):
        x = self.MLP_decode_bn(x)'''
      #why not..?
    return torch.sigmoid(x)

  def forward(self, x):
    return self.decode(self.dp(self.encode(x))).narrow(1,0,self.output_dim)
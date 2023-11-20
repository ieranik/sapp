import torch
import torch.nn as nn
from mpnet.Layers import MLP, Encoder

class SAPPNet(nn.Module):
    def __init__(self, AE_input_size, state_size,):
        super(SAPPNet, self).__init__()
        self.encoder = Encoder()

        x = self.encoder(torch.autograd.Variable(torch.rand([1] + AE_input_size)))    
        self.mlp = MLP(x.shape[-1] + state_size*2, state_size)

    def get_environment_encoding(self, obs):
        return self.encoder(obs)

    def forward(self, x):
        return self.mlp(x)
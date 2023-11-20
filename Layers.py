import torch
import torch.nn as nn

from torch.autograd import Variable

from einops.layers.torch import Rearrange


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_size, 1280),
            nn.PReLU(),
            nn.Dropout(),
            
            nn.Linear(1280, 1024),
            nn.PReLU(),
            nn.Dropout(),
            
            nn.Linear(1024, 896),
            nn.PReLU(),
            nn.Dropout(),
            
            nn.Linear(896, 768),
            nn.PReLU(),
            nn.Dropout(),
            
            nn.Linear(768, 512),
            nn.PReLU(),
            nn.Dropout(),
            
            nn.Linear(512, 384),
            nn.PReLU(),
            nn.Dropout(),
            
            nn.Linear(384, 256),
            nn.PReLU(), 
            nn.Dropout(),
            
            nn.Linear(256, 256),
            nn.PReLU(), 
            nn.Dropout(),
            
            nn.Linear(256, 128),
            nn.PReLU(), 
            nn.Dropout(),
            
            nn.Linear(128, 64),
            nn.PReLU(), 
            nn.Dropout(),
            
            nn.Linear(64, 32),
            nn.PReLU(),
            
            nn.Linear(32, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.head(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=[7, 7], stride=[1, 1]),
            nn.MaxPool2d(kernel_size=3),
            nn.PReLU(),  
           
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=[5, 5], stride=[1, 1]),
            nn.MaxPool2d(kernel_size=3),
            nn.PReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[1, 1]),
            nn.MaxPool2d(kernel_size=3),
            nn.PReLU(),
            
            Rearrange('b c h w -> b (c h w)')
        )
        
    def forward(self, obs):
        return self.encoder(obs)

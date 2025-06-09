import torch
import torch.nn as nn
import torch.nn.functional as F 

from diffusers.models.embeddings import GaussianFourierProjection 


class CompassEncoder(nn.Module): 
    def __init__(self, output_dim): 
        super().__init__() 
        self.input_dim = 1  
        self.output_dim = output_dim 
        self.linear1 = nn.Linear(2, output_dim)  
        self.linear2 = nn.Linear(output_dim, output_dim)  
        self.linear3 = nn.Linear(output_dim, output_dim) 
        self.gaussian_fourier_embedding = GaussianFourierProjection(output_dim // 2, log=False)  


    def forward(self, x):  
        x = torch.stack([torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * x)], dim=-1)  
        x = self.linear1(x) 
        x = F.relu(x) 
        x = self.linear2(x) 
        x = F.relu(x) 
        x = self.linear3(x) 
        return x 


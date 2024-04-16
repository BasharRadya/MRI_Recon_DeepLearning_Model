import torch
from models.subsampling import SubsamplingLayer

class VanillaModel(torch.nn.Module):
    def __init__(self,drop_rate, device, learn_mask):
        super().__init__()
        
        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask) #initialize subsampling layer - use this in your own model
        self.conv = torch.nn.Conv2d(1,1,3,padding="same").to(device) # some conv layer as a naive reconstruction model - you probably want to find something better.
        
    def forward(self, x):
        x = self.subsample(x) #get subsampled input in image domain - use this as first line in your own model's forward
        x = self.conv(x).squeeze(1)
        return x
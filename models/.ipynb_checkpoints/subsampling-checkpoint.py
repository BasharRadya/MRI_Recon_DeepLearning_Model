import torch
from data import transforms

RES = 320 #used resolution (image size to which we crop data) - for our purposes it's constant

class SubsamplingLayer(torch.nn.Module):
    def __init__(self,drop_rate, device, learn_mask):
        super().__init__()
         
        self.learn_mask = learn_mask
        self.drop_rate = 1-drop_rate 
       
        self.mask = torch.randn(1,RES,RES,2).to(device)
        self.binary_mask = self.mask.clone().to(device)
        
        self.mask = torch.nn.Parameter(self.mask,requires_grad=False)
        self.binary_mask = torch.nn.Parameter(self.binary_mask,requires_grad=learn_mask)
         
    
    def apply_mask(self,x):
    
        drop_threshold = torch.topk(self.mask.flatten(), int(self.drop_rate * len(self.mask.flatten())), largest=True).values.min()
        with torch.no_grad():
            self.binary_mask.data = (self.mask >= drop_threshold).to(torch.float)
        
        return self.binary_mask*x

    
    def mask_grad(self,lr):
        
        if self.learn_mask:
            self.mask-= lr*self.binary_mask.grad
            self.mask.clamp(-1,1)
        
        return
    
    def forward(self, x):
        x = self.apply_mask(x) #apply mask over input, removing part of frequency domain data
        return transforms.complex_abs(transforms.ifft2_regular(x)).unsqueeze(1) #cast to image domain
        
     
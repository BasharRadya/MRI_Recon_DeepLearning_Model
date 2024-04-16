import torch
import torch.nn as nn

from models.subsampling import SubsamplingLayer
RES = 320 #used resolution (image size to which we crop data) - for our purposes it's constant

class PEARModel(torch.nn.Module):
    def __init__(self,drop_rate, device, learn_mask):
        super().__init__()
        
        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask) #initialize subsampling layer - use this in your own model
        # self.conv = torch.nn.Conv2d(1,1,3,padding="same").to(device) # some conv layer as a naive reconstruction model - you probably want to find something better.
        # encoder (downsampling)
        self.encoder_seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU()
        )
        # 320x320x64
        self.encoder_seq1_maxpool=nn.MaxPool2d(kernel_size=2, stride=2)
        # 160x160x64


        self.encoder_seq2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
        )
        # 160x160x64
        self.encoder_seq2_maxpool=nn.MaxPool2d(kernel_size=2, stride=2)
        # 80x80x128


        self.encoder_seq3  = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
        )
        # 80x80x128
        self.encoder_seq3_maxpool=nn.MaxPool2d(kernel_size=2, stride=2)
        # 40x40x256

        
        self.encoder_seq4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
        )
        # decoder (upsampling)

        # 40x40x256
        self.decoder_seq1_upsample=nn.Upsample(scale_factor=2)
        # 80x80x256
        self.decoder_seq1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
        )
        
        # 80x80x128
        self.decoder_seq2_upsample=nn.Upsample(scale_factor=2)
        # 160x160x128
        self.decoder_seq2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
        )
        
        # 160x160x64
        self.decoder_seq3_upsample=nn.Upsample(scale_factor=2)
        # 320x320x64
        self.decoder_seq3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.Dropout(p=drop_rate),
            nn.ReLU(),
        )
        # 320x320x1

    def forward(self, x):
        
        x1 = self.subsample(x) #get subsampled input in image domain - use this as first line in your own model's forward
        # encoder
        x2 = self.encoder_seq1(x1)
        x3 = self.encoder_seq1_maxpool(x2)
        x4 = self.encoder_seq2(x3)
        x5 = self.encoder_seq2_maxpool(x4)
        x6 = self.encoder_seq3(x5)
        x7 = self.encoder_seq3_maxpool(x6)

        x8 = self.encoder_seq4(x7)

        # decoder
        x9 = self.decoder_seq1_upsample(x8)
        x10 = self.decoder_seq1(x9+x6)
        x11 = self.decoder_seq2_upsample(x10)
        x12 = self.decoder_seq2(x11+x4)
        x13 = self.decoder_seq3_upsample(x12)
        x14 = self.decoder_seq3(x13+x2)
        x = x14
        return x
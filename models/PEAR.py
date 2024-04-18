import torch
import torch.nn as nn

from models.subsampling import SubsamplingLayer
RES = 320 #used resolution (image size to which we crop data) - for our purposes it's constant

class PEARModel(torch.nn.Module):
    def __init__(
        self,
        drop_rate, 
        device, 
        learn_mask,
        
        block_len,
        blocks_num,
        bottleneck_block_len,
        first_channel,
        in_channel,
        k_size,
        st
    ):
        super().__init__()
    

        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask) #initialize subsampling layer - use this in your own model
        self.learn_mask = learn_mask
        # self.conv = torch.nn.Conv2d(1,1,3,padding="same").to(device) # some conv layer as a naive reconstruction model - you probably want to find something better.
        # encoder (downsampling)
        
        pool_li = []
        upscale_li = []
        encoder_li = []
        decoder_li = []
        is_first_block = True
        block_channel = None

        def getBlock(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, padding=1),
                nn.Dropout(p=drop_rate),
                nn.ReLU(),
            )
        for block_i in range(blocks_num):
            encoder_seq_list = []
            decoder_seq_list = []
            is_first = True
            if is_first_block:
                block_channel = first_channel
            else:
                block_channel *= 2
            for i in range(block_len):
                cur_in = None
                if is_first and is_first_block:
                    cur_in = in_channel
                elif is_first:
                    cur_in =  block_channel // 2
                else:
                    cur_in = block_channel
                cur_out = block_channel
                encoder_seq_list.append(getBlock(cur_in, cur_out))
                decoder_seq_list.append(getBlock(cur_out, cur_in))
                is_first = False
            encoder_li.append(nn.Sequential(*encoder_seq_list))
            decoder_seq_list.reverse()
            decoder_li.append(nn.Sequential(*decoder_seq_list))
            pool_li.append(nn.MaxPool2d(kernel_size=2, stride=st))
            upscale_li.append(nn.Upsample(scale_factor=st))
            is_first_block = False
        decoder_li.reverse()

        last_block_channel = block_channel
        bottleneck_li = []
        for i in range(bottleneck_block_len):
            is_first = i == 0
            is_last = (i + 1 ==  bottleneck_block_len)
            cur_in = None
            cur_out = None
            if is_first:
                cur_in = last_block_channel
            else:
                cur_in = last_block_channel * 2
            if is_last:
                cur_out = last_block_channel
            else:
                cur_out = last_block_channel * 2
            bottleneck_li.append(getBlock(cur_in, cur_out))
        bottleneck = nn.Sequential(*bottleneck_li)


        def convert_models_to_cuda(li):
            return [model.to(device) for model in li]
        self.encoder_li = convert_models_to_cuda(encoder_li)
        self.decoder_li = convert_models_to_cuda(decoder_li)
        self.pool_li = convert_models_to_cuda(pool_li)
        self.upscale_li = convert_models_to_cuda(upscale_li)
        self.bottleneck = bottleneck.to(device)
        
        # self.encoder_seq1 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU()
        # )
        # # 320x320x64
        # self.encoder_seq1_maxpool=nn.MaxPool2d(kernel_size=2, stride=2)
        # # 160x160x64


        # self.encoder_seq2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        # )
        # # 160x160x64
        # self.encoder_seq2_maxpool=nn.MaxPool2d(kernel_size=2, stride=2)
        # # 80x80x128


        # self.encoder_seq3  = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        # )
        # # 80x80x128
        # self.encoder_seq3_maxpool=nn.MaxPool2d(kernel_size=2, stride=2)
        # # 40x40x256

        
        # self.encoder_seq4 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        # )
        # # decoder (upsampling)

        # # 40x40x256
        # self.decoder_seq1_upsample=nn.Upsample(scale_factor=2)
        # # 80x80x256
        # self.decoder_seq1 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        # )
        
        # # 80x80x128
        # self.decoder_seq2_upsample=nn.Upsample(scale_factor=2)
        # # 160x160x128
        # self.decoder_seq2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        # )
        
        # # 160x160x64
        # self.decoder_seq3_upsample=nn.Upsample(scale_factor=2)
        # # 320x320x64
        # self.decoder_seq3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
        #     nn.Dropout(p=drop_rate),
        #     nn.ReLU(),
        # )
        # # 320x320x1

    def forward(self, x):
        x = self.subsample(x) #get subsampled input in image domain - use this as first line in your own model's forward
        skip_conn = []
        encoder_i = iter(self.encoder_li)
        pool_i = iter(self.pool_li)
        # encoder
        try:
            while True:
                
                cur_encoder_block = next(encoder_i)
                cur_pool_block = next(pool_i)
                x = cur_encoder_block(x)
                skip_conn.append(x)
                x = cur_pool_block(x)
        except StopIteration:
            pass
        skip_conn.reverse()
        #bottleneck
        x = self.bottleneck(x)
        

        decoder_i = iter(self.decoder_li)
        upscale_i = iter(self.upscale_li)
        skip_conn_i = iter(skip_conn)
        #decoder
        try:
            while True:
                cur_decoder_block = next(decoder_i)
                cur_upscale_block = next(upscale_i)
                cur_skip_conn = next(skip_conn_i) 
                x = cur_upscale_block(x)
                x = cur_decoder_block(x + cur_skip_conn)
        except StopIteration:
            pass

        
        return x










        # x1 = self.subsample(x) #get subsampled input in image domain - use this as first line in your own model's forward
        # # encoder
        # x2 = self.encoder_seq1(x1)
        # x3 = self.encoder_seq1_maxpool(x2)
        # x4 = self.encoder_seq2(x3)
        # x5 = self.encoder_seq2_maxpool(x4)
        # x6 = self.encoder_seq3(x5)
        # x7 = self.encoder_seq3_maxpool(x6)

        # x8 = self.encoder_seq4(x7)

        # # decoder
        # x9 = self.decoder_seq1_upsample(x8)
        # x10 = self.decoder_seq1(x9+x6)
        # x11 = self.decoder_seq2_upsample(x10)
        # x12 = self.decoder_seq2(x11+x4)
        # x13 = self.decoder_seq3_upsample(x12)
        # x14 = self.decoder_seq3(x13+x2)
        # x = x14
        # return x
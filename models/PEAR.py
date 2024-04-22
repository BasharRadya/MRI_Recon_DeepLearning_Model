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

        def getBlock(in_ch, out_ch, act_and_drop=True):
            li = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, padding=1)]
            if act_and_drop:
                li +=  [nn.Dropout(p=drop_rate), nn.LeakyReLU()]
            return nn.Sequential(*li)
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
                act_and_drop = not (is_first and is_first_block)
                decoder_seq_list.append(getBlock(cur_out, cur_in, act_and_drop=act_and_drop))
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

        self.encoder_li = nn.ModuleList(self.encoder_li)
        self.decoder_li = nn.ModuleList(self.decoder_li)
        self.pool_li = nn.ModuleList(self.pool_li)
        self.upscale_li = nn.ModuleList(self.upscale_li)
        
       
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

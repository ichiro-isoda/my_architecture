import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TestModel(nn.Module):
    def __init__(
            self,
            in_ch=1,expansion =4
    ):
        super().__init__()
        self.encoder = ResEncoder(in_ch=in_ch)
        self.decoder = TestDecoder(expansion=expansion)
    def forward(self, x):
        decode = self.decoder(self.encoder(x))
        return decode
    
class ResEncoder(nn.Module):
    def __init__(
            self,
            in_ch=1, layer=50
    ):
        super().__init__()
        self.phase = 'train'
        init_ch = 64
        if layer == 50:
            layer = [3,4,6,3]
        elif layer == 101:
            layer = [3,4,23,3]
        elif layer == 152:
            layer == [3,8,36,3]
        self.basic_layer = nn.Sequential(
            conv_block(in_ch=in_ch, out_ch=init_ch, kernel_size=7, pad=2, stride=3, bias=False),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            conv_25(init_ch, init_ch, stride=1, layer=layer[0]),
            conv_25(init_ch*4, init_ch*2, stride=2, layer=layer[1]),
            conv_25(init_ch*8, init_ch*4, stride=2, layer=layer[2]),
            conv_25(init_ch*16, init_ch*8, stride=2, layer=layer[3]),
        )
    
    def forward(self, x):
        l = ['conv_block', 'pool', 'conv1','conv2','conv3','conv4']
        i=0
        print('==============\n{}'.format(x.shape))
        for el in self.basic_layer:
            print('{}: {}->'.format(l[i],x.shape),end=' ')
            x = el(x)
            print(x.shape)
            i+=1
        print('================')
        return x

class conv_25(nn.Module):
    def __init__(
            self,
            in_ch,out_ch,
            stride,layer,expansion=4
    ):
        super().__init__()

        self.basic_layers = nn.ModuleList()
        self.basic_layers.append(
            res_block(in_ch, out_ch,stride,expansion,down=True)
        ) 

        for i in range(1, layer):
            self.basic_layers.append(
                res_block(out_ch*expansion, out_ch, stride=1,expansion=expansion)  
            )

    def forward(self, x):
        for cl in self.basic_layers:
            x =cl(x)
        return x
    
class res_block(nn.Module):
    def __init__(
            self,
            in_ch, out_ch,
            stride, expansion=4, down=False
    ):
        super().__init__()
        self.down_sample = None
        if down: self.down_sample = down_sample(in_ch, out_ch*expansion, stride)
        self.act_layer = nn.ReLU(inplace=False)
        self.basic_layers = nn.Sequential(
            conv_block(in_ch, out_ch, kernel_size=1, pad=0, stride=1),
            conv_block(out_ch, out_ch, kernel_size=3, pad=1, stride=stride),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch*expansion, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_ch*expansion)
        )
    def forward(self, x):
        identity = x #residual connection
        for cl in self.basic_layers:
            x = cl(x)
        #residual connection
        if self.down_sample is not None:
            x += self.down_sample(identity)  
        else:
            x += identity

        x = self.act_layer(x)
        return x 
    
class down_sample(nn.Module): # down sampling for residual connection
    def __init__(
            self,
            in_ch, out_ch,
            stride, bias=False
    ):
        super().__init__()
        self.stride=stride
        self.basic_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        for l in self.basic_layer:
            x = l(x)
        return x
    
class conv_block(nn.Module):
    def __init__(
            self,
            in_ch, out_ch,
            kernel_size, pad, stride, bias=False,
    ):
        super().__init__()
        self.basic_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)  
        )

    def forward(self, x):
        for cl in self.basic_layers:
            x = cl(x)
        return x
    
class TestDecoder(nn.Module):
    def __init__(
            self, expansion
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*expansion, 10)
        
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x,dim=1)
        return x


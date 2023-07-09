import torch
import torch.nn as nn

# Whole model
class U_Net(nn.Module):
    def __init__(
            self,
            in_ch,
            mid_ch,
            out_ch,
            depth,
            kernel_size,
            stride,
            bias,
            pool_size,
            residual 
     ):
        self.depth = depth
        self.encoder = U_encoder(in_ch,mid_ch,depth,kernel_size,stride,bias,pool_size,residual)
        self.decoder = U_decoder(mid_ch*(2**depth),out_ch,depth,kernel_size,stride,bias,pool_size,residual)

    def forward(self,x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out

# Encoder part    
class U_encoder(nn.Module):
    def __init__(
            self,
            in_ch,
            mid_ch,
            depth,
            kernel_size,
            stride,
            bias,
            pool_size,
            residual
    ):
        super().__init__()
        self.in_layer = conv_block(in_ch,mid_ch, kernel_size,stride,bias,residual)
        
        self.layers = nn.ModuleList()
        for i in range(1,depth):
            self.layers.append(U_block(mid_ch * (2**(i-1)),mid_ch * (2**i),kernel_size,stride,bias,residual,pool_size))

    def forward(self,x):
        x = self.in_layer(x)
        features = [x]
        for l in self.layers:
            x = l(x)
            features.append(x)
        return features

#decoder part
class U_decoder(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            depth,
            kernel_size,
            stride,
            bias,
            pool_size,
            residual
    ):
        super().__init__()
        self.up_layers = nn.ModuleList()
        for i in range(depth):
            self.up_layers.append(nn.ConvTranspose2d(in_ch/(2**i),in_ch,pool_size,pool_size,0))

        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            self.conv_layers.append(conv_block(in_ch/(2**i),in_ch/(2**(i+1)),kernel_size,stride,bias,residual))
        self.last_layer = nn.Conv2d(in_ch/(2**depth),out_ch,1,1)

    def forward(self,features):
        features = features.transopose()
        out = features.pop()
        for l, c in zip(self.up_layers,self.conv_layers):
            out = torch.cat(l(out), features.pop(), dim=1)
            out = c(out)
        return self.last_layer(out)

    
# ===================
#  partial structure 
# ===================

class U_block(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size,
            stride,
            bias,
            residual,
            pool_size
    ):
        super().__init__()
        self.pool =  nn.MaxPool2d(pool_size, pool_size)
        self.conv = conv_block(in_ch,out_ch,kernel_size,stride,bias,residual)

    def forward(self,x):
        x = self.pool(x)
        return self.conv(x)

class conv_block(nn.Module):
    def __init__(
            self,
            in_ch, out_ch,
            kernel_size, stride, bias=False,
            residual=False
    ):
        super().__init__()
        self.residual = residual
        self.basic_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size/2), bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),  
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size/2), bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)  
        )

        if residual:
            self.conv_skip == nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size/2), bias=bias)

    def forward(self, x):
        if self.residual:
            return self.basic_layers(x) + self.conv_skip(x)
        else: return self.basic_layers(x)


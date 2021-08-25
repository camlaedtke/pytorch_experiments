import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from models.mix_transformer import mit_b5



def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    

class Segformer(nn.Module):
    def __init__(
        self,
        *,
        pretrained=None,
        dims = (32, 64, 160, 256),
        decoder_dim = 256,
        num_classes = 4,
        feature_strides=[4, 8, 16, 32],
        dropout_ratio=0.1
    ):
        super(Segformer, self).__init__()
        
        self.mit = mit_b5()
        self.mit.init_weights(pretrained=pretrained)
        
        self.linear_c4 = MLP(input_dim=dims[3], embed_dim=decoder_dim)
        self.linear_c3 = MLP(input_dim=dims[2], embed_dim=decoder_dim)
        self.linear_c2 = MLP(input_dim=dims[1], embed_dim=decoder_dim)
        self.linear_c1 = MLP(input_dim=dims[0], embed_dim=decoder_dim)
        
        self.linear_fuse = nn.Conv2d(4 * decoder_dim, decoder_dim, 1)
        
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        
        
    def forward(self, x):
        x = self.mit(x) # len=4, 1/4,1/8,1/16,1/32

        c1, c2, c3, c4 = x 

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        h_out, w_out = c1.size()[2]*4, c1.size()[3]*4

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim = 1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        
        x = F.interpolate(input = x, size = (h_out, w_out), mode = 'bilinear', align_corners = False)
        x = x.type(torch.float32)
    
        return x
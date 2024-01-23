import PIL
import time, json
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat
import collections
import torch.nn as nn


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):

    def __init__(self, dim, heads, dim_heads, dropout):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

#    def forward(self, x, mask=None):
#        for attention, mlp in self.layers:
#            x = attention(x, mask=mask)  # go to attention
#            x = mlp(x)  # go to MLP_Block
#        return x


    def forward(self, x, mask=None):
        x_center = []
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
            index = int(x.shape[1] // 2)
            x_center.append(x[:,index,:])
        return x, x_center 



class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)


class SQSFormer(nn.Module):
    def __init__(self, params):
        super(SQSFormer, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        self.model_type = net_params.get("model_type", 0)

        num_classes = data_params.get("num_classes", 16)
        patch_size = data_params.get("patch_size", 13)
        self.spectral_size = data_params.get("spectral_size", 200)

        depth = net_params.get("depth", 1)
        heads = net_params.get("heads", 8)
        mlp_dim = net_params.get("mlp_dim", 8)
        kernal = net_params.get('kernal', 3)
        padding = net_params.get('padding', 1)
        dropout = net_params.get("dropout", 0)
        conv2d_out = 64
        dim = net_params.get("dim", 64)
        dim_heads = dim
        mlp_head_dim = dim
        
        image_size = patch_size * patch_size

        self.pixel_patch_embedding = nn.Linear(conv2d_out, dim)

        self.local_trans_pixel = Transformer(dim=dim, depth=depth, heads=heads, dim_heads=dim_heads, mlp_dim=mlp_dim, dropout=dropout)
        self.new_image_size = image_size

        self.pixel_pos_embedding = nn.Parameter(torch.randn(self.new_image_size, dim))
        self.pixel_pos_embedding_relative = nn.Parameter(torch.randn(self.new_image_size, dim))
        self.pixel_pos_scale = nn.Parameter(torch.ones(1) * 0.01)
        # self.center_weight = nn.Parameter(torch.ones(depth, 1, 1) * 0.01)
        self.center_weight = nn.Parameter(torch.ones(depth, 1, 1) * 0.001)


        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(kernal, kernal), padding=(padding,padding)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
            # featuremap 
            # nn.Conv2d(in_channels=conv2d_out,out_channels=dim,kernel_size=3,padding=1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU()
        )

        self.senet = SE(conv2d_out, 5)

        self.cls_token_pixel = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent_pixel = nn.Identity()

        self.mlp_head =nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)
        self.dropout = nn.Dropout(0.1)

        linear_dim = dim * 2
        self.classifier_mlp = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, num_classes),
        )


    def centerlize(self, x):
        x = rearrange(x, 'b s h w-> b h w s')
        b, h, w, s = x.shape
        center_w = w // 2
        center_h = h // 2
        center_pixel = x[:,center_h, center_w, :]
        center_pixel = torch.unsqueeze(center_pixel, 1)
        center_pixel = torch.unsqueeze(center_pixel, 1)
        x_pixel = x +  center_pixel
        x_pixel = rearrange(x_pixel, 'b h w s-> b s h w')
        return x_pixel
        

    def get_position_embedding(self, x, center_index, cls_token=False):
        center_h, center_w = center_index
        b, s, h, w = x.shape
        pos_index = []
        for i in range(h):
            temp_index = []
            for j in range(w):
                temp_index.append(max(abs(i-center_h), abs(j-center_w)))
            pos_index.append(temp_index[:])
        pos_index = np.asarray(pos_index)
        pos_index = pos_index.reshape(-1)
        if cls_token:
            pos_index = np.asarray([-1] + list(pos_index))
        pos_emb = self.pixel_pos_embedding_relative[pos_index, :]
        return pos_emb
        


    def encoder_block(self, x):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height
        '''
        x_pixel = x 

        b, s, w, h = x_pixel.shape
        img = w * h
        x_pixel = self.conv2d_features(x_pixel)
        # SQSFormer
        pos_emb = self.get_position_embedding(x_pixel, (h//2, w//2), cls_token=False)
        x_pixel = rearrange(x_pixel, 'b s w h-> b (w h) s') # (batch, w*h, s)
        x_pixel = x_pixel + torch.unsqueeze(pos_emb, 0)[:,:,:] * self.pixel_pos_scale
        x_pixel = self.dropout(x_pixel)
        x_pixel, x_center_list = self.local_trans_pixel(x_pixel) #(batch, image_size+1, dim)
        x_center_tensor = torch.stack(x_center_list, dim=0) # [depth, batch, dim] 
        logit_pixel = torch.sum(x_center_tensor * self.center_weight, dim=0)
        logit_x = logit_pixel 
        reduce_x = torch.mean(x_pixel, dim=1)
        return logit_x, reduce_x

    def forward(self, x,left=None,right=None):
        '''
        x: (batch, s, w, h), s=spectral, w=weigth, h=height

        '''
        logit_x, _ = self.encoder_block(x)
        mean_left, mean_right = None, None
        if left is not None and right is not None:
            _, mean_left = self.encoder_block(left)
            _, mean_right = self.encoder_block(right)

        return  self.classifier_mlp(logit_x), mean_left, mean_right 


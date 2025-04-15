import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange 
from torch.distributions.normal import Normal
import numpy as np
from collections import OrderedDict
import math


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
##########################################################################
##---------- Dual Attention Unit ----------
def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class DualAttention(nn.Module):
    def __init__(
            self,infeat, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(DualAttention, self).__init__()
        modules_body = [conv(infeat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class Upsample2(nn.Module):
    def __init__(self):
        super(Upsample2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(512, 384*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class DEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.CAFF = CAFF(in_ch, in_ch*8)
        # self.DCBlk = DCBlock(out_ch)
        self.conv2 = nn.Conv2d(512, out_ch, 3, 1, 1)

    def forward(self, h, y):
        # h = F.interpolate(h, scale_factor=2)
        h = self.conv(h)
        h_m = nn.LeakyReLU(0.2, inplace=True)(self.CAFF(h, y))
        h_m= self.conv2(h_m)
        # h_s = nn.LeakyReLU(0.2, inplace=True)(self.DCBlk(h, mask, y))
        # weights = self.fc(ti).unsqueeze(-1).unsqueeze(-1)
        # h = weights*h_m + (1-weights)*h_s
        h = h_m + h
        return h

class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x

class D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bn=True):
        super(D_Block, self).__init__()
        self.bn = bn
        self.conv = nn.Conv2d(in_ch, 512, kernel_size, stride, padding, bias=False)
        self.out = nn.Conv2d(512, out_ch, kernel_size, stride, padding, bias=False)
        if bn==True:
            self.batchnorm = nn.BatchNorm2d(512)
        else:
            self.batchnorm = None

    def forward(self, x, c):
        h = self.conv(x)
        if self.bn==True:
            h = self.batchnorm(h)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        text_features = c.view(1,c.shape[1],1,1).expand_as(h)
        h_t = h * text_features
        h_t = self.out(h_t)
        h = h_t + x
        return h
    
class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()

        self.block0 = D_Block(1, 48, 3, 1, 1, bn=False)#128
        self.block1 = D_Block(49, 49, 3, 1, 1)#64
        self.block2 = D_Block(50, 50, 3, 1, 1)#32
        # self.block3 = D_Block(48, 48, 4, 2, 1)#16
        # self.block4 = D_Block(48, 48, 4, 2, 1)#8
        # self.block5 = D_Block(48, 48, 4, 2, 1)#4
        self.conv9 = nn.Conv2d(50, 1, 3, 1, 1)

    def forward(self,out1, out2, out3, features):

        h = self.block0(out1, features)
        h = F.interpolate(h, scale_factor=0.5, mode='bilinear');
        h = self.block1(torch.cat((out2, h), 1), features)
        h = F.interpolate(h, scale_factor=0.5, mode='bilinear');
        h = self.block2(torch.cat((out3, h), 1), features)
        # h = self.block3(h)
        # h = self.block4(h)
        # h = self.block5(h)
        out = self.conv9(h)
        return out


class TextCorrespond(nn.Module):
    def __init__(self, dim, text_channel, amplify=8):
        super(TextCorrespond, self).__init__()

        #d = max(int(dim/reduction), 4)
        d = int(dim*amplify);

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_vis = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, text_channel, 1, bias=False)
        )        
        self.mlp_ir = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, text_channel, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_vis, in_ir, text_features):
        # in_feats: b*c*h*w, text_featurees: 1*512
        x_vis = self.mlp_vis(in_vis);                
        x_ir = self.mlp_ir(in_ir)

        text_features = text_features.view(1,text_features.shape[1],1,1).expand_as(x_ir)
        
        x = x_vis + text_features * x_ir
        return x


class Dual(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 24,
        num_blocks = [2,2,2,2], 
        # num_blocks = [1,1,1,1], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Dual, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        self.patch_embed_b = OverlapPatchEmbed(inp_channels, dim) 
        
        atom_dim = 24
        atom_num = 32 
        # self.dict_generator = RIN(in_dim=inp_channels, atom_num=atom_num, atom_dim=atom_dim) 
        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.fuse1 = DEBlock(dim, dim)
        self.fuse1_b = DEBlock(dim, dim)
        self.fuse2 = DEBlock(dim*2, dim*2)
        self.fuse2_b = DEBlock(dim*2, dim*2)
        self.fuse3 = DEBlock(dim*4, dim*4)
        self.fuse3_b = DEBlock(dim*4, dim*4)
        
        self.spatial_routing_encoder_level1 = DualAttention( atom_dim, dim) 
        self.spatial_routing_encoder_level2 = DualAttention( atom_dim*2, int(dim*2**1))
        self.spatial_routing_encoder_level3 = DualAttention( atom_dim*4, int(dim*2**2))
        self.spatial_routing_encoder_level1_b = DualAttention( atom_dim, dim) 
        self.spatial_routing_encoder_level2_b = DualAttention( atom_dim*2, int(dim*2**1))
        self.spatial_routing_encoder_level3_b = DualAttention( atom_dim*4, int(dim*2**2))


        
        self.channel_routing_latent = DualAttention( atom_dim*8, int(dim*2**3))
        self.channel_routing_latent_b = DualAttention( atom_dim*8, int(dim*2**3))

        self.channel_routing_decoder_level3 = DualAttention( atom_dim*4, int(dim*2**2)) 
        self.channel_routing_decoder_level2 = DualAttention( atom_dim*2, int(dim*2**1)) 
        self.channel_routing_decoder_level1 = DualAttention( atom_dim*2, int(dim*2**1)) 
        self.channel_routing_decoder_level3_b = DualAttention( atom_dim*4, int(dim*2**2)) 
        self.channel_routing_decoder_level2_b = DualAttention( atom_dim*2, int(dim*2**1)) 
        self.channel_routing_decoder_level1_b = DualAttention( atom_dim*2, int(dim*2**1)) 

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_b = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        # B
        self.down1_2_b = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2_b = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3_b = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3_b = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4_b = Downsample(int(dim*2**2)) ## From Level 3 to Level 4

        self.latent_b = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])



        self.up4_3 = Upsample(int(dim*2**4)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3 + 96), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output3 = nn.Conv2d(int(dim*4**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.feature_fusion_1 = Fusion_Embed(embed_dim=dim)
        self.feature_fusion_2 = Fusion_Embed(embed_dim=dim*2)
        self.feature_fusion_3 = Fusion_Embed(embed_dim=dim*2*2)

    def forward(self, vis, ir, text_features): 
        
        # VIS encoder
        inp_enc_level1 = self.patch_embed(vis) #[1, 42, 512, 640]
        inp_enc_level1 = self.spatial_routing_encoder_level1(inp_enc_level1) 
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 
        text_image1 = self.fuse1(out_enc_level1, text_features)


        inp_enc_level2 = self.down1_2(text_image1) 
        inp_enc_level2 = self.spatial_routing_encoder_level2(inp_enc_level2) 
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 
        text_image2 = self.fuse2(out_enc_level2, text_features)


        inp_enc_level3 = self.down2_3(text_image2) 
        inp_enc_level3 = self.spatial_routing_encoder_level3(inp_enc_level3) 
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        text_image3 = self.fuse3(out_enc_level3, text_features)


        inp_enc_level4 = self.down3_4(text_image3)        
        inp_enc_level4 = self.channel_routing_latent(inp_enc_level4)
        latent = self.latent(inp_enc_level4) 

        # IR encoder
        inp_enc_level1_b = self.patch_embed_b(ir) #[1, 42, 512, 640]
        inp_enc_level1_b = self.spatial_routing_encoder_level1_b(inp_enc_level1_b) 
        out_enc_level1_b = self.encoder_level1_b(inp_enc_level1_b) 
        text_image1_b = self.fuse1_b(out_enc_level1_b, text_features)


        inp_enc_level2_b = self.down1_2(text_image1_b) 
        inp_enc_level2_b = self.spatial_routing_encoder_level2_b(inp_enc_level2_b) 
        out_enc_level2_b = self.encoder_level2_b(inp_enc_level2_b) 
        text_image2_b = self.fuse2(out_enc_level2_b, text_features)


        inp_enc_level3_b = self.down2_3_b(text_image2_b) 
        inp_enc_level3_b = self.spatial_routing_encoder_level3_b(inp_enc_level3_b) 
        out_enc_level3_b = self.encoder_level3_b(inp_enc_level3_b) 
        text_image3_b = self.fuse3(out_enc_level3_b, text_features)


        inp_enc_level4_b = self.down3_4_b(text_image3_b)        
        inp_enc_level4_b = self.channel_routing_latent_b(inp_enc_level4_b)
        latent_b = self.latent_b(inp_enc_level4_b)
        
        text1 = self.feature_fusion_1(text_image1, text_image1_b)
        text2 = self.feature_fusion_2(text_image2, text_image2_b)
        text3 = self.feature_fusion_3(text_image3, text_image3_b)
        latent, latent_b = self.cross_attention(latent, latent_b)

        # Decoder          
        inp_dec_level3 = self.up4_3(torch.concat([latent, latent_b], dim=1))
        inp_dec_level3 = torch.cat([inp_dec_level3, text3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = self.channel_routing_decoder_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) # ([1, 96, 128, 160])
        out_3 = self.output3(out_dec_level3)
        

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, text2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.channel_routing_decoder_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) # ([1, 48, 256, 320])
        out_2 = self.output2(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, text1], 1) 
        inp_dec_level1 = self.channel_routing_decoder_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        
        out_dec_level1 = self.refinement(out_dec_level1) 

        out_dec_level1 = self.output(out_dec_level1) # [1, 1, 512, 640]

        return out_dec_level1 , out_2, out_3 # [1, 1, 512, 640], [1, 1, 256, 320], [1, 1, 128, 160]




def Net_G():
    return Dual()

def Net_D():
    return NetD()
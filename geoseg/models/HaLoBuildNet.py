import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from .lwganet import (LWGANet_L0_1242_e32_k11_GELU,
                      LWGANet_L1_1242_e64_k11_GELU_drop01,
                      LWGANet_L2_1442_e96_k11_ReLU)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GMGM(nn.Module):
   
    def __init__(self, c1, c2, c3, reduction=16):
        super(GMGM, self).__init__()
        self.c1, self.c2, self.c3 = c1, c2, c3
        c_total = c1 + c2 + c3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(c_total, c_total // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_total // reduction, 3, 1, bias=True),
        )
        fused_channels = c1 + c2 + c3 
        self.fuse_conv1 = ConvBNReLU(fused_channels, c1)
        self.fuse_conv2 = ConvBNReLU(fused_channels, c2)
        self.fuse_conv3 = ConvBNReLU(fused_channels, c3)

    def forward(self, x1, x2, x3):
        B, _, H1, W1 = x1.shape
        _, _, H2, W2 = x2.shape
        _, _, H3, W3 = x3.shape
        
        x_avg1 = self.avg_pool(x1)
        x_avg2 = self.avg_pool(x2)
        x_avg3 = self.avg_pool(x3)
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg3], dim=1)
        attention_score = self.ca(fea_avg)
        w1, w2, w3 = torch.chunk(attention_score, 3, dim=1)
        
        x1_reweight = x1 * w1
        x2_reweight = x2 * w2
        x3_reweight = x3 * w3
        
        x2_up = F.interpolate(x2_reweight, size=(H1, W1), mode='bilinear', align_corners=False)
        x3_up = F.interpolate(x3_reweight, size=(H1, W1), mode='bilinear', align_corners=False)
        fuse_feature = torch.cat((x1_reweight, x2_up, x3_up), dim=1)
        
        guide_feat_1 = self.fuse_conv1(fuse_feature)
        guide_feat_2 = self.fuse_conv2(fuse_feature)
        guide_feat_3 = self.fuse_conv3(fuse_feature)
        
        guide_feat_2 = F.interpolate(guide_feat_2, size=(H2, W2), mode='bilinear', align_corners=False)
        guide_feat_3 = F.interpolate(guide_feat_3, size=(H3, W3), mode='bilinear', align_corners=False)
        
        return guide_feat_1, guide_feat_2, guide_feat_3
    


class SFFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.dct_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim)
        )
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        B, C, H, W = x.shape
        fft_x = torch.fft.fft2(x, norm='ortho')
        fft_x = torch.fft.fftshift(fft_x)

        h_center, w_center = H // 2, W // 2
        h_radius, w_radius = H // 8, W // 8

        fft_x_low = fft_x[:, :, h_center-h_radius:h_center+h_radius,w_center-w_radius:w_center+w_radius]
        fft_x_low_mean = torch.abs(fft_x_low).mean(dim=2)
        freq_weight = torch.sigmoid(self.dct_mlp(fft_x_low_mean)).view(B, C, 1, 1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn * freq_weight
        
    
class SFFBlock(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)        
        self.attn = SFFM(dim=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    
class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat



class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out

class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

class MGFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim, dim)
        self.li2 = nn.Linear(dim, dim)

        self.qx = DSC(dim, dim)
        self.kx = DSC(dim, dim)
        self.vx = DSC(dim, dim)
        self.projx = DSC(dim, dim)

        self.qy = DSC(dim, dim)
        self.ky = DSC(dim, dim)
        self.vy = DSC(dim, dim)
        self.projy = DSC(dim, dim)

        self.concat_attn = nn.Conv2d(dim * 2, dim, 1)

        self.fusion = nn.Sequential(
            IDSC(dim * 4, dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            DSC(dim, dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x, y):
        B, C, H, W = x.shape
        _, N, _ = y.shape

        y_2d = y.transpose(1, 2).reshape(B, C, H, W)

        avg_x = self.avg(x).squeeze(-1).squeeze(-1) # (B, C)
        avg_y = self.avg(y_2d).squeeze(-1).squeeze(-1) # (B, C)
        
        x_weight = self.li1(avg_x).unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        y_weight = self.li2(avg_y).unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        
        x_att = x * torch.sigmoid(x_weight)
        y_att = y_2d * torch.sigmoid(y_weight)
        
        out1_corr = x_att * y_att

        qy = self.qy(y_att)
        kx = self.kx(x_att)
        vx = self.vx(x_att)
        attnx = self.projx(qy * kx + vx) 

        qx = self.qx(x_att)
        ky = self.ky(y_att)
        vy = self.vy(y_att)
        attny = self.projy(qx * ky + vy) 
        
        out2_cross = self.concat_attn(torch.cat([attnx, attny], dim=1))

        out_final = torch.cat([x, y_2d, out1_corr, out2_cross], dim=1)
        out_final = self.fusion(out_final)
        
        return out_final

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()
        
        e_c1, e_c2, e_c3, e_c4 = encoder_channels

        self.pre_conv = ConvBN(e_c4, e_c3, kernel_size=1)
        self.b4 = SFFBlock(dim=e_c3, num_heads=8, window_size=window_size)
        self.up4_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ffm3 = MGFM(dim=e_c3)
        self.b3 = SFFBlock(dim=e_c3, num_heads=8, window_size=window_size)

        self.up3_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.projection_conv3_2 = Conv(e_c3, e_c2, kernel_size=1) 
        self.ffm2 = MGFM(dim=e_c2)
        self.b2 = SFFBlock(dim=e_c2, num_heads=8, window_size=window_size)

        self.up2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.projection_conv2_1 = Conv(e_c2, e_c1, kernel_size=1)
        self.ffm1 = MGFM(dim=e_c1)

        self.msaam = GMGM(c1=e_c1, c2=e_c2, c3=e_c3)
        
        self.injection_conv3 = ConvBNReLU(e_c3 + e_c3, e_c3, kernel_size=1) 
        self.injection_conv2 = ConvBNReLU(e_c2 + e_c2, e_c2, kernel_size=1) 
        self.injection_conv1 = ConvBNReLU(e_c1 + e_c1, e_c1, kernel_size=1) 

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(e_c1, e_c1),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(e_c1, num_classes, kernel_size=1)
        )
        self.init_weight()

    
    
    def forward(self, res1, res2, res3, res4, h, w):
        guide_feat_1, guide_feat_2, guide_feat_3 = self.msaam(res1, res2, res3)

        x = self.b4(self.pre_conv(res4))

        x_decoder_up = self.up4_3(x)
        x_decoder_seq = x_decoder_up.flatten(2).transpose(1, 2)
        fused3 = self.ffm3(res3, x_decoder_seq) 
        
        guide_feat_3_down = F.interpolate(guide_feat_3, size=fused3.shape[2:], mode='bilinear', align_corners=False)
        x_injected = self.injection_conv3(torch.cat([fused3, guide_feat_3_down], dim=1))
        x = self.b3(x_injected)

        x_decoder_up = self.up3_2(x)
        x_decoder_proj = self.projection_conv3_2(x_decoder_up)
        x_decoder_seq = x_decoder_proj.flatten(2).transpose(1, 2)
        fused2 = self.ffm2(res2, x_decoder_seq) 
        
        guide_feat_2_down = F.interpolate(guide_feat_2, size=fused2.shape[2:], mode='bilinear', align_corners=False)
        x_injected = self.injection_conv2(torch.cat([fused2, guide_feat_2_down], dim=1))
        x = self.b2(x_injected)

        x_decoder_up = self.up2_1(x)
        x_decoder_proj = self.projection_conv2_1(x_decoder_up)
        x_decoder_seq = x_decoder_proj.flatten(2).transpose(1, 2)
        fused1 = self.ffm1(res1, x_decoder_seq) 
        
        guide_feat_1_down = F.interpolate(guide_feat_1, size=fused1.shape[2:], mode='bilinear', align_corners=False)
        x_injected = self.injection_conv1(torch.cat([fused1, guide_feat_1_down], dim=1))
        x = x_injected
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x
        

    
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class HaLoBuildNet(nn.Module):
    def __init__(self,
                 dropout=0.1,
                 num_classes=6
                 ):
        super().__init__()

        self.backbone = LWGANet_L2_1442_e96_k11_ReLU(dropout=dropout) 
        
        self.decoder = Decoder(
            encoder_channels=[96, 192, 384, 768],
            num_classes=num_classes, 
            dropout=dropout
        )

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        output = self.decoder(res1, res2, res3, res4, h, w)
        
        return output

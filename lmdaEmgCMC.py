# -*- coding: utf-8 -*-
# @Time    : 2024/9/1 10:53
# @Author  : sjx_alo！！
# @FileName: lmda.py
# @Algorithm ：
# @Description:


# from torchsummary import summary
import torch.nn as nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
import torch
import math


class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        :arg
        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        # print('查看参数是否变化:', conv.bias)

        return y * self.C * x



class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))



class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)
def block1(depth, F1, D, kernel, num_electrodes, dropout=0.25):
    block = nn.Sequential(
        nn.BatchNorm2d(depth),
        nn.Conv2d(depth, F1, (1, kernel), stride=1, padding=(0, kernel // 2), bias=False),
        nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
        Conv2dWithConstraint(F1,
                             F1* D, (num_electrodes, 1),
                             max_norm=1,
                             stride=1,
                             padding=(0, 0),
                             groups=F1,
                             bias=False), nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
        nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))
    return block


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(127, 127, (1, 25), (1, 1)),
            nn.BatchNorm2d(127),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.7),
        )

        self.rmodel = nn.Sequential(
            Rearrange('b (h) (w) t -> b t (h) (w) '),
            # nn.Conv2d(27, 27, (8, 4), (1, 1)),
            nn.Conv2d(27, emb_size, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(emb_size, momentum=0.01, affine=True, eps=1e-3),
            nn.Dropout(0.7),
            # nn.ReLU(),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.module_list = nn.ModuleList(
            [TransformerEncoderBlock(emb_size) for _ in range(depth)]
        )

    def forward(self, input):
        output = self.cnn(input)
        output = self.rmodel(output)

        for block in self.module_list:
            output = block(output)

        return output


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """
    def __init__(self, chans=22, samples=1125, num_classes=4, depth=20, kernel=75, channel_depth1=24, channel_depth2=9,
                ave_depth=1, avepool=5, encoder_depth=6, emb_size=20):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        self.F1 = 16
        self.F2 = 64
        self.in_depth = 1
        self.D = 2
        self.num_classes = num_classes
        self.num_electrodes = chans
        self.kernel_1 = 128
        self.kernel_2 = 64
        self.dropout = 0.5


        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)
        # nn.init.kaiming_normal_(self.channel_weight.data, nonlinearity='relu')
        # nn.init.normal_(self.channel_weight.data)
        # nn.init.constant_(self.channel_weight.data, val=1/chans)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        self.block1_emg = block1(1, self.F1, self.D, self.kernel_1, 4, self.dropout)
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=0.3))
        # 定义自动填充模块
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out = self.avgPool1(out)
        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        print('In ShallowNet, n_out_time shape: ', n_out_time)
        self.classifier = nn.Sequential(
            nn.Linear(19517, num_classes),
            # nn.ReLU(),
            # nn.Linear(128, num_classes),
            )

        self.transEncoder = TransformerEncoder(encoder_depth, emb_size)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



    def forward(self, x, emg, cmc):
        # x = torch.unsqueeze(x, dim=1)
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # 导联权重筛选

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        eegfeature = torch.flatten(x, 1)

        emgx = self.block1_emg(emg)
        emgx = self.block2(emgx)
        emgx = emgx.flatten(start_dim=1)


        cmcx = self.transEncoder(cmc)
        cmcx = cmcx.contiguous().view(x.size(0), -1)
        print(cmcx.shape)
        features = torch.concat([eegfeature, emgx, cmcx], dim=-1)


        cls = self.classifier(features)
        return cls


if __name__ == '__main__':
    model = LMDA(chans=60,num_classes=4,samples=2500).cuda()
    a = torch.randn(1, 1, 60, 2500).cuda().float()
    b = torch.randn(1, 1, 4, 2500).cuda().float()
    c = torch.randn(1, 127, 4, 2500).cuda().float()

    l2 = model(a,b,c)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    #summary(model,input(True))#, show_input=True
    print(l2.shape)


import torch
from torch import nn
import logging, time, math
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn.parameter import Parameter



class Multi_Scale_Temporal_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, stride=1, residual=True, **kwargs):
        super(Multi_Scale_Temporal_Layer, self).__init__()
        dilations = [1, 2, 3, 4]
        padding = [(temporal_window_size + (temporal_window_size - 1) * (dilation - 1) - 1) // 2 for dilation in
                   dilations]
        num_branches = len(dilations) + 2
        inner_channel = channel // 4
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
                act,
                nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding[i], 0),
                          dilation=(dilations[i], 1), bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
            for i in range(len(dilations))
        ])

#################################################  3*1Max
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
                act,
                nn.MaxPool2d((temporal_window_size, 1), (stride, 1), (padding[0], 0)),
                nn.BatchNorm2d(inner_channel),
            ))
########     1*1
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(inner_channel),
            ))

#############################################################################################
        self.transform = nn.Sequential(
            nn.BatchNorm2d(inner_channel * num_branches),
            act,
            nn.Conv2d(inner_channel * num_branches, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        x = torch.cat(branch_outs, dim=1)  ### 不用拼接直接修改
        x = self.transform(x)
        return x + res


class CA_Block1(nn.Module):
    def __init__(self, in_channel, out_channel, group):
        super(CA_Block1, self).__init__()
        if in_channel != out_channel:
            in_channel = out_channel
        reduction = 1

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channel // reduction), nn.ReLU()
        )

        self.F_h = nn.Conv2d(in_channels=in_channel // reduction, out_channels=in_channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=in_channel // reduction, out_channels=in_channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(in_channel)
        self.avgpool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, None))
        self.groups = group
        #0 1
        self.weight = Parameter(torch.zeros(1, self.groups, 1, 1))
        self.bias = Parameter(torch.ones(1, self.groups, 1, 1))

    def forward(self, x):
        N, C, T, V = x.size()
        input = x
        # (b, c, t, v) --> (b, c, t, 1)  --> (b, c, 1, v)
        x_h = self.avgpool1(x).permute(0, 1, 3, 2)
        # (b, c, t, v) --> (b, c, 1, v)
        x_w = self.avgpool2(x)
        # (b, c, 1, v) cat (b, c, 1, t) --->  (b, c, 1, t+v)
        # (b, c, 1, t+v) ---> (b, c/r, 1, t+v)
        x_cat_conv_relu = self.conv_1x1(torch.cat((x_h, x_w), 3))
        # (b, c/r, 1, t+v) ---> (b, c/r, 1, t)  、 (b, c/r, 1, v)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([T, V], 3)
        # (b, c/r, 1, t) ---> (b, c, t, 1)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # (b, c/r, 1, v) ---> (b, c, 1, v)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        # s_h往宽方向进行扩展， s_w往高方向进行扩展
        xn = s_h.expand_as(x) * s_w.expand_as(x) * x
        ################################################################################
        ##可以加入归一化+缩放伸缩机制  t=a*t+b
        xn = xn.sum(dim=1, keepdim=True)  # (b*32, 1, h, w)
        t = xn.view(N, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(-1, self.groups, T, V)
        ## a= y*ci+R
        t = t * self.weight + self.bias
        t = t.view(N, 1, T, V)
        x = x * self.sigmoid_h(t)
        x = x.view(-1, C, T, V)
        return x


class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


class ChannelShuffle(nn.Module):
    def __init__(self, in_channel, out_channel, groups, A):
        super(ChannelShuffle, self).__init__()
        self.out_channels = out_channel
        self.groups = groups
#################################################################
#################################################################
        self.anl_channels=self.out_channels//4
#################################################################
#################################################################
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, self.out_channels, kernel_size=1),
                                   nn.Conv2d(self.out_channels, self.anl_channels, kernel_size=1, groups=self.groups,
                                             stride=1),
                                   # nn.BatchNorm2d(self.out_channels), nn.ReLU()
                                   )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.anl_channels, self.out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel))
        self.A = A.numpy()

        self.num_point = 25

        self.DecoupleA = nn.Parameter(
            torch.tensor(np.reshape(self.A.astype(np.float32), [3, 1, self.num_point, self.num_point]),
            # [3 c 25 25]修改
            dtype=torch.float32, requires_grad=True).repeat(1, self.groups, 1, 1), requires_grad=True)

        eye_array = []
        for i in range(self.anl_channels):  ############
            eye_array.append(torch.eye(self.num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)  # [16 25 25]
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001) ** (-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A


    def forward(self, x):
        learn_A = self.DecoupleA.repeat(1, self.anl_channels // self.groups, 1, 1)
        #[3,c,v v]
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])],0)
        K, C, V, V = norm_learn_A.size()  #[3,c,v v]
        norm_learn_A = norm_learn_A.view(K, self.groups , -1, V, V)
        norm_learn_A = torch.transpose(norm_learn_A, 1, 2).contiguous().view(K, self.groups, -1, V, V)  # 洗牌  # 洗牌
        n, c, t, v = x.size()

        x = self.conv2(x)
        x = x.view(n, self.groups, -1, t, v)
        x = torch.einsum('ngctv,kgcvw->ngctw', (x, norm_learn_A)).view(n, -1, t, v)

        x = self.conv3(x)

        return x


########      S-GC
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, A, bias, act, edge, parts, body, **kwargs):
        super(SpatialGraphConv, self).__init__()
        self.parts = parts
        self.body = body
        self.s_kernel_size = max_graph_distance + 1  #####  333333
####################################################################
####################################################################
        self.groups = 4  ############
####################################################################
####################################################################
        self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        self.gcn = nn.Conv2d(in_channel, out_channel * self.s_kernel_size, 1, bias=bias)
        self.channel_shuffle = ChannelShuffle(in_channel, out_channel, self.groups, self.A)
        self.CA_A = CA_Block1(out_channel // self.groups, out_channel // self.groups, self.groups)

        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1

        if in_channel == out_channel:
            self.residual = nn.Identity()
        elif in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )

        self.concat = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, kernel_size=1),
                                    nn.BatchNorm2d(out_channel), nn.ReLU(),
                                    nn.Conv2d(out_channel, out_channel, kernel_size=1),
                                    nn.BatchNorm2d(out_channel)
                                    )

        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act

    def forward(self, x):  # x=[n c t v m]
        n, c, t, v = x.size()
        res = self.residual(x)
        x_res = self.gcn(x)
        n, kc, t, v = x_res.size()
        x_res = x_res.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        x_res = torch.einsum('nkctv,kvw->nctw', (x_res, self.A * self.edge)).contiguous()
##########################################################################################################
        x_shuffle = self.channel_shuffle(x)
        x_shuffle = x_shuffle.view(n * self.groups, -1, t, v)
        x_shuffle = self.CA_A(x_shuffle)
        x_shuffle = x_shuffle.view(n, self.groups, -1, t, v).view(n, -1, t, v)
        x = self.act(self.concat(torch.cat([self.bn(x_res), x_shuffle], dim=1)) + res)
        return x


class STA_GC(nn.Module):
    def __init__(self, in_channel, out_channel, A, kernel_size, stride, depth, t_scale, **kwargs):  ####**kwargs 传参
        super(STA_GC, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size
        num_T = 288 // t_scale
        self.in_channel = in_channel
        if in_channel < 64:  # init block
            self.STAGC = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                SpatialGraphConv(in_channel, out_channel, max_graph_distance, A, **kwargs),  #####S-GC
            )

        else:
            self.STAGC = nn.Sequential(
                SpatialGraphConv(in_channel, out_channel, max_graph_distance, A, **kwargs),
            )

        self.TCN = nn.Sequential(
            Multi_Scale_Temporal_Layer(out_channel, temporal_window_size, stride=stride, **kwargs),
            Multi_Scale_Temporal_Layer(out_channel, temporal_window_size, stride=1, **kwargs),
        )

    def forward(self, x):

        x = self.STAGC(x)
        x = self.TCN(x)

        return x





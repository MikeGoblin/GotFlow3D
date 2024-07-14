import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flot.gconv import SetConv


class MotionEncoder(nn.Module):
    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.conv_corr = nn.Conv1d(64, 64, 1)  ###### *modified* ######
        self.conv_flow = nn.Conv1d(3, 64, 1)
        self.conv = nn.Conv1d(64+64, 64-3, 1)

    def forward(self, flow, corr):
        # flow和corr分别是流动信息和相关性信息，将两者进行编码与合并
        cor = F.relu(self.conv_corr(corr)) 
        flo = F.relu(self.conv_flow(flow.transpose(1, 2).contiguous()))
        cor_flo = torch.cat([cor, flo], dim=1)
        out_conv = F.relu(self.conv(cor_flo))
        # 最终输出将out_conv和原始的流动信息结合
        out = torch.cat([out_conv, flow.transpose(1, 2).contiguous()], dim=1)
        return out


class ConvGRU(nn.Module): # 处理时序的GRU
    def __init__(self, input_dim=128, hidden_dim=64):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convr = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convq = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        # h是隐藏状态，x是输入
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        rh_x = torch.cat([r*h, x], dim=1)
        q = torch.tanh(self.convq(rh_x))

        h = (1 - z) * h + z * q
        return h


class ConvRNN(nn.Module): # 用来和GRU做对比
    def __init__(self, input_dim=128, hidden_dim=64):
        super(ConvRNN, self).__init__()
        self.convx = nn.Conv1d(input_dim, hidden_dim, 1)
        self.convh = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        xt = self.convx(x)
        ht = self.convh(h)

        h = torch.tanh(xt + ht)
        return h


class FlowHead(nn.Module): # 用来输出最终的流动估计
    def __init__(self, input_dim=128):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.setconv = SetConv(64, 64)
        self.out_conv = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x, graph):
        out = self.conv1(x)
        out_setconv = self.setconv(x.transpose(1, 2).contiguous(), graph).transpose(1, 2).contiguous()
        out = self.out_conv(torch.cat([out_setconv, out], dim=1))
        return out


class UpdateBlock(nn.Module): # 用来更新流动估计的模块，结合了上述所有模块
    def __init__(self, input_dim=128, hidden_dim=64):
        super(UpdateBlock, self).__init__()
        self.motion_encoder = MotionEncoder()
        self.gru = ConvGRU(input_dim=input_dim, hidden_dim=hidden_dim)
        self.flow_head = FlowHead(input_dim=hidden_dim)

    def forward(self, net, inp, corr, flow, graph):
        motion_features = self.motion_encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)  # 128d
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net, graph).transpose(1, 2).contiguous()
        return net, delta_flow

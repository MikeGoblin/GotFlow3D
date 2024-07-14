import torch.nn as nn

from model.flot.gconv import SetConv
from model.flot.graph import Graph


class FlotEncoder(nn.Module): # 这个类用于将点云数据编码为高维数据表示，并构建点云的图结构
    def __init__(self, num_neighbors=32):
        super(FlotEncoder, self).__init__()
        n = 32 # 特征维度
        self.num_neighbors = num_neighbors

        self.feat_conv1 = SetConv(3, n) # 从3维输入(点云的x,y,z坐标)到n维特征的卷积
        self.feat_conv2 = SetConv(n, 2 * n) # 从n维特征到2*n维特征的卷积
        self.feat_conv3 = SetConv(2 * n, 4 * n) # 从2*n维特征到4*n维特征的卷积

    def forward(self, pc):
        # pc为输入的点云数据，形状是 [B, N, 3]
        graph = Graph.construct_graph(pc, self.num_neighbors)
        x = self.feat_conv1(pc, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)
        x = x.transpose(1, 2).contiguous() # B,C,N

        return x, graph

class FlotGraph(nn.Module):
    def __init__(self, num_neighbors=32):
        super(FlotGraph, self).__init__()

        self.num_neighbors = num_neighbors

    def forward(self, pc):
        # 用于构建图结构
        graph = Graph.construct_graph(pc, self.num_neighbors)

        return graph

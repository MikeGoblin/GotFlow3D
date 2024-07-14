import torch
import torch.nn as nn

from model.extractor import FlotEncoder, FlotGraph
from model.corr2 import CorrBlock2
from model.update import UpdateBlock
from model.scale import KnnDistance
import model.ot as ot
from model.model_dgcnn import GeoDGCNN_flow2


class RSF_DGCNN(nn.Module):
    def __init__(self, args):
        super(RSF_DGCNN, self).__init__()
        # 隐藏层维度和上下文特征维度
        self.hidden_dim = 64
        self.context_dim = 64
        
        # 使用GeoDGCNN_flow2进行特征提取，k=32表示每个点考虑的32个最近邻
        self.feature_extractor = GeoDGCNN_flow2(k=32, emb_dims=1024, dropout=0.5)

        # 使用FlotEncoder进行上下文特征提取
        self.context_extractor = FlotEncoder()
        # self.graph_extractor = FlotGraph()

        # CorrBlock2模块用于计算特征间的相关性矩阵
        self.corr_block = CorrBlock2(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k)

        # UpdateBlock模块用于递归更新流动估计
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)

        # 这三个参数用于调整尺度偏移和Sinkhorn算法中的参数
        self.scale_offset = nn.Parameter(torch.ones(1)/2.0) # torch.ones(1)/10.0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, p, num_iters=12): # 一共有12个时间帧
        ## 输入与特征提取 ##
        # p是一个包含两个点云的列表，xyz1和xyz2分别表示两个点云的三维坐标
        [xyz1, xyz2] = p # B x N x 3
        
        # fmap1和fmap2通过特征提取器从p[0]和p[1]中提取特征映射
        fmap1 = self.feature_extractor(p[0])
        fmap2 = self.feature_extractor(p[1])

        ## 计算最近邻距离和体素尺度 ##
        ## modified scale ##
        # nn_distance计算p[0]中的每个点的3个最近邻距离
        nn_distance = KnnDistance(p[0], 3)
        # voxel_scale根据scale_offset和最近邻距离调整体素尺度
        voxel_scale = self.scale_offset * nn_distance

        # correlation matrix
        ## 最优传输计算 ##
        # transport使用Sinkhorn算法计算fmap1和fmap2之间的最优传输计划，epsilon和gamma是算法的参数
        transport = ot.sinkhorn(fmap1.transpose(1,-1), fmap2.transpose(1,-1), xyz1, xyz2, 
            epsilon=torch.exp(self.epsilon) + 0.03, 
            gamma=self.gamma, #torch.exp(self.gamma), 
            max_iter=1)
        
        self.corr_block.init_module(fmap1, fmap2, xyz2, transport)

        ## 上下文特征提取 ##
        # fct1和graph_context通过上下文特征提取器从p[0]中提取特征
        fct1, graph_context = self.context_extractor(p[0])

        # net和inp通过拆分fct1得到
        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        ## 流动估计递归更新 ##
        coords1, coords2 = xyz1, xyz1
        
        # flow_predictions用于存储每次迭代的流动预测，all_delta_flow用于存储每次迭代的流动增量
        flow_predictions = []
        all_delta_flow = []  

        # 通过相关性模块计算相关性矩阵，更新流动估计，并将结果存储在flow_predictions中
        for itr in range(num_iters):
            coords2 = coords2.detach()
            corr = self.corr_block(coords=coords2, all_delta_flow=all_delta_flow, num_iters=num_iters, scale=voxel_scale)  
            flow = coords2 - coords1
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
            all_delta_flow.append(delta_flow)  
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)

        return flow_predictions

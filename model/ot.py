import torch


def sinkhorn(feature1, feature2, pcloud1, pcloud2, epsilon, gamma, max_iter):
    # 两个点云的数量不一定相同，feature1和feature2分别是两个点云经过编码后的特征，pcloud1和pcloud2则是点云的原始数据
    """
    Sinkhorn algorithm

    Parameters
    ----------
    feature1 : torch.Tensor
        Feature for points cloud 1. Used to computed transport cost. 
        Size B x N x C.
    feature2 : torch.Tensor
        Feature for points cloud 2. Used to computed transport cost. 
        Size B x M x C.
    pcloud1 : torch.Tensor
        Point cloud 1. Size B x N x 3.
    pcloud2 : torch.Tensor
        Point cloud 2. Size B x M x 3.
    epsilon : torch.Tensor
        Entropic regularisation. Scalar.
    gamma : torch.Tensor
        Mass regularisation. Scalar.
    max_iter : int
        Number of unrolled iteration of the Sinkhorn algorithm.

    Returns
    -------
    torch.Tensor
        Transport plan between point cloud 1 and 2. Size B x N x M.
    """
    
    # 计算点云1和点云2之间的平方L2距离矩阵distance_matrix
    distance_matrix = torch.sum(pcloud1 ** 2, -1, keepdim=True)
    distance_matrix = distance_matrix + torch.sum(
        pcloud2 ** 2, -1, keepdim=True
    ).transpose(1, 2)
    distance_matrix = distance_matrix - 2 * torch.bmm(pcloud1, pcloud2.transpose(1, 2))
    
    # 用于限制传输距离在10米以内的点对
    support = (distance_matrix < 10 ** 2).float()

    # Transport cost matrix
    # feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    # feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    # C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))
    
    # 计算点云特征之间的相关性
    corr = torch.matmul(feature1, feature2.transpose(1, 2))
    # 计算特征的L2范数
    n1 = torch.norm(feature1, dim=2, keepdim=True)
    n2 = torch.norm(feature2, dim=2, keepdim=True)
    # 计算传输成本矩阵
    C = 1.0 - corr / torch.matmul(n1, n2.transpose(1, 2))

    # 使用传输成本矩阵C和熵正则化参数epsilon计算正则化的传输成本矩阵K
    K = torch.exp(-C / epsilon) * support

    # Early return if no iteration (FLOT_0)
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob1 = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob2 = (
        torch.ones(
            (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
        )
        / K.shape[2]
    )

    # Sinkhorn algorithm，通过迭代更新变量a和b，逐步逼近最优传输方案
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))

    return T

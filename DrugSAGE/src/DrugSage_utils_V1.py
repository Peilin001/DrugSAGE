import sys, os
import torch
import random
import numpy as np
import math

#from sklearn.utils import shuffle
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.DrugSage_models_V1 import *

##############################################################################
#### functions for get_r2_numpy_manual
##############################################################################
def get_r2_numpy_manual(x, y):
    zx = (x-np.mean(x))/np.std(x, ddof=1)
    zy = (y-np.mean(y))/np.std(y, ddof=1)
    r = np.sum(zx*zy)/(len(x)-1)
    return r**2


##############################################################################
#### functions for sampling
##############################################################################

def sampling(src_nodes, sample_num, neighbor_table):
    """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    
    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表
    
    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        if len(neighbor_table[sid]) == 0:
            continue
        res = np.random.choice(neighbor_table[sid], size=(sample_num, ))
        ################# res = np.random.choice(neighbor_table[sid], size=(sample_num, ), replace=False)
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样
    
    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射
    
    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result

def get_gnnmask_embeddings(gnn_model, dataCenter, device):
    features           = torch.FloatTensor(getattr(dataCenter, 'feats')).to(device)
    nodes              = np.arange(len(getattr(dataCenter, 'labels'))).tolist()
    adjacency_dict     = getattr(dataCenter, 'adjacency_dict')
    NUM_NEIGHBORS_LIST = gnn_model.num_neighbors_list

    embs = []
    
    batch_src_index = list(range(len(nodes)))
    batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, adjacency_dict)
    batch_sampling_x = [features[idx].float().to(device) for idx in batch_sampling_result]
    embs_batch = gnn_model(batch_sampling_x)
    embs.append(embs_batch)
    
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    #return embs.detach()
    
    ### 07062024
    return embs.detach(), batch_sampling_result


def get_adjacency_dict(dataCenter, new_int_file):
    node_map  = getattr(dataCenter, 'node_map')
    adj_lists = getattr(dataCenter, 'adj_lists')

    with open(new_int_file) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            if not info[0] in node_map:
                n = len(node_map)
                node_map[info[0]] = n

            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            
    adjacency_dict = defaultdict(list)
    for index in range(len(adj_lists)):
        a = adj_lists[index]
        a_list = list(a)
        adjacency_dict[index] = a_list

    return adjacency_dict, node_map




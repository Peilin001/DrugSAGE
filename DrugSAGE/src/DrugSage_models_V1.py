import sys, os
import torch
import random
import numpy as np
import math

from sklearn.utils import shuffle
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.models_V1 import *
from src.customized_linear import *

##############################################################################
#### functions for NeighborAggregator
##############################################################################

class CustomizedLinearNeighborAggregator(nn.Module):
    def __init__(self, customizedLinear, aggr_method="mean"):
        super(CustomizedLinearNeighborAggregator, self).__init__()
        self.aggr_method = aggr_method
        self.customizedLinear = customizedLinear
    
    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor, index = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}".format(self.aggr_method))
        
        neighbor_hidden = self.customizedLinear(aggr_neighbor)

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.customizedLinear.input_features, self.customizedLinear.output_features, self.aggr_method)

##############################################################################
#### functions for SageCustomizedLinear
##############################################################################

class SageCustomizedLinear(nn.Module):
    def __init__(self, customizedLinear,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        
        super(SageCustomizedLinear, self).__init__()

        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]

        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method   = aggr_hidden_method
        self.activation           = activation
        self.customizedLinear     = customizedLinear
        self.aggregator = CustomizedLinearNeighborAggregator(customizedLinear, aggr_method=aggr_neighbor_method)
        self.input_dim  = customizedLinear.input_features
        self.output_dim = customizedLinear.output_features
        
        if self.aggr_hidden_method == "concat":
            self.output_dim = customizedLinear.output_features * 2
    
    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden     = self.customizedLinear(src_node_features)
        
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden  ### 06122023
            
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))
        
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_hidden_method)


##############################################################################
#### functions for GraphSageMask no hop
##############################################################################

class GraphSageMask_nohop(nn.Module):
    def __init__(self, 
                 pathway_mask, 
                 emb_dim, 
                 num_neighbors_list, 
                 aggr_neighbor_method = "mean",
                 aggr_hidden_method   = "sum", 
                 activation_function  = "Tanh"):
        
        super(GraphSageMask_nohop, self).__init__()
        
        self.num_neighbors_list = num_neighbors_list
        self.aggr_hidden_method = aggr_hidden_method
        self.pathway_mask = pathway_mask
        self.n_pathways   = self.pathway_mask.shape[1]
        self.input_dim    = self.n_pathways
        
        if activation_function == 'Tanh':
            self.tanh = torch.nn.Tanh()
        if activation_function == 'Sigmoid':
            self.tanh = torch.nn.Sigmoid()
        if activation_function == 'ReLU':
            self.tanh = torch.nn.ReLU()
        
        
        if aggr_hidden_method == "concat":
            self.input_dim = 2 * self.n_pathways
            self.bn2 = nn.BatchNorm1d(2*emb_dim)
            self.net = nn.Sequential(nn.Dropout(0.2), nn.BatchNorm1d(2 * self.n_pathways))
        
        if aggr_hidden_method == "sum":
            self.net = nn.Sequential(nn.BatchNorm1d(self.n_pathways), nn.Dropout(0.2))
        
        
        self.customizedLinear = CustomizedLinear(pathway_mask)
        
        self.sagenet = SageCustomizedLinear(self.customizedLinear, 
                                            activation=self.tanh, 
                                            aggr_neighbor_method=aggr_neighbor_method, 
                                            aggr_hidden_method=aggr_hidden_method )
        
    def forward(self, node_features_list):
        
        hidden = node_features_list
        next_hidden = []
        
        src_node_features = hidden[0]
        src_node_num = len(src_node_features)  ## batch_size, e.g., 20
        neighbor_node_features = hidden[1].view((src_node_num, self.num_neighbors_list[0], -1))
        
        h = self.sagenet(src_node_features, neighbor_node_features)
        h2 = self.net(h)
        #next_hidden.append(h2)
        #hidden = next_hidden
         
        #return hidden[0]
        return h2




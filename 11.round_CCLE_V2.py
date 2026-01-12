
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import sys, os
import torch
import random
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

from src.DataCenter_V1 import *
from src.models_V1 import *

from src.DrugSage_V2 import *
from src.DrugSage_models_V1 import *
from src.DrugSage_utils_V1 import *

from src.GraphSageMask_utils_V1 import *

########################################################################
################################# CCLE #################################
########################################################################

################### argv

ccle_drug_name = sys.argv[1]
R = sys.argv[2]
N = int(sys.argv[3])

########################################################################
ulm = 'margin' ### !!!
an  = 'max'    ### !!!
ah  = 'sum'    ### --- sum or concat
af  = 'Sigmoid'
lr  = 0.001
wd  = 1e-5
opt = 'Adam'
b_sz = 128
lm   = 'net_regression'
nn_af = 'ReLU'
########################################################################

drugs = ["17-AAG", "AEW541", "AZD0530", "AZD6244", "Erlotinib", "Irinotecan", "L-685458", "Lapatinib", "LBW242", "Nilotinib", "Nutlin-3", "Paclitaxel", "Panobinostat", "PD-0325901", "PD-0332991", "PF2341066", "PHA-665752", "PLX4720", "RAF265", "Sorafenib", "TAE684", "TKI258", "Topotecan", "ZD-6474"]

pathway_file = "data/CCLE/KEGG.CCLE.F124.gmt"

########################################################################

for k in range(len(drugs)):
    drug     = drugs[k]
    drug_idx = k+1
    if drug != ccle_drug_name:
        continue
    
    ccle_label_file_2 = 'data/CCLE/cell_label/'+drug+'.cell_label.txt'
    ccle_label_file   = 'data/CCLE/CCLE__Y.txt'
    ccle_inter_file   = 'data/CCLE/KEGG.E'+R+'.CCLE.cites'
    ccle_feate_file   = 'data/CCLE/KEGG.INTC.CCLE.tsv'
    
    for rount in range(10):
        best_fold = DrugSage_V2(drug, drug_idx, rount, 'CCLE', pathway_file, ccle_feate_file, ccle_label_file, ccle_inter_file, ccle_label_file_2, 1.0, 1.0, 
                                              ulm, an, ah, R, [N],
                                              af,
                                              nn_af,
                                              lr,
                                              wd,
                                              opt,
                                              b_sz,
                                              lm
                                              )
    
    
    



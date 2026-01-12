
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

gdsc_drug_name = sys.argv[1]
R = sys.argv[2]
N = int(sys.argv[3])


ulm = 'margin' ### !!!
an  = 'max'    ### !!!
ah  = 'sum'    ### --- sum or concat
af  = 'ReLU'
af  = 'Sigmoid'
lr  = 0.001
wd  = 1e-5
b_sz  = 128
opt   = 'Adam'
lm    = 'net_regression'
nn_af = "ReLU"

########################################################################

drug_info = pd.read_csv("/home/peilin/work/23-GNN-Drug/data/Pathways/KEGG/GDSC1/GDSC1__24Jul22__Y.drug_names.V2.txt",sep='\t',header=None)
drugs = drug_info.iloc[:,2]
drug_index = drug_info.loc[:,0]

pathway = 'KEGG'
pathway_file = "/home/peilin/work/23-GNN-Drug/data/Pathways/KEGG/KEGG.GDSC2.F124.gmt"

########################################################################

for k in range(len(drugs)):
    
    drug     = drugs[k]
    drug_idx = drug_index[k]
    
    if drug != gdsc_drug_name:
        continue
    
    print("##################################\n"+"Run for "+drug+", drug_idx = "+str(drug_idx)+"\n##################################\n")
    
    ccle_label_file   = 'data/GDSC1__24Jul22__Y.V2.txt'
    ccle_label_file_2 = 'data/GDSC1/cell_label/'+drug+'.cell_label.txt'
    ccle_inter_file   = 'data/GDSC1/KEGG.E'+R+'.GDSC2.cites'
    ccle_feate_file   = 'data/GDSC1/KEGG.INTC.GDSC2.tsv'
    
    for rount in range(10):
        best_fold = DrugSage_V2(drug, drug_idx, rount, 'GDSC1', pathway_file, ccle_feate_file, ccle_label_file, ccle_inter_file, ccle_label_file_2, 1.0, 1.0, 
                                              ulm, an, ah, R, [N],
                                              af,
                                              nn_af,
                                              lr,
                                              wd,
                                              opt,
                                              b_sz,
                                              lm
                                              )
    



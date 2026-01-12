#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import sys, os
import torch
import random
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

from src.DataCenter_V1 import *
from src.models_V1 import *
from src.DrugSage_models_V1 import *
from src.DrugSage_utils_V1 import *
from src.GraphSageMask_utils_V1 import *
from src.get_fixed_embeddings_prediction import *

def DrugSage_V2(drug, drug_idx, rount, tag, pathway_file, ccle_feate_file, ccle_label_file, ccle_inter_file, ccle_label_file_2, 
                      unsupervised_loss_ratio, cell_loss_ratio, unsup_loss_method, aggr_neighbor_method, aggr_hidden_method, R, NUM_NEIGHBORS_LIST,
                      activation_function,
                      nn_af,
                      learning_rate,
                      weight_decay,
                      optimier_method,
                      batch_size = 32,
                      learn_method = 'net_regression'
                      ):
    print('Working on i =', drug_idx, ', drug =', drug)
    
    print("ccle_feate_file   = ", ccle_feate_file)
    print("ccle_label_file   = ", ccle_label_file)
    print("ccle_inter_file   = ", ccle_inter_file)
    print("ccle_label_file_2 = ", ccle_label_file_2)
    
    print("unsupervised_loss_ratio =", unsupervised_loss_ratio)
    print("cell_loss_ratio         =", cell_loss_ratio)
    print("unsup_loss_method       =", unsup_loss_method)
    print("aggr_neighbor_method    =", aggr_neighbor_method)
    print("aggr_hidden_method      =", aggr_hidden_method)
    print("NUM_NEIGHBORS_LIST      =", NUM_NEIGHBORS_LIST)
    print("ccle_inter_file         =", ccle_inter_file)
    print("learning_rate           =", learning_rate)
    print("weight_decay            =", weight_decay)
    print("optimier_method         =", optimier_method)
    
    device = torch.device("cpu")
    b_sz = batch_size
    
    if learn_method == 'category':
        dataCenter = DataCenterL1()
        dataCenter.load_dataSet(ccle_feate_file, ccle_label_file, ccle_inter_file, "classification", drug_idx)
    else:
        dataCenter = DataCenterL2()
        dataCenter.load_dataSet(ccle_feate_file, ccle_label_file, ccle_label_file_2, ccle_inter_file, "regression", drug_idx)
    
    features  = torch.FloatTensor(getattr(dataCenter, 'feats')).to(device)
    adj_lists = getattr(dataCenter,'adj_lists')
    adjacency_dict = getattr(dataCenter, 'adjacency_dict')
    node_map  = getattr(dataCenter, 'node_map')
    num_nodes = getattr(dataCenter, 'num_nodes')
    labels    = getattr(dataCenter, 'labels')

    features  = torch.FloatTensor(getattr(dataCenter, 'feats')).to(device)
    adj_lists = getattr(dataCenter,'adj_lists')
    adjacency_dict = getattr(dataCenter, 'adjacency_dict')
    node_map  = getattr(dataCenter, 'node_map')
    num_nodes = getattr(dataCenter, 'num_nodes')
    labels    = getattr(dataCenter, 'labels')
    
    data  = pd.read_csv(ccle_feate_file,delimiter='\t',index_col=0).astype(np.float32)
    genes = data.columns.tolist()
    pathway_dict = read_gmt(pathway_file, min_g=20, max_g=1000)
    pathway_mask = create_pathway_mask(genes, pathway_dict, add_missing=1, fully_connected=True)
    n_cell_lines = len(features)
    n_pathways   = len(pathway_dict)
    print("pathway_file", pathway_file)
    print('# genes:', len(genes))
    print("# cell_lines:", n_cell_lines)
    print("# pathways:", n_pathways)
    print("pathway_mask.shape:", pathway_mask.shape)

    name = drug + '.'+ tag + '.R'+ R + 'N' + str(NUM_NEIGHBORS_LIST[0]) +'.round_'+str(rount)
    print('name = ', name)
    
    pseudo_X = range(num_nodes)
    labels_binary = []
    for k in range(len(labels)):
        if(labels[k] == -9):
            labels_binary.append(0)
        else:
            labels_binary.append(1)
    
    nfold = 10
    kfold = StratifiedKFold(n_splits=nfold, shuffle=True)
    kf    = KFold(n_splits=nfold, shuffle=True)
    
    train_loss_list = []
    test_loss_list  = []
    train_idx_list  = []
    test_idx_list   = []
    for i, (train_idx, test_idx) in enumerate(kfold.split(pseudo_X, labels_binary)):
        train_idx_list.append(train_idx)
        test_idx_list.append(test_idx)
        print('------------------------------Fold %d-------------------------------' % i)
        graphSageMask, regression_1, regression_2, train_loss, test_loss = kfold_train_test_DrugSage(dataCenter, pathway_mask, 
                                                                            device, tag, name, b_sz,
                                                                            train_idx, test_idx, i, 
                                                                            unsupervised_loss_ratio, cell_loss_ratio,
                                                                            epochs = 100,
                                                                            learn_method       = learn_method, 
                                                                            aggr_neighbor_func = aggr_neighbor_method, 
                                                                            aggr_hidden_func   = aggr_hidden_method,
                                                                            unsupervised_loss_method = unsup_loss_method,
                                                                            NUM_NEIGHBORS_LIST = NUM_NEIGHBORS_LIST,
                                                                            activation_function = activation_function,
                                                                            learning_rate = learning_rate,
                                                                            weight_decay = weight_decay,
                                                                            optimier_method = optimier_method)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        models = [graphSageMask, regression_1, regression_2, train_idx, test_idx, train_loss, test_loss]
        torch.save(models, 'models/'+tag+'/{}.fold_{}.torch'.format(name, i))
    
    info = [train_idx_list, test_idx_list, train_loss_list, test_loss_list]
    torch.save(info, 'models/'+tag+'/{}.info.torch'.format(name))

    lst=[np.ones(nfold), np.ones(nfold), np.ones(nfold),  np.ones(nfold), np.ones(nfold), np.ones(nfold) ]
    arr = np.array(lst)
    final_res = pd.DataFrame(arr).T

    labels0 = getattr(dataCenter, 'labels')
    best_fold = 0
    best_test_R2 = 0

    for fold in range(nfold):
        [graphSageMask_new, regression_1_new, regression_2_new, train_index, test_index, train_loss, test_loss] = torch.load('models/'+tag+'/{}.fold_{}.torch'.format(name, fold))
        embeddings,batch_sampling_result = get_gnnmask_embeddings(graphSageMask_new, dataCenter, device) #####
        
        y_predict = regression_2_new(regression_1_new(embeddings))
        y2 = y_predict.detach().numpy()
        y2 = y2.flatten()
    
        ### all
        non_na_index = labels0 != -9
        y2_all_9 = y2[non_na_index]
        y1_all_9 = labels0[non_na_index]
        y_res = pd.DataFrame({"Y":y1_all_9,"Y_pred":y2_all_9})
        pcc_all = y_res['Y'].corr(y_res['Y_pred'])
        r2_all = get_r2_numpy_manual(y_res['Y_pred'], y_res['Y'])
        final_res.iloc[fold:(fold+1),0:1] = pcc_all
        final_res.iloc[fold:(fold+1),1:2] = r2_all
    
        ### train
        y2_train = y2[train_index]
        y1_train = labels0[train_index]
        non_na_index = y1_train != -9
        y2_train_9 = y2_train[non_na_index]
        y1_train_9 = y1_train[non_na_index]
        y_res = pd.DataFrame({"Y":y1_train_9,"Y_pred":y2_train_9})
        pcc_train = y_res['Y'].corr(y_res['Y_pred'])
        r2_train = get_r2_numpy_manual(y_res['Y_pred'], y_res['Y'])
        final_res.iloc[fold:(fold+1),2:3] = pcc_train
        final_res.iloc[fold:(fold+1),3:4] = r2_train
    
        ### test
        y2_test = y2[test_index]
        y1_test = labels0[test_index]
        non_na_index = y1_test != -9
        y2_test_9 = y2_test[non_na_index]
        y1_test_9 = y1_test[non_na_index]
        y_res = pd.DataFrame({"Y":y1_test_9,"Y_pred":y2_test_9})
        pcc_test = y_res['Y'].corr(y_res['Y_pred'])
        r2_test = get_r2_numpy_manual(y_res['Y_pred'], y_res['Y'])
        final_res.iloc[fold:(fold+1),4:5] = pcc_test
        final_res.iloc[fold:(fold+1),5:6] = r2_test
        print("Fold: "+str(fold)+", r2_all = "+str(r2_all)+", r2_train = " + str(r2_train) + ", r2_test = "+str(r2_test))
        if r2_test > best_test_R2:
            best_test_R2 = r2_test
            best_fold = fold
        
    final_res.to_csv('models/'+tag+'/'+name+'.kfold_R2.txt', index=False)
    return best_fold


##############################################################################
#### functions k-fold cross-validation
##############################################################################

def kfold_train_test_DrugSage(dataCenter, pathway_mask, device, tag, name, b_sz,
                           train_index, test_index, fold, unsupervised_loss_ratio, cell_loss_ratio,
                           epochs=50, learn_method='plus_nn', 
                           aggr_neighbor_func = 'mean', 
                           aggr_hidden_func = 'concat', 
                           unsupervised_loss_method = 'normal',
                           NUM_NEIGHBORS_LIST = [10, 10],
                           activation_function = 'Tanh',
                           nn_af = 'ReLU',
                           learning_rate = 0.001,
                           weight_decay = 0.0001,
                           optimier_method = 'Adam'
                           ):
    
    labels0 = getattr(dataCenter, 'labels')
    non_na_index = labels0 != -9
    
    best_vali_loss = 10
    
    n_pathways = pathway_mask.shape[1]
    n_genes = pathway_mask.shape[0]

    #### 1. object of graphSageMask  input_dim, hidden_dim, num_neighbors_list, aggr_neighbor_method="mean", aggr_hidden_method="sum"
    graphSageMask = GraphSageMask_nohop(pathway_mask=pathway_mask, emb_dim=n_pathways,
                                        num_neighbors_list=NUM_NEIGHBORS_LIST,
                                        aggr_neighbor_method=aggr_neighbor_func, 
                                        aggr_hidden_method=aggr_hidden_func,
                                        activation_function = activation_function
                                        ).to(device)
    
    #### 2. object of regression
    
    regression_1 = NN_1_FC(n_pathways).to(device)
    regression_2 = NN_2_LN(n_pathways, 1).to(device)
        
    #### 3. object of unsupervised_loss
    unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, 'adj_lists'), train_index, device)
    
    #### 4. object of cell_loss
    cell_loss = TargetLoss(getattr(dataCenter, 'adj_lists'), device)

    train_loss = []
    test_loss = []
    best_score = 10
    N_early_stop = 0
    
    lst=[np.ones(epochs), np.ones(epochs), np.ones(epochs),  np.ones(epochs), np.ones(epochs), np.ones(epochs), np.ones(epochs), np.ones(epochs), np.ones(epochs), np.ones(epochs), np.ones(epochs) ]
    arr = np.array(lst)
    final_res = pd.DataFrame(arr).T
    
    for epoch in range(epochs):
        print('Epoch', epoch, end='')
        
        graphSageMask.train()
        graphSageMask.net.train()
        graphSageMask.sagenet.train()
        regression_1.train()
        regression_2.train()
        
        graphSageMask, regression_1, regression_2, avg_loss, avg_loss_nn, avg_loss_net, avg_loss_cell = one_round_apply_DrugSage(dataCenter, graphSageMask, regression_1, regression_2, 
                                                                            unsupervised_loss, unsupervised_loss_ratio, unsupervised_loss_method, 
                                                                            cell_loss, cell_loss_ratio, 
                                                                            train_index, test_index,
                                                                            b_sz, device, learn_method, 
                                                                            learning_rate, weight_decay, optimier_method)
        
        graphSageMask.eval()
        graphSageMask.net.eval()
        graphSageMask.sagenet.eval()
        regression_1.eval()
        regression_2.eval()
        
        embeddings,batch_sampling_result = get_gnnmask_embeddings(graphSageMask, dataCenter, device) ###
        
        ###
        ndarray = embeddings.cpu().numpy()
        y_res = pd.DataFrame(ndarray)
        
        y_1 = regression_1(embeddings)
        y_predict = regression_2(y_1)
        
        y2 = y_predict.detach().numpy()
        y2 = y2.flatten()
        non_na_index = labels0 != -9
        y_res = pd.DataFrame({"Y":labels0[non_na_index],"Y_pred":y2[non_na_index]})
        all_pcc = y_res['Y'].corr(y_res['Y_pred'])
        
        y_res = pd.DataFrame({"Y":labels0,"Y_pred":y2})
        
        y2_train = y2[train_index]
        y1_train = labels0[train_index]
        non_na_index = y1_train != -9
        y2_train_9 = y2_train[non_na_index]
        y1_train_9 = y1_train[non_na_index]
        y_res = pd.DataFrame({"Y":y1_train_9,"Y_pred":y2_train_9})
        train_R2 = get_r2_numpy_manual(y_res['Y_pred'], y_res['Y'])
        train_MSE = mean_squared_error(y_res['Y_pred'], y_res['Y'])
        train_PCC = y_res['Y'].corr(y_res['Y_pred'])
        
        y2_test = y2[test_index]
        y1_test = labels0[test_index]
        non_na_index = y1_test != -9
        y2_test_9 = y2_test[non_na_index]
        y1_test_9 = y1_test[non_na_index]
        y_res = pd.DataFrame({"Y":y1_test_9,"Y_pred":y2_test_9})
        
        test_R2 = get_r2_numpy_manual(y_res['Y_pred'], y_res['Y'])
        test_PCC = y_res['Y'].corr(y_res['Y_pred'])
        test_MSE = mean_squared_error(y_res['Y_pred'], y_res['Y'])
        
        print(', PCC = {:.4f}'.format(all_pcc), end="")
        print(', train_MSE = {:.4f}'.format(train_MSE), end="")
        print(', test_R2 = {:.4f}'.format(test_R2), end="")
        print(', test_MSE = {:.4f}'.format(test_MSE))
        
        final_res.iloc[epoch:(epoch+1),0:1] = all_pcc
        final_res.iloc[epoch:(epoch+1),1:2] = train_PCC
        final_res.iloc[epoch:(epoch+1),2:3] = test_PCC
        final_res.iloc[epoch:(epoch+1),3:4] = train_MSE
        final_res.iloc[epoch:(epoch+1),4:5] = test_MSE
        final_res.iloc[epoch:(epoch+1),5:6] = train_R2
        final_res.iloc[epoch:(epoch+1),6:7] = test_R2
        
        final_res.iloc[epoch:(epoch+1),7:8] = avg_loss
        final_res.iloc[epoch:(epoch+1),8:9] = avg_loss_nn
        final_res.iloc[epoch:(epoch+1),9:10] = avg_loss_net
        final_res.iloc[epoch:(epoch+1),10:11] = avg_loss_cell
        
        new_score = test_MSE ###### 11232023
        
        if new_score < best_score:
            N_early_stop = 0
            best_score = new_score
            models = [graphSageMask, regression_1, regression_2]
            torch.save(models, 'models/{}/{}.fold_{}.best_model.torch'.format(tag, name, fold))
            print("best_model: "+str(epoch))
        else:
            N_early_stop = N_early_stop + 1
        
        if N_early_stop >= 20:
            N_early_stop = 0
            break
            
        train_loss.append(train_R2)
        test_loss.append(test_R2)
    final_res.to_csv('models/{}/{}.fold_{}.info.txt'.format(tag, name, fold), index=False)

    return graphSageMask, regression_1, regression_2, train_loss, test_loss

##############################################################################
#### functions for apply_model
##############################################################################
def one_round_apply_DrugSage(dataCenter, graphSageMask, regression_1, regression_2, 
                          unsupervised_loss, unsupervised_loss_ratio, unsupervised_loss_method, 
                          cell_loss, cell_loss_ratio, 
                          train_index, test_index, b_sz, device, learn_method, 
                          learning_rate, weight_decay, optimier_method):
    train_nodes = train_index
    test_nodes  = test_index
    labels0     = getattr(dataCenter, 'labels')
    labels      = torch.reshape(torch.from_numpy(labels0), (len(labels0), 1))
    NUM_NEIGHBORS_LIST = graphSageMask.num_neighbors_list
    adjacency_dict = getattr(dataCenter, 'adjacency_dict')
    features       = torch.FloatTensor(getattr(dataCenter, 'feats')).to(device)
    
    #######################################################
    positive_nodes = getattr(dataCenter, 'positive_nodes')
    new = []
    for i in positive_nodes:
        if i in train_nodes:
            new.append(i)
    positive_nodes = np.asarray(new, dtype=np.int)
    
    negative_nodes = getattr(dataCenter, 'negative_nodes')
    new = []
    for i in negative_nodes:
        if i in train_nodes:
            new.append(i)
    negative_nodes = np.asarray(new, dtype=np.int)
    #######################################################
    
    if unsupervised_loss_method == 'margin':
        num_neg = 6
    elif unsupervised_loss_method == 'normal':
        num_neg = 10
    else:
        print("unsupervised_loss_method can be only 'margin' or 'normal'.")
        sys.exit(1)
    
    models = [graphSageMask, regression_1, regression_2]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)
    
    
    if optimier_method == 'Adam':
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay = weight_decay)
    elif optimier_method == 'Adagrad':
        optimizer = torch.optim.Adagrad(params, lr=learning_rate, weight_decay = weight_decay)
    elif optimier_method == 'SGD':
        optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay = weight_decay, momentum=0.01)
    else:
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay = weight_decay)
    
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()
    
    criterion = nn.MSELoss().to(device)
    criterion_category = nn.CrossEntropyLoss().to(device)

    batches = math.ceil(len(train_nodes) / b_sz)
    avg_loss = 0
    avg_loss_nn = 0
    avg_loss_net = 0
    avg_loss_cell = 0
    for index in range(batches):
        ######### 将模型的参数梯度初始化为0
        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
        #######################################################
        
        ######### # 前向传播计算预测值,损失
        
        start_idx = index*b_sz
        end_idx   = (index+1)*b_sz
        if end_idx > len(train_nodes):
            end_idx = len(train_nodes)
        raw_batch_src_index = train_nodes[start_idx:end_idx]
        
        ########## prepare embs for unsupervised_loss ##########
        ext_batch_src_index = np.asarray(list(unsupervised_loss.extend_nodes(raw_batch_src_index, num_neg=num_neg)))
        
        batch_sampling_result = multihop_sampling(ext_batch_src_index, NUM_NEIGHBORS_LIST, adjacency_dict)
        batch_sampling_x = [features[idx].float().to(device) for idx in batch_sampling_result]
        ext_embs_batch = graphSageMask(batch_sampling_x)
        
        ############## prepare embs for cell_loss ##############
        batch_negative = random.choices(list(negative_nodes), k=20)
        batch_positive = random.choices(list(positive_nodes), k=20)
        
        if(len(batch_positive) > 0 and len(batch_negative) > 0):
            cell_batch_index = np.asarray(list(cell_loss.extend_nodes(batch_positive, batch_negative)))
            cell_batch_index_sampling = multihop_sampling(cell_batch_index, NUM_NEIGHBORS_LIST, adjacency_dict)
            cell_batch_x     = [features[idx].float().to(device) for idx in cell_batch_index_sampling]
            cell_batch_embs  = graphSageMask(cell_batch_x)
        #######################################################

        if learn_method == 'net_regression':
            ### 1. unsupervised loss
            if unsupervised_loss_method == 'margin':
                loss_net  = unsupervised_loss.get_loss_margin(ext_embs_batch, ext_batch_src_index)
                loss_cell = cell_loss.get_loss_margin(cell_batch_embs)
            elif unsupervised_loss_method == 'normal':
                loss_net  = unsupervised_loss.get_loss_sage(ext_embs_batch, ext_batch_src_index)
                loss_cell = cell_loss.get_loss_sage(cell_batch_embs)
            
            loss_net  = loss_net.to(device)
            loss_cell = loss_cell.to(device)

            ### 2. linear regression: use only cell lines with available IC50
            non_na_index = labels0[raw_batch_src_index] != -9
            nodes_batch  = raw_batch_src_index[non_na_index]
            labels_batch = labels[nodes_batch]
            if len(nodes_batch) < 2:
                continue
            
            batch_sampling_result = multihop_sampling(nodes_batch, NUM_NEIGHBORS_LIST, adjacency_dict)
            batch_sampling_x = [features[idx].float().to(device) for idx in batch_sampling_result]
            embs_batch = graphSageMask(batch_sampling_x)
            y1 = regression_1(embs_batch)
            y_predict = regression_2(y1).to(device)
            loss_nn = criterion(y_predict,labels_batch)
            
            loss = loss_nn + unsupervised_loss_ratio * loss_net + cell_loss_ratio * loss_cell
        
        avg_loss += loss.item()/batches
        avg_loss_nn += loss_nn.item()/batches
        avg_loss_net += loss_net.item()/batches
        avg_loss_cell += loss_cell.item()/batches
        
        #######################################################
        
        ############ # 反向传播计算梯度
        loss.backward()
        
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 6)
        
        ########### # 更新所有参数
        optimizer.step()
        ############################## End of one batch ##############################
        
    print(', avg_loss = {:.4f}, avg_loss_nn = {:.4f}, avg_loss_net = {:.4f}, avg_loss_cell = {:.4f}'.format(avg_loss, avg_loss_nn, avg_loss_net, avg_loss_cell), end='')
    return graphSageMask, regression_1, regression_2, avg_loss, avg_loss_nn, avg_loss_net, avg_loss_cell



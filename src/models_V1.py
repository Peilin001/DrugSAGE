import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.customized_linear import CustomizedLinear


class Classification(nn.Module):
	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		self.layer = nn.Sequential(nn.Linear(emb_size, num_classes))
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists

class BinaryClassification(nn.Module):
    def __init__(self, emb_size):
        super(BinaryClassification, self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(emb_size, 1),
            nn.Sigmoid()
        )
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        output = self.layer(embeds)
        return output.squeeze(1)  

    def binary_loss(self, predictions, targets):
        loss_func = torch.nn.BCELoss()
        loss = loss_func(predictions, targets)
        return loss
		
class NN(torch.nn.Module):
    def __init__(self,emb_size,output_n):
        super(NN, self).__init__()

        self.predict = torch.nn.Sequential(
                                           torch.nn.Dropout(0.2), 
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(emb_size,emb_size), ###
                                           torch.nn.Linear(emb_size,output_n)
                                           )
        
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            
    def forward(self,x):
        y = self.predict(x)
        return y

############### 02052025
class NN_1_FC(torch.nn.Module):
    def __init__(self,emb_size):
        super(NN_1_FC, self).__init__()

        self.predict = torch.nn.Sequential(
                                           torch.nn.Dropout(0.2), 
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(emb_size,emb_size)
                                           )
        
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            
    def forward(self,x):
        y = self.predict(x)
        return y

class NN_2_LN(torch.nn.Module):
    def __init__(self,emb_size,output_n):
        super(NN_2_LN, self).__init__()

        self.predict = torch.nn.Sequential(
                                           torch.nn.Linear(emb_size,output_n)
                                           )
        
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            
    def forward(self,x):
        y = self.predict(x)
        return y

##############################################################

class NN_Sigmoid(torch.nn.Module):
    def __init__(self,emb_size,output_n):
        super(NN, self).__init__()

        self.predict = torch.nn.Sequential(torch.nn.Dropout(0.2), 
                                           torch.nn.Sigmoid(),
                                           torch.nn.Linear(emb_size,output_n)
                                           )
        
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            
    def forward(self,x):
        y = self.predict(x)
        return y

class NN_linear(torch.nn.Module):
    def __init__(self,emb_size,output_n):
        super(NN, self).__init__()

        self.predict = torch.nn.Sequential(torch.nn.Dropout(0.2), 
                                           torch.nn.Linear(emb_size,output_n)
                                           )
        
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            
    def forward(self,x):
        y = self.predict(x)
        return y


class UnsupervisedLoss(object):
	"""docstring for UnsupervisedLoss"""
	def __init__(self, adj_lists, train_nodes, device):
		super(UnsupervisedLoss, self).__init__()
		self.Q = 10
		self.N_WALKS = 6
		self.WALK_LEN = 1
		self.N_WALK_LEN = 6
		self.MARGIN = 1  ####
		self.adj_lists = adj_lists
		self.train_nodes = train_nodes
		self.device = device

		self.target_nodes = None
		self.positive_pairs = []
		self.negtive_pairs = []
		self.node_positive_pairs = {}
		self.node_negtive_pairs = {}
		self.unique_nodes_batch = []

	def get_loss_sage(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			#print(len(pps), len(nps))
			if len(pps) == 0 or len(nps) == 0:
				continue

			# Q * Exception(negative score)
			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
			#print('neg_score', neg_score)

			# multiple positive score
			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score = torch.log(torch.sigmoid(pos_score))
			#print('pos_score', pos_score)

			nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
				
		loss = torch.mean(torch.cat(nodes_score, 0)).to(self.device)
		
		return loss

	def get_loss_margin(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			#print('node: '+str(node))
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			#print(pps, nps)
			if len(pps) == 0 or len(nps) == 0:
				continue

			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			#print(indexs)
			#print(node_indexs, neighb_indexs)
			
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			#print(len(embeddings[node_indexs[0]]))
			#print(pos_score)
			pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)
			#print(pos_score)

			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			#print(neg_score)
			neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)
			#print(neg_score)
			nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score-pos_score+self.MARGIN).view(1,-1))
			
			### only consider pos_score: ###########################################################
			#nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score+self.MARGIN).view(1,-1))
			
			#print(nodes_score)
		loss = torch.mean(torch.cat(nodes_score, 0),0)

		return loss


	def extend_nodes(self, nodes, num_neg=6):
		self.positive_pairs = []
		self.node_positive_pairs = {}
		self.negtive_pairs = []
		self.node_negtive_pairs = {}

		self.target_nodes = nodes
		self.get_positive_nodes(nodes)
		self.get_negtive_nodes(nodes, num_neg)
		self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
		#assert set(self.target_nodes) < set(self.unique_nodes_batch)
		if set(self.target_nodes) >= set(self.unique_nodes_batch):
			print("Error! target_nodes: ", self.target_nodes)
			print("Error! positive_pairs: ", self.positive_pairs)
			print("Error! negtive_pairs: ", self.negtive_pairs)
		return self.unique_nodes_batch

	def get_positive_nodes(self, nodes):
		return self._run_random_walks(nodes)

	def get_negtive_nodes(self, nodes, num_neg):
		for node in nodes:
			neighbors = set([node])
			frontier  = set([node])
			for i in range(self.N_WALK_LEN):
				current = set()
				for outer in frontier:
					current |= self.adj_lists[int(outer)]
				frontier = current - neighbors
				neighbors |= current
				#print(i, len(neighbors))
			far_nodes = set(self.train_nodes) - neighbors
			#print(len(far_nodes))
			neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
			self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
			self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
		return self.negtive_pairs

	def _run_random_walks(self, nodes):
		for node in nodes:
			if len(self.adj_lists[int(node)]) == 0:
				continue
			cur_pairs = []
			selected_nodes = set()
			for i in range(self.N_WALKS):
				curr_node = node
				for j in range(self.WALK_LEN):
					neighs = self.adj_lists[int(curr_node)]
					next_node = random.choice(list(neighs))
					if next_node != node and next_node in self.train_nodes and next_node not in selected_nodes:
						selected_nodes.add(next_node)
						self.positive_pairs.append((node,next_node))
						cur_pairs.append((node,next_node))
					curr_node = next_node
			self.node_positive_pairs[node] = cur_pairs
		return self.positive_pairs
		


class TargetLoss(object):
	
	def __init__(self, adj_lists, device):
		super(TargetLoss, self).__init__()
		self.Q = 10
		self.MARGIN = 1
		self.adj_lists = adj_lists
		self.device = device

		self.target_nodes = None
		self.positive_pairs = []
		self.negtive_pairs = []
		self.node_positive_pairs = {}
		self.node_negtive_pairs = {}
		self.unique_nodes_batch = []

	def get_loss_sage(self, embeddings):
		assert len(embeddings) == len(self.unique_nodes_batch)
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			# Q * Exception(negative score)
			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
			#print('neg_score', neg_score)

			# multiple positive score
			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score = torch.log(torch.sigmoid(pos_score))
			#print('pos_score', pos_score)

			nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
				
		loss = torch.mean(torch.cat(nodes_score, 0)).to(self.device)
		return loss

	def get_loss_margin(self, embeddings):
		assert len(embeddings) == len(self.unique_nodes_batch)
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			#print('pos_score', pos_score, torch.log(torch.sigmoid(pos_score)))
			pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)
			#print('pos_score after', pos_score)
			
			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			#print('neg_score', neg_score, torch.log(torch.sigmoid(neg_score)))
			neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)
			#print('neg_score after', neg_score)
			
			nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score-pos_score+self.MARGIN).view(1,-1))
			
		loss = torch.mean(torch.cat(nodes_score, 0),0)
		return loss


	def extend_nodes(self, positive_nodes, negative_nodes):
		self.positive_pairs = []
		self.node_positive_pairs = {}
		self.negtive_pairs = []
		self.node_negtive_pairs = {}

		self.unique_nodes_batch = np.hstack((positive_nodes,negative_nodes))

		for node in positive_nodes:
			pos_node = random.choice(list(positive_nodes))
			if pos_node == node:
				continue
			self.positive_pairs.extend([(node, pos_node)])
			self.node_positive_pairs[node] = [(node, pos_node)]

			neg_node = random.choice(list(negative_nodes))
			self.negtive_pairs.extend([(node, neg_node)])
			self.node_negtive_pairs[node] = [(node, neg_node)]

		return self.unique_nodes_batch



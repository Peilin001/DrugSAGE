import sys
import os

from collections import defaultdict
import numpy as np


class DataCenterL2(object):
	"""docstring for DataCenter"""
	def __init__(self):
		super(DataCenterL2, self).__init__()
		
	def load_dataSet(self, ccle_feate_file, ccle_label_file, ccle_label_file_2, ccle_inter_file, task='regression', drug=1):
		
		self.ccle_inter_file = ccle_inter_file
		self.ccle_feate_file = ccle_feate_file
		self.ccle_label_file = ccle_label_file
		self.ccle_label_file_2 = ccle_label_file_2
		#print('ccle_inter_file: '+ccle_inter_file)
		#print('ccle_feate_file: '+ccle_feate_file)
		#print('ccle_label_file: '+ccle_label_file)
		#print('ccle_label_file_2: '+ccle_label_file_2)
		

		feat_data = []
		labels = []
		labels_binary = []
		positive_nodes = []
		negative_nodes = []
		node_map = {}
		label_map = {}

		check_nodes = {}
		with open(ccle_inter_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				if info[0] in check_nodes:
					check_nodes[info[0]] += 1
				else:
					check_nodes[info[0]] = 0
				if info[1] in check_nodes:
					check_nodes[info[1]] += 1
				else:
					check_nodes[info[1]] = 0
		effect_nodes = {}
		for i, value in enumerate(check_nodes):
			if check_nodes[value] < 1000:
				effect_nodes[value]  = check_nodes[value]
		#print('effect_nodes:', len(effect_nodes))

		count = 0
		with open(ccle_feate_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				if not info[0] in effect_nodes:
					continue

				if not i == 0:
					feat_data.append([float(x) for x in info[1:]])
					node_map[info[0]] = count
					count += 1
		
		if(task == 'classification'):
			with open(ccle_label_file) as fp:
				for i, line in enumerate(fp):
					info = line.strip().split()
					if not info[1] in label_map:
						label_map[info[1]] = len(label_map)
					labels.append(label_map[info[1]])
					if info[1] == '-9':
						labels_binary.append(0)
					else:
						labels_binary.append(1)
			labels = np.asarray(labels, dtype=np.int64)
		else:
			with open(ccle_label_file) as fp:
				for i, line in enumerate(fp):
					info = line.strip().split()

					if not info[0] in effect_nodes:
						continue

					#print('drug='+drug)
					labels.append(info[drug])
					#print(info[drug])
			labels = np.asarray(labels, dtype=np.float32)

		feat_data = np.asarray(feat_data)
		adj_lists = defaultdict(set)
		
		with open(ccle_inter_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				#assert len(info) == 2
				x = info[0] in effect_nodes and info[1] in effect_nodes
				if not x:
					continue
				paper1 = node_map[info[0]]
				paper2 = node_map[info[1]]
				adj_lists[paper1].add(paper2)
				adj_lists[paper2].add(paper1)

		#print(adj_lists[59])
		with open(ccle_label_file_2) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				if not info[0] in effect_nodes:
					continue
				if info[1] == 'positive':
					positive_nodes.append(node_map[info[0]])
				if info[1] == 'negative':
					negative_nodes.append(node_map[info[0]])
				
		positive_nodes = np.asarray(positive_nodes, dtype=np.int)
		negative_nodes = np.asarray(negative_nodes, dtype=np.int)

		adjacency_dict = defaultdict(list)
		for index in range(len(adj_lists)):
			a = adj_lists[index]
			a_list = list(a)
			adjacency_dict[index] = a_list


		#print('feat_data:      ', len(feat_data), type(feat_data))
		#print('labels:         ', len(labels), type(labels))
		#print('adj_lists:      ', len(adj_lists), type(adj_lists))
		#print('adjacency_dict: ', len(adjacency_dict))
		#print('node_map:       ', len(node_map))
		#print('positive_nodes: ', len(positive_nodes))
		#print('negative_nodes: ', len(negative_nodes))

		setattr(self, 'node_map', node_map)
		setattr(self, 'num_nodes', feat_data.shape[0])
		setattr(self, 'num_features', feat_data.shape[1])

		setattr(self, 'feats', feat_data)
		setattr(self, 'labels', labels)
		setattr(self, 'labels_binary', labels_binary)
		setattr(self, 'negative_nodes', negative_nodes)
		setattr(self, 'positive_nodes', positive_nodes)
		setattr(self, 'adj_lists', adj_lists)
		setattr(self, 'adjacency_dict', adjacency_dict)


class DataCenterL2_set(object):
	"""docstring for DataCenter"""
	def __init__(self):
		super(DataCenterL2_set, self).__init__()
		
	def load_dataSet(self, ccle_feate_file, ccle_label_file, ccle_label_file_2, ccle_inter_file, task='regression', drug=1):
		
		self.ccle_inter_file = ccle_inter_file
		self.ccle_feate_file = ccle_feate_file
		self.ccle_label_file = ccle_label_file
		self.ccle_label_file_2 = ccle_label_file_2
		
		feat_data = []
		labels = []
		labels_binary = []
		positive_nodes = []
		negative_nodes = []
		node_map = {}
		label_map = {}

		check_nodes = {}
		with open(ccle_inter_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				if info[0] in check_nodes:
					check_nodes[info[0]] += 1
				else:
					check_nodes[info[0]] = 0
				if info[1] in check_nodes:
					check_nodes[info[1]] += 1
				else:
					check_nodes[info[1]] = 0
		effect_nodes = {}
		for i, value in enumerate(check_nodes):
			if check_nodes[value] < 1000:
				effect_nodes[value]  = check_nodes[value]
		
		count = 0
		with open(ccle_feate_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				if not info[0] in effect_nodes:
					continue

				if not i == 0:
					feat_data.append([float(x) for x in info[1:]])
					node_map[info[0]] = count
					count += 1
		
		if(task == 'classification'):
			with open(ccle_label_file) as fp:
				for i, line in enumerate(fp):
					info = line.strip().split()
					if not info[1] in label_map:
						label_map[info[1]] = len(label_map)
					labels.append(label_map[info[1]])
					if info[1] == '-9':
						labels_binary.append(0)
					else:
						labels_binary.append(1)
			labels = np.asarray(labels, dtype=np.int64)
		else:
			with open(ccle_label_file) as fp:
				for i, line in enumerate(fp):
					info = line.strip().split()

					if not info[0] in effect_nodes:
						continue

					labels.append(info[drug])
			labels = np.asarray(labels, dtype=np.float32)

		feat_data = np.asarray(feat_data)
		adj_lists = defaultdict(set)
		
		with open(ccle_inter_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				x = info[0] in effect_nodes and info[1] in effect_nodes
				if not x:
					continue
				paper1 = node_map[info[0]]
				paper2 = node_map[info[1]]
				adj_lists[paper1].add(paper2)
				adj_lists[paper2].add(paper1)

		with open(ccle_label_file_2) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				if not info[0] in effect_nodes:
					continue
				if info[1] == 'positive':
					positive_nodes.append(node_map[info[0]])
				if info[1] == 'negative':
					negative_nodes.append(node_map[info[0]])
				
		positive_nodes = np.asarray(positive_nodes, dtype=np.int)
		negative_nodes = np.asarray(negative_nodes, dtype=np.int)

		adjacency_dict = defaultdict(list)
		for index in range(len(adj_lists)):
			a = adj_lists[index]
			a_list = list(a)
			adjacency_dict[index] = a_list

		setattr(self, 'node_map', node_map)
		setattr(self, 'num_nodes', feat_data.shape[0])
		setattr(self, 'num_features', feat_data.shape[1])
		setattr(self, 'feats', feat_data)
		setattr(self, 'labels', labels)
		setattr(self, 'labels_binary', labels_binary)
		setattr(self, 'negative_nodes', negative_nodes)
		setattr(self, 'positive_nodes', positive_nodes)
		setattr(self, 'adj_lists', adj_lists)
		setattr(self, 'adjacency_dict', adjacency_dict)
	def set_node_feat_zero(self, new_features):
		feat_data = new_features
		setattr(self, 'feats', feat_data)


class DataCenterL1(object):
	"""docstring for DataCenter"""
	def __init__(self):
		super(DataCenterL1, self).__init__()
		
	def load_dataSet(self, ccle_feate_file, ccle_label_file, ccle_inter_file, task='classification', drug=1):
		self.ccle_inter_file = ccle_inter_file
		self.ccle_feate_file = ccle_feate_file
		self.ccle_label_file = ccle_label_file
		

		feat_data = []
		labels = []
		labels_binary = []
		positive_nodes = []
		negative_nodes = []
		node_map = {}
		label_map = {}

		check_nodes = {}
		with open(ccle_inter_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				if info[0] in check_nodes:
					check_nodes[info[0]] += 1
				else:
					check_nodes[info[0]] = 0
				if info[1] in check_nodes:
					check_nodes[info[1]] += 1
				else:
					check_nodes[info[1]] = 0
		effect_nodes = {}
		for i, value in enumerate(check_nodes):
			if check_nodes[value] < 1000:
				effect_nodes[value]  = check_nodes[value]
		#print('effect_nodes:', len(effect_nodes))

		count = 0
		with open(ccle_feate_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				#print(info[0])
				if not info[0] in effect_nodes:
					continue

				if not i == 0:
					feat_data.append([float(x) for x in info[1:]])
					node_map[info[0]] = count
					#print(info[0], count)
					count += 1

		
		if(task == 'classification'):
			with open(ccle_label_file) as fp:
				for i, line in enumerate(fp):
					info = line.strip().split()
					if not info[1] in label_map:
						label_map[info[1]] = len(label_map)
					labels.append(label_map[info[1]])
					if info[1] == '-9':
						labels_binary.append(-9)
					else:
						labels_binary.append( label_map[info[1]] )
			labels = np.asarray(labels, dtype=np.int64)
		else:
			with open(ccle_label_file) as fp:
				for i, line in enumerate(fp):
					info = line.strip().split()

					if not info[0] in effect_nodes:
						continue

					labels.append(info[drug])
					
			labels = np.asarray(labels, dtype=np.float32)

		feat_data = np.asarray(feat_data)
		adj_lists = defaultdict(set)
		
		with open(ccle_inter_file) as fp:
			for i, line in enumerate(fp):
				info = line.strip().split()
				#assert len(info) == 2
				x = info[0] in effect_nodes and info[1] in effect_nodes
				if not x:
					continue
				paper1 = node_map[info[0]]
				paper2 = node_map[info[1]]
				adj_lists[paper1].add(paper2)
				adj_lists[paper2].add(paper1)
		
		adjacency_dict = defaultdict(list)
		for index in range(len(adj_lists)):
			a = adj_lists[index]
			a_list = list(a)
			adjacency_dict[index] = a_list
		
		setattr(self, 'node_map', node_map)
		setattr(self, 'num_nodes', feat_data.shape[0])
		setattr(self, 'num_features', feat_data.shape[1])

		setattr(self, 'feats', feat_data)
		setattr(self, 'labels', labels)
		setattr(self, 'labels_binary', labels_binary)
		setattr(self, 'adj_lists', adj_lists)
		setattr(self, 'adjacency_dict', adjacency_dict)



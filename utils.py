#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '7/12/2020'
"""
import json
import random
import dgl
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
import  torch.utils.data as Data
from dgl.data import TUDataset,MiniGCDataset
from sklearn.model_selection import KFold
from torch import nn


def shuffle(data:list,labels:list,seed):
	"""
	shuffle the data to balance the distribution of samples in different class
	:param data: the data need to shuffle
	:param labels: the labels need to shuffle
	:param seed: seed used in generator of random number
	:return: shuffled data
	"""
	zipped = list(zip(data,labels))
	random.seed(seed)
	random.shuffle(zipped)
	new_zipped = list(map(list, zip(*zipped)))
	new_data,new_labels = new_zipped[0], new_zipped[1]
	return new_data, torch.tensor(new_labels,dtype=torch.long)

def cross_vavildation(graphs,n_split):
	"""
	split the whole graphs set into train data and test data proportionally
	:param graphs: the graphs needed to split
	:param labels: the labels corresponding to graphs
	:param split_ratio: the ratio of train data to total
	:return:
	"""
	
	k_fold = KFold(n_splits=n_split)
	data_fold = k_fold.split(graphs)
	return data_fold

def encode_labels(graph_labels):
	"""
	encode the labels of graphs start at index 0
	:param graph_labels: the labels of graphs required encoding
	:return: new_labels
	"""
	label_set = set(graph_labels[:,0].tolist())
	for index, label in enumerate(label_set):
		graph_labels[graph_labels[:,0]==label] = index
	return graph_labels

def add_selfloop(graphs):
	"""
	add self-loop to nodes in each graph
	:param graphs: graphs without self-loop
	"""
	for graph in graphs:
		graph.add_edges(graph.nodes(),graph.nodes())
		# dgl.transform.add_self_loop(graph)
	return graphs

class GraphPreprocessor(nn.Module):
	""" Graph Preprocessing Class """
	def __init__(self,batch_graph, n_degrees):
		super(GraphPreprocessor, self).__init__()
		self.bg = batch_graph
		self.n_degrees = n_degrees
		self.attr_list = self.generate_encoder()
	
	def generate_encoder(self):
		"""
			encode node to its representation in vector space according node label
			:param bg: graph containing a set of nodes
			:return: new node representation
			"""
		bg = self.bg
		attr_list = {}
		if 'node_attr' in bg.ndata.keys():
			if bg.ndata['node_attr'].size(1)==1:
				bg.ndata['node_attr'] = bg.ndata['node_attr'].type(dtype=torch.long)
				max_node_attr = int(bg.ndata['node_attr'].max()) + 1
				nattr_to_onehot = torch.eye(max_node_attr)
				attr_list['node_attr'] = nattr_to_onehot
			else:
				bg.ndata['node_attr'] = bg.ndata['node_attr'].type(dtype=torch.float)
				bg.ndata['node_attr'] = normalize_features(bg.ndata['node_attr'])
				# max_bg_nattr = bg.ndata['node_attr'].max(dim=1,keepdim=True)[0].expand_as(bg.ndata['node_attr'])
				# bg.ndata['node_attr'] = torch.div(bg.ndata['node_attr'],max_bg_nattr)
				attr_list['node_attr'] = None
		if 'node_labels' in bg.ndata.keys():
			bg.ndata['node_labels'] = bg.ndata['node_labels'].type(dtype=torch.long)
			max_node_labels = int(bg.ndata['node_labels'].max()) + 1
			nlabel_to_onehot = torch.eye(max_node_labels)
			attr_list['node_labels'] = nlabel_to_onehot
		if 'node_degrees1' in bg.ndata.keys():
			bg.ndata['node_degrees1']=bg.in_degrees().reshape(-1,1).type(dtype=torch.long)
			min_node_degrees = int(min([bg.ndata['node_degrees'+str(idx)].min() for idx in range(1,self.n_degrees+1)]))
			for idx in range(1, self.n_degrees + 1):
				bg.ndata['node_degrees'+str(idx)] = torch.sub(bg.ndata['node_degrees'+str(idx)], min_node_degrees)
			max_node_degrees = int(max([bg.ndata['node_degrees'+str(idx)].max() for idx in range(1,self.n_degrees+1)]))+1
			ndegree_to_onehot = torch.eye(max_node_degrees)
			attr_list['node_degrees1'] = ndegree_to_onehot
		if 'edge_labels' in bg.edata.keys():
			bg.edata['edge_labels'] = bg.edata['edge_labels'].type(dtype=torch.long)
			max_edge_labels = int(bg.edata['edge_labels'].max()) + 1
			elabel_to_onehot = torch.eye(max_edge_labels)
			attr_list['edge_labels'] = elabel_to_onehot
		return attr_list

	def node_udf(self,nodes):
		""" node-level user defined function """
		message_list = []
		for attr, encoder in self.attr_list.items():
			if attr == 'node_attr':
				if not encoder is None:
					message_list.append(encoder[nodes.data['node_attr']].squeeze())
				else:
					message_list.append(self.bg.ndata['node_attr'])
			elif attr == 'node_labels':
				message_list.append(encoder[nodes.data['node_labels']].squeeze())
			elif attr == 'node_degrees1':
				for idx in range(1,self.n_degrees+1):
					message_list.append(encoder[nodes.data['node_degrees'+str(idx)]].squeeze())
			else:
				pass
		new_message = torch.cat(message_list,dim=1)
		return {'n_feat':new_message}
	
	def message_func(self,edges):
		""" message passing function """
		message = 0
		edge_features = 0
		if 'edge_labels' in self.attr_list.keys():
			encoder = self.attr_list['edge_labels']
			message = encoder[edges.data['edge_labels']].squeeze()
			max_elabel = float(self.bg.edata['edge_labels'].max())
			self.bg.edata['edge_labels'] += 1
			if 'node_labels' in self.attr_list.keys():
				self.bg.ndata['node_labels'] += 1
				max_nlabel = float(self.bg.ndata['node_labels'].max())
				edge_features = torch.stack([edges.src['node_labels']/max_nlabel,edges.dst['node_labels']/max_nlabel,edges.data['edge_labels']/max_elabel],dim=1).type(dtype=torch.float).squeeze()
			else:
				max_deg_ord = 'node_degrees' + str(self.n_degrees)
				max_ndegrees = float(self.bg.ndata[max_deg_ord].max())
				self.bg.ndata[max_deg_ord] += 1
				edge_features = torch.stack(
					[edges.src[max_deg_ord]/max_ndegrees, edges.dst[max_deg_ord]/max_ndegrees,edges.data['edge_labels']/max_elabel],
					dim=1).type(dtype=torch.float).squeeze()
		elif 'node_labels' in self.attr_list.keys():
			encoder = self.attr_list['node_labels']
			message = torch.add(encoder[edges.src['node_labels']], encoder[edges.dst['node_labels']]).squeeze()
			message = torch.mul(message,0.5)
			self.bg.ndata['node_labels'] += 1
			edge_features = torch.stack([edges.src['node_labels'],edges.dst['node_labels'],edges.src['node_labels']+edges.dst['node_labels']],dim=1).type(dtype=torch.float).squeeze()
		elif 'node_degrees1' in self.attr_list.keys():
			encoder = self.attr_list['node_degrees1']
			temp_msg = []
			for idx in range(1,self.n_degrees+1):
				temp_msg.append(torch.add(encoder[edges.src['node_degrees'+str(idx)]], encoder[edges.dst['node_degrees'+str(idx)]]).squeeze())
			message = torch.stack(temp_msg,dim=1)
			torch.mul(message, 0.5)
			max_deg_ord = 'node_degrees'+str(self.n_degrees)
			self.bg.ndata[max_deg_ord] += 1.
			edge_features = torch.stack([edges.src[max_deg_ord],edges.dst[max_deg_ord],edges.src[max_deg_ord]+edges.dst[max_deg_ord]],dim=1).type(dtype=torch.float).squeeze()
		else:
			pass
		edges.data['e_feat'] = normalize_features(edge_features)
		return {'edge_labels': message}

	def reduced_func(self, nodes):
		""" reduced user defined function """
		if 'edge_labels' in nodes.mailbox.keys():
			edge_embedding = torch.mean(nodes.mailbox['edge_labels'],dim=1) # [N,edge_embedding_size]
			node_features = torch.cat([nodes.data['n_feat'],edge_embedding],dim=1)
			return {'n_feat':node_features}
		else:
			return None

	def forward(self):
		""" preprocessing """
		self.bg.apply_nodes(self.node_udf)
		self.bg.update_all(self.message_func, self.reduced_func)
		for attr, _ in self.attr_list.items():
			if attr != 'edge_labels':
				self.bg.ndata.pop(attr)
			else:
				self.bg.edata.pop(attr)
		self.bg.edata.pop('_ID')
		self.bg.ndata.pop('_ID')
		for idx in range(2,self.n_degrees+1):
			self.bg.ndata.pop('node_degrees'+str(idx))
		return self.bg
	
def add_highdegrees(graphs,n_degrees=3):
	""" add the node degree of high oder  to the attribute of nodes in each graph """
	cluster_matrices = {}
	for index,graph in enumerate(graphs):
		sp_adj = graph.adjacency_matrix().type(dtype=torch.long)
		ones_ = torch.ones(1, dtype=torch.long)
		bg_adj = sp_adj.to_dense()
		for idx in range(1,n_degrees+1):
			bg_adj = torch.spmm(sp_adj, bg_adj) if idx>1 else bg_adj
			norm_bg_adj = torch.where(bg_adj > 0, ones_, bg_adj) if idx>1 else bg_adj
			graph.ndata['node_degrees'+str(idx)] = norm_bg_adj.sum(dim=1)

def normalize_features(mx):
	""" Row-normalize tensor """
	row_sum = torch.sum(mx,dim=1)
	r_inv = torch.pow(row_sum, -1).flatten()
	r_inv[torch.isinf(r_inv)] = 0.
	r_inv = r_inv.reshape(-1,1).expand_as(mx)
	mx = torch.mul(r_inv,mx)
	return mx

def load_data(data_path, raw_data_path, dataset_name, seed,n_degrees=3):
	"""
	load the dataset specified by dataset_name
	:param data_path: the path of processed dataset
	:param raw_data_path: the path of raw dataset
	:param dataset_name: name of the dataset
	:param seed: the seed of generator of random number
	:return: graphs and labels
	"""
	dataset = TUDataset(dataset_name)
	graphs = dataset.graph_lists
	add_selfloop(graphs) # add self-loop
	add_highdegrees(graphs,n_degrees=n_degrees)
	labels = encode_labels(dataset.graph_labels)
	labels = labels.squeeze()
	batch_graph = dgl.batch(graphs)
	gp = GraphPreprocessor(batch_graph,n_degrees=n_degrees)
	bg = gp() # graph preprocessing
	bg.ndata['n_feat'] = normalize_features(bg.ndata['n_feat'])
	new_graphs = dgl.unbatch(batch_graph)
	new_graphs, labels = shuffle(data=new_graphs,labels=labels,seed=seed) # shuffle data and labels randomly
	return batch_graph, new_graphs, labels

class GraphDataset(object):
	""" graph dataset """
	def __init__(self,graphs,labels,name):
		self.num_graphs = len(graphs)
		self.name = name
		self.graphs = graphs
		self.labels = labels
	
	def __len__(self):
		"""Return the number of graphs in the dataset."""
		return self.num_graphs
	
	def __getitem__(self, idx):
		"""Get the i^th sample.

				Paramters
				---------
				idx : int
					The sample index.

				Returns
				-------
				(dgl.DGLGraph, int)
					The graph and its label.
				"""
		return self.graphs[idx], self.labels[idx]
	
	@property
	def num_classes(self):
		"""Number of classes."""
		return max(self.labels)

def collate(samples):
	""" collate data and label then batch both them
	:param samples: a list of pairs (graph, label)
	"""
	graphs, labels = map(list, zip(*samples))
	batched_graph = dgl.batch(graphs)
	return batched_graph, torch.tensor(labels)

def save_parameters(parameters,name,file='experiment_settings.json'):
	"""
	save the parameters or settings of model to file, default file: experiment_settings.json
	:param parameters: the parameters or settings required to save
	:param name: the ID of parameters used to index
	:param file: the file name in which parameters will be writen
	:return: None
	"""
	parameters_dicts = vars(parameters)
	old_dicts = json.load(open(file))
	old_dicts[name] = parameters_dicts
	json.dump(old_dicts,open(file,'w'))

def evaluate(model,data,labels,mode='eval'):
	"""
	evaluate performance of the model
	:param data: data for evaluation
	:param labels: labels corresponding to data
	:param mode: the evaluation mode included train and test
	:return: accuracy
	"""
	correct_p, correct_n = 0, 0
	total_p, total_n = labels[labels==1].size(0), labels[labels==0].size(0)
	if mode == 'eval':
		model.eval()
	else:
		model.train()
	with torch.no_grad():
		output = model(data)
		_, indices = torch.max(output, dim=1)
		compare_ = indices == labels
		correct = torch.sum(compare_)
		correct_p = sum(labels[compare_==True]==1)
		correct_n = sum(labels[compare_==True]==0)
		# print('total_p: {:d} correct_p: {:d} total_n {:d} correct_n: {:d}'.format(total_p,correct_p,total_n,correct_n))
		return correct.item() * 1.0 / len(labels),[total_p,total_n,correct_p,correct_n]

if __name__ == "__main__":
	data_path = "../datasets/"
	raw_data_path = "../.dgl/"
	dataset_name = 'PROTEINS'
	batch_graph, new_graphs, labels = load_data(data_path,raw_data_path,dataset_name,seed=42)
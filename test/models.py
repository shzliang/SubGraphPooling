#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '2020/6/23'
"""
import math
from torch import nn
import torch as pt
from torch.nn import Parameter
import torch.nn.functional as F
from test.layers import GraphConvolution

class GCN(nn.Module):
	r"""
	A base model for preprocessing data
	"""
	def __init__(self, n_features, n_hidden, n_class, dropout_ratio):
		super(GCN, self).__init__()
		self.gc1 = GraphConvolution(n_features, n_hidden)
		self.gc2 = GraphConvolution(n_hidden, n_class)
		self.dropout_ratio = dropout_ratio

	def forward(self, adj,features=None):
		r"""
		:param adj: adjacent matrix of graph
		:param features: graph feature
		:return: the probability distribution of graph
		"""
		if features is None:
			features = pt.eye(adj.size(1)).reshape(1,adj.size(1),-1).expand(size=(adj.size(0),adj.size(1),-1))
			features = features.cuda()
		x = F.relu(self.gc1(features, adj))
		x = F.dropout(x, self.dropout_ratio, training=self.training)
		x = F.relu(self.gc2(x,adj))
		out = x.mean(dim=1)
		return out

class SVM(nn.Module):
	def __init__(self, in_features, out_features):
		super(SVM, self).__init__()
		self.SVM_w = Parameter(pt.FloatTensor(out_features, in_features))
		self.SVM_b = Parameter(pt.FloatTensor(out_features))
		self.reset_parameters()
	
	def reset_parameters(self):
		"""
		Initialize all parameter with random uniform distribution
		:return: None
		"""
		std_v = 1. / math.sqrt(self.SVM_w.size(0))
		self.SVM_w.data.uniform_(-std_v, std_v)
		self.SVM_b.data.uniform_(-std_v, std_v)
	
	def forward(self, x):
		r"""
		:param x: a graph embedding
		:return: the probability of graph
		"""
		out = F.linear(x, self.SVM_w, self.SVM_b)
		return out

class GCTL(nn.Module):
	r"""
	A Graph Convolution based Transfer Learning for Graph Classification
	"""
	def __init__(self, in_features, n_hidden, n_embedding, out_features, dropout_ratio):
		super(GCTL,self).__init__()
		self.GCN = GCN(in_features, n_hidden, n_embedding, dropout_ratio)
		self.SVM = SVM(n_embedding, out_features)

	def forward(self, adj, features=None):
		r"""
		:param adj: adjacent matrix of graph
		:param features: set of node features
		:return: the label of input graph
		"""
		if features is None:
			features = pt.eye(adj.size(1)).reshape(1,adj.size(1),-1).expand(size=(adj.size(0),adj.size(1),-1))
			features = features.cuda()
		graph_embeddings = self.GCN(features, adj)
		new_embeddings = graph_embeddings.mean(dim=1)
		out = self.SVM(new_embeddings)
		return out
	
if __name__ == "__main__":
	adj = pt.FloatTensor(2,4,4)
	features = pt.FloatTensor(2,4,3)
	model = GCTL(3,10,5,1,0.5)
	out = model(adj,features)
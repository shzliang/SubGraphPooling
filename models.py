#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '7/12/2020'
"""
import dgl
import dgl.function as fn  # 使用内置函数并行更新API
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalize_features
import networkx as nx
import matplotlib.pyplot as plt

class GCNLayer(nn.Module):
	""" Graph Attention Network """
	def __init__(self, in_features, out_features):
		super(GCNLayer, self).__init__()
		self.fc = nn.Linear(in_features, out_features, bias=False)

	def message_func(self, edges):
		""" message UDF """
		return {'z': edges.src['z']}

	def reduce_func(self, nodes):
		""" reduce UDF """
		h = torch.mean(nodes.mailbox['z'], dim=1)
		return {'h': h}

	def forward(self, g, h):
		""" equation (1) """
		z = self.fc(h)
		g.ndata['z'] = z
		g.update_all(self.message_func, self.reduce_func)
		g.ndata.pop('z')
		return g.ndata.pop('h')

class GATLayer(nn.Module):
	""" Graph Attention Network """
	def __init__(self, in_features, out_features,edge_dim):
		super(GATLayer, self).__init__()
		self.alpha = 0.2
		# equation (1)
		self.fc = nn.Linear(in_features, out_features, bias=False)
		# equation (2)
		self.attn_fc = nn.Linear(2 * out_features+edge_dim, 1, bias=False)

	def edge_attention(self, edges):
		"""edge UDF for equation (2)"""
		z2 = torch.cat([edges.src['z'], edges.dst['z'],edges.data['e_feat']], dim=1)
		a = self.attn_fc(z2)
		return {'e': F.leaky_relu(a,self.alpha)}

	def message_func(self, edges):
		""" message UDF for equation (3) & (4) """
		return {'z': edges.src['z'], 'e': edges.data['e']}

	def reduce_func(self, nodes):
		# reduce UDF for equation (3) & (4)
		# equation (3)
		alpha = F.softmax(nodes.mailbox['e'], dim=1)
		# equation (4)
		h = torch.sum(alpha*nodes.mailbox['z'],dim=1)
		# h = torch.bmm(alpha,nodes.mailbox['z']).view(len(nodes),-1)
		return {'h': h}

	def forward(self, g, h):
		""" equation (1) """
		z = self.fc(h)
		g.ndata['z'] = z
		# equation (2)
		g.apply_edges(self.edge_attention)
		# equation (3) & (4)
		g.update_all(self.message_func, self.reduce_func)
		g.ndata.pop('z')
		g.edata.pop('e')
		return g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
	def __init__(self, in_features, out_features, num_heads, merge='cat'):
		super(MultiHeadGATLayer, self).__init__()
		self.heads = nn.ModuleList()
		for i in range(num_heads):
			self.heads.append(GATLayer(in_features, out_features))
		self.merge = merge

	def forward(self, g, h):
		"""
		:param h: features of nodes
		:return: new representation of nodes
		"""
		head_outs = [attn_head(g,h) for attn_head in self.heads]
		if self.merge == 'cat':
			# concat on the output feature dimension (dim=1)
			return torch.cat(head_outs, dim=1)
		else:
			# merge using average
			return torch.mean(torch.stack(head_outs,dim=1),dim=1)

class ReadoutLayer(nn.Module):
	""" Graph Attention Network """
	def __init__(self, in_features):
		super(ReadoutLayer, self).__init__()
		self.att_fc = nn.Linear(in_features,1)
		# self.edge_fc = nn.Linear(edge_features,1)
		
	def node_attention(self, nodes):
		"""node UDF for equation (2)"""
		a = F.leaky_relu(self.att_fc(nodes.data['z']))
		return {'att': torch.exp(a)}
	
	def forward(self, g, h):
		g.ndata['z'] = h
		g.apply_nodes(self.node_attention)
		# g.apply_edges(self.edge_attention)
		embedding_list = []
		node_embedding = dgl.mean_nodes(g, 'z', 'att')
		# edge_embedding = dgl.mean_edges(g, 'e_feat', 'e_att')
		g.ndata.pop('z')
		g.ndata.pop('att')
		# return torch.cat([node_embedding,edge_embedding],dim=1)
		return node_embedding

# '读出和分类'
class Classifier(nn.Module):
	def __init__(self, in_features, edge_dim, hidden_dim, n_classes, node_attr_name,
				 edge_attr_name, dropout=0.3, fnc_types:dict=None,nheads=2):
		super(Classifier, self).__init__()
		# 两层GCN（图卷积）  一层线性分类
		self.dropout = dropout
		self.readout_fnc = fnc_types['readout_fnc']
		self.node_attr_name = node_attr_name
		self.edge_attr_name = edge_attr_name
		self.gat_1 = GATLayer(in_features,hidden_dim,edge_dim)
		self.gat_2 = GATLayer(hidden_dim, hidden_dim,edge_dim)
		self.gat_3 = GATLayer(hidden_dim, hidden_dim,edge_dim)
		self.readout = ReadoutLayer(hidden_dim)
		self.classifier = nn.Linear(3*hidden_dim, n_classes)
		
	def forward(self, g):
		""" forward propagation of network """
		with g.local_scope():
			h_0 = g.ndata[self.node_attr_name].squeeze()
			h_1 = F.dropout(F.relu(self.gat_1(g, h_0)), self.dropout, training=self.training) # output of first attention layer, shape:(N,nheads*hidden_dim)
			h_g1 = self.readout(g,h_1)
			h_2 = F.dropout(F.relu(self.gat_2(g, h_1)), self.dropout, training=self.training) # output: shape:(N,nheads*hidden_dim)
			h_g2 = self.readout(g,h_2)
			h_3 = F.dropout(F.relu(self.gat_3(g, h_2)), self.dropout, training=self.training)
			# h_concat = torch.cat([h_1,h_2,h_3],dim=1)
			h_g3 = self.readout(g,h_3)   # readout the embedding of graph for classification
			h_concat = torch.cat([h_g1,h_g2,h_g3],dim=1)
			y = self.classifier(h_concat)
			return F.log_softmax(y, dim=1)


def msg_fnc(edges):
	"""message function"""
	return {'edge_ID':edges.data['w']}

def reduced_fnc(nodes):
	""" reduced function """
	h = torch.sum(nodes.mailbox['edge_ID'],dim=1)
	print('mailbox',nodes.mailbox['edge_ID'])
	return {'h':h}

if __name__ == "__main__":
	g1 = dgl.DGLGraph()
	g1.add_nodes(3,data={'h':torch.eye(3)})
	g1.add_edges([0,0,1,2],[1,2,0,0],data={'w':torch.ones(size=(4,3))})
	g1.add_edges(g1.nodes(), g1.nodes())
	print(g1.ndata['h'])
	g1.update_all(message_func=msg_fnc,reduce_func=reduced_fnc)
	print(g1.edata['w'])
	print(g1.ndata['h'])
	
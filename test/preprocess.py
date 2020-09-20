#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '7/6/2020'
"""
import math
import os
import time

import dgl
from dgl.data import TUDataset, utils
import dgl.function as fn
import numpy as np
import copy as cp

from torch import nn
from tqdm import tqdm
import torch
from dgl import DGLGraph
import networkx as nx
import matplotlib.pyplot as plt

node_features = np.array([
	[0, 0, 0, 1.],
	[0, 0, 1, 0],
	[0, 0, 1, 1],
	[0, 1, 0, 0],
	[0, 1, 0, 1],
	[0, 1, 1, 0],
	[0, 1, 1, 1],
	[1, 0, 0, 0],
	[1, 0, 0, 1],
	[1, 0, 1, 0],
	[1, 0, 1, 1],
	[1, 1, 0, 0],
	[1, 1, 0, 1],
	[1, 1, 1, 0],
	[1, 1, 1, 1]
])
edge_weights = np.array([
	[0, 2, 3.],
	[0, 4, 1],
	[1, 4, 2],
	[2, 4, 2],
	[3, 4, 1],
	[4, 5, 2],
	[5, 6, 1],
	[5, 7, 2],
	[5, 8, 2],
	[6, 13, 1],
	[6, 14, 3],
	[7, 8, 1],
	[7, 11, 3],
	[7, 12, 2],
	[8, 9, 1],
	[8, 10, 2],
	[9, 10, 3],
	[11, 12, 1],
	[13, 14, 2]
])

def create_graph_bydgl(n_nodes, edges ,node_features=None, edge_weights=None):
	"""
	create an undirected graph using deep graph library(DGL)
	:param n_nodes: number of nodes in this graph, @:type:int
	:param edges: number of edges in this graph, @:type list
	:param node_features: features of nodes, default: None
	:param edge_weights: weights of edges, default: None
	:return: the object of generated graph.  type: dgl.DGLGraph
	"""
	G = DGLGraph()
	G.add_nodes(n_nodes)
	edges = np.array(edges)
	G.add_edges(edges[:,0],edges[:,1])
	G.add_edges(edges[:,1], edges[:,0])
	weights = edge_weights[:,2].reshape(-1,1)
	if not node_features is None:
		G.ndata['feature'] = torch.from_numpy(node_features)
	if not edge_weights is None:
		edge_weight = np.concatenate((weights, weights), axis=0)
		G.edata['weight'] = torch.from_numpy(edge_weight)
	return G

def BFS(graph, s, unseen_G=None, center_node=None):  # graph图  s指的是开始结点
	# 需要一个队列
	# start_time = time.time()
	unseen_graph = cp.deepcopy(unseen_G)
	visit_order = []
	all_paths = {}
	queue = [s]
	seen = set()  # 看是否访问过该结点
	seen.add(s)
	while len(queue) > 0:
		vertex = queue.pop(0)  # 保存第一结点，并弹出，方便把他下面的子节点接入
		nodes = graph.neighbors(vertex)  # 子节点的数组
		for w in nodes:
			if w not in seen:  # 判断是否访问过，使用一个数组
				queue.append(w)
				seen.add(w)
				if vertex == s:
					if not center_node is None:
						all_paths[w] = [(center_node, s),(s, w)]
					else:
						all_paths[w] = [(s, w)]
				else:
					all_paths[w] = all_paths[vertex] + [(vertex,w)]
				if (vertex,w) in unseen_graph.edges:
					unseen_graph.remove_edge(vertex,w)
		# print(vertex)
		visit_order.append(vertex)
	# end_time = time.time()
	# print("elapsed time of DFS: % .6f s" %(end_time-start_time))
	return all_paths, visit_order, unseen_graph

def DFS(graph, s):  # 图  s指的是开始结点
	# 需要一个队列
	start_time = time.time()
	all_paths = {}
	visit_order = []
	stack = [s]
	seen = set()  # 看是否访问过
	seen.add(s)
	while len(stack) > 0:
		# 拿出邻接点
		vertex = stack.pop()  # 这里pop参数没有0了，最后一个元素
		nodes = graph.neighbors(vertex)
		for w in nodes:
			if w not in seen:  # 如何判断是否访问过，使用一个数组
				stack.append(w)
				seen.add(w)
				if vertex == s:
					all_paths[w] = [s,w]
				else:
					all_paths[w] = all_paths[vertex]+[w]
					# if vertex not in all_paths[eval(w)]:
					# 	all_paths[eval(w)] += all_paths[eval(vertex)]
					# all_paths[eval(w)].append(w)
		# print(vertex)
	print(all_paths)
	end_time = time.time()
	print("elapsed time of DFS: % .6f" % (end_time - start_time))
	return all_paths, visit_order

def get_path_weight(dgl_graph:DGLGraph, path, edge_attr_name='weight'):
	"""
	calculate the product of weight on each edge in given path
	:param path: a path contains much of edges with weight
	:return: the product of weight on each edge
	"""
	path_weight = 1.
	for edge in path:
		path_weight *= dgl_graph.edata[edge_attr_name][dgl_graph.edge_id(edge[0],edge[1])][0]
	return float(path_weight)
	
def calcul_node_degree(node:int, nx_graph:nx.Graph, pre_node=None, order=1):
	"""
	calculate the n_th order degree of a node in given graph
	:param node: a node required degree
	:param nx_graph: a undirected graph
	:param order: the order of neighborhood
	:return: the degree of the given node
	"""
	node_degree = 0
	if order == 1:
		node_degree = nx_graph.adj[node].__len__()
	else:
		node_degree += nx_graph.adj[node].__len__()
		order -= 1
		neighbors = list(nx_graph.neighbors(node))
		if not pre_node is None:
			neighbors.remove(pre_node)
		for node_ in neighbors:
			node_degree += calcul_node_degree(node_,nx_graph, pre_node=node, order=order)-1
	return node_degree

def find_center_node(nx_graph):
	""" find a node located in the graph"""
	pass

def decycle(unseen_edges, paths, dgl_graph:DGLGraph, edge_attr_name='weight'):
	"""
	change a cyclic graph to acyclic graph
	:param unseen_edges: the edges not be visited in traversal of Graph using BFS
	:param paths: the visited path from source node to target node, like [(4,5),(5,6)] from 4 to 6
	:param dgl_graph: the given graph required to update the weights of itself
	:return: None
	"""
	for edge in unseen_edges:
		weight = dgl_graph.edata[edge_attr_name][dgl_graph.edge_id(edge[0],edge[1])][0]
		path_weight1 = get_path_weight(dgl_graph,paths[edge[0]],edge_attr_name)
		weight1 = cp.deepcopy(dgl_graph.edata[edge_attr_name][dgl_graph.edge_id(paths[edge[0]][-1][0],paths[edge[0]][-1][1])][0])
		path_weight2 = get_path_weight(dgl_graph, paths[edge[1]],edge_attr_name)
		weight2 = cp.deepcopy(dgl_graph.edata[edge_attr_name][dgl_graph.edge_id(paths[edge[1]][-1][0], paths[edge[1]][-1][1])][0])
		dgl_graph.edata[edge_attr_name][dgl_graph.edge_id(paths[edge[0]][-1][0],paths[edge[0]][-1][1])][0] += weight*path_weight2 * weight1/path_weight1
		dgl_graph.edata[edge_attr_name][
			dgl_graph.edge_id(paths[edge[0]][-1][1], paths[edge[0]][-1][0])][0] += weight * path_weight2 *weight1 / path_weight1
		dgl_graph.edata[edge_attr_name][
			dgl_graph.edge_id(paths[edge[1]][-1][0], paths[edge[1]][-1][1])][0] += weight * path_weight1 * weight2 / path_weight2
		dgl_graph.edata[edge_attr_name][
			dgl_graph.edge_id(paths[edge[1]][-1][1], paths[edge[1]][-1][0])][0] += weight * path_weight1 *weight2 / path_weight2
		dgl_graph.remove_edges(dgl_graph.edge_id(edge[0],edge[1]))
		dgl_graph.remove_edges(dgl_graph.edge_id(edge[1], edge[0]))

def preprocess(graphs, node_attr_name='feature',edge_attr_name='weight',dataset='MUTAG',max_label=2):
	""" preprocessing of raw data : cyclic to acyclic """
	for index,dgl_graph in tqdm(enumerate(graphs)):
		dgl_graph.edata[edge_attr_name] = torch.add(dgl_graph.edata[edge_attr_name], 1.)
		dgl_graph.edata[edge_attr_name] = dgl_graph.edata[edge_attr_name].reshape(-1,1)
		dgl_graph.ndata[node_attr_name] = dgl_graph.ndata[node_attr_name].type(dtype=torch.long)
		nx_graph = dgl_graph.to_networkx().to_undirected()
		result = []
		unvisited_node_sets = set(range(nx_graph.number_of_nodes()))
		start_node = 0
		for node in range(dgl_graph.number_of_nodes()):
			result.append(calcul_node_degree(node, nx_graph, order=3))
		start_node = result.index(max(result))
		paths, visit_order, unseen_graph = BFS(nx_graph, start_node, unseen_G=nx_graph)
		#if this graph is not strong connected graph, then search the unvisited sub_graph
		while len(visit_order)<nx_graph.number_of_nodes():
			unseen_nodes = unvisited_node_sets.difference(set(visit_order))
			node_degrees = []
			for node in unseen_nodes:
				node_degrees.append([node,calcul_node_degree(node, nx_graph, order=3)])
			node_degrees.sort(key=lambda ele:ele[1],reverse=True)
			start_n = node_degrees[0][0]
			dgl_graph.add_edges(u=[start_node,start_n],v=[start_n,start_node],data={edge_attr_name:torch.tensor([[1.],[1.]])})
			paths_, visit_order_, unseen_graph_ = BFS(nx_graph,start_n,unseen_graph,center_node=start_node)
			paths.update(paths_)
			unseen_graph = unseen_graph_
			visit_order += visit_order_ #
		graph_normalize(dgl_graph, max_label, node_attr_name, edge_attr_name)  # graph normalization in place
		decycle(unseen_graph.edges, paths, dgl_graph, edge_attr_name=edge_attr_name)
		new_graph = reduced_graph(dgl_graph,center_node=start_node,paths=paths,node_attr_name=node_attr_name,edge_attr_name=edge_attr_name)
		nodes_weight = torch.zeros(new_graph.number_of_nodes())
		nodes_weight[start_node] = 1.
		new_graph.ndata['w']= nodes_weight
		graphs[index] = new_graph
	return graphs
	
def save_graphs(graphs,graph_labels,dataset_name):
	"""
	save the graphs to file with DGLGraph format, default path: ./datasets/
	:param graphs: the graph to be saved
	:param graph_label: the labels corresponding to the graphs
	:param dataset_name: name of this dataset
	:return: None
	"""
	if not isinstance(graphs,torch.Tensor):
		graph_labels = {'graph_labels': torch.from_numpy(graph_labels)}
	else:
		graph_labels = {'graph_labels': graph_labels}
	file_path = "./datasets/" + dataset_name + '/'
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	utils.save_graphs(file_path + '{}_graphs.dg'.format(dataset_name), g_list=graphs,labels=graph_labels)

def encode_table(max_value):
	"""
	generate the full permutation for a given number
	:param max_value: the given number
	:return: the full permutation
	"""
	n_bits = list(map(int,bin(max_value)[2:])).__len__()
	encode_matrix = []
	temp = np.zeros(shape=(n_bits,))
	for i in range(max_value+1):
		bin_num = list(map(int,bin(i)[2:]))
		temp[n_bits-len(bin_num):] = bin_num[:]
		encode_matrix.append(temp.copy().tolist())
		temp[:]=0
	return torch.tensor(encode_matrix,dtype=torch.float)

def one_hot(node_features,max_label):
	"""
	convert the node label (if not None) to one hot code
	:param node_features: the features of nodes in a graph
	:return: new features of node
	"""
	encode_matrix = nn.Embedding(node_features.size(0),5)
	node_features = node_features.type(dtype=torch.long)
	new_features = encode_matrix(node_features[:,0])
	return new_features

def graph_normalize(dgl_graph,max_label,node_attr_name,edge_attr_name):
	"""
	normalize the node features and edge weights in the given graph
	:param dgl_graph: the graph required normalization
	:return: new graph
	"""
	dgl_graph.ndata[node_attr_name] = dgl_graph.ndata[node_attr_name].type(dtype=torch.long)
	dgl_graph.edata[edge_attr_name] = torch.div(dgl_graph.edata[edge_attr_name],max(dgl_graph.edata[edge_attr_name]))
	
def reduced_graph(graph:DGLGraph,center_node:int,paths:dict,node_attr_name,edge_attr_name):
	"""
	reduced graph into a simpler graph with only one center node
	:param graph: the graph need to be reduced
	:param center_node the reference node
	:param paths: the traversal path of nodes using BFS
	:return: new_graph
	"""
	new_graph = DGLGraph()
	new_graph.add_nodes(num=graph.number_of_nodes())
	new_graph.ndata[node_attr_name] = graph.ndata[node_attr_name]
	for node, path in paths.items():
		path_weight = torch.tensor([1.])
		for index,edge in enumerate(path):
			path_weight *= graph.edata[edge_attr_name][graph.edge_id(edge[0],edge[1])]*math.exp(-index)
		new_graph.add_edge(center_node,node,data={edge_attr_name:path_weight})
		new_graph.add_edge(node, center_node, data={edge_attr_name: path_weight})
	new_graph.add_edges(new_graph.nodes(), new_graph.nodes(),
	                    data={edge_attr_name: torch.ones(new_graph.number_of_nodes(), )})
	new_graph.edata[edge_attr_name] = new_graph.edata[edge_attr_name].softmax(dim=0)
	pass
	return new_graph

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

def load_dataset(data_path,raw_data_path,dataset_name):
	""" load the dataset named MUTAG
	:param data_path: the path which processed dataset existed in
	:param raw_data_path: the path which raw dataset located in
	:return new graphs and labels
	"""
	file_name = data_path + dataset_name + '/{}_graphs.dg'.format(dataset_name)
	if os.path.exists(file_name):
		data = utils.load_graphs(file_name)
		new_graphs = data[0]
		labels = data[1]
		graph_labels = labels[list(labels.keys())[0]].reshape(-1, ).type(dtype=torch.long)
	else:
		dataset = TUDataset(dataset_name)
		graphs = dataset.graph_lists
		# start_time = time.time()
		node_attr_name = 'node_labels'
		edge_attr_name = 'edge_labels'
		node_attr_file =  raw_data_path + 'tu_{}\{}\{}_node_attributes.txt'.format(dataset_name, dataset_name, dataset_name)
		node_label_file = raw_data_path + 'tu_{}\{}\{}_node_labels.txt'.format(dataset_name, dataset_name, dataset_name)
		edge_label_file = raw_data_path + 'tu_{}\{}\{}_edge_labels.txt'.format(dataset_name, dataset_name, dataset_name)
		batch_graphs = dgl.batch(graphs)
		if os.path.exists(node_attr_file):
			node_attr_name = 'node_attr'
			max_label = int(batch_graphs.ndata[node_attr_name].max())
		elif os.path.exists(node_label_file):
			max_label = int(batch_graphs.ndata[node_attr_name].max())
		else:
			batch_graphs.ndata[node_attr_name] = batch_graphs.out_degrees(batch_graphs.nodes())
			max_label = batch_graphs.out_degrees(batch_graphs.nodes()).max()
		if not os.path.exists(edge_label_file):
			batch_graphs.edata[edge_attr_name] = torch.ones(size=(batch_graphs.number_of_edges(),))
		graphs = dgl.unbatch(batch_graphs)
		graph_labels = encode_labels(dataset.graph_labels)
		new_graphs = preprocess(graphs, node_attr_name, edge_attr_name, dataset.name, max_label)
		# labels = dataset.graph_labels
		save_graphs(new_graphs, graph_labels, dataset.name)
	return new_graphs,graph_labels

if __name__ == "__main__":
	data_path = "../datasets/"
	raw_data_path = "../.dgl/"
	new_graphs,encode_labels = load_dataset(data_path,raw_data_path,'NCI109')
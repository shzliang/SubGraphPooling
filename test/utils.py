#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '2020/6/23'
"""
import os.path as osp
import time
import math
import numpy as np
import scipy.sparse as sp
import torch as pt
import torch.utils.data as Data
from torch_geometric.datasets import TUDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from visualization import Visualization


class Preprocess():
	def __init__(self, n_split=5, test_number=0):
		super(Preprocess, self).__init__()
		self.batch_size = 8
		self.n_split = n_split
		self.batch_num = self.n_split - 1
		self.test_number = test_number
		self.use_reduce_flag = 'yes'
		self.LDA = LDA(n_components=1)
		self.skf = KFold(n_splits=self.n_split,shuffle=True,random_state=43)
		self.category_expected_dataset = {'MUTAG'   : '',
		                             'PTC'     : 'PTC' ,
		                             'NCI'     : 'NCI' ,
		                             'PROTEINS': '',
		                             'COLLAB'  : '',
		                             'IMDB-B'  : ''}
		self.expected_dataset_name = { 'MUTAG'   : ['MUTAG'],
		                          'PTC'     : ['PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR'],
		                          'NCI'     : ['NCI1', 'NCI109'],
		                          'PROTEINS': ['PROTEINS'],
		                          'COLLAB'  : ['COLLAB'],
		                          'IMDB-B'  : ['IMDB-BINARY']}
		self.new_dim = 0
		
	def Dataloder(self,X, y,batch_size, new_dim):
		X_data = X.reshape(-1, new_dim, new_dim)
		X_data, y_data = X_data.astype(np.float32), y.astype(np.float32)# convert the data type of training data to float
		tor_X, tor_y = pt.from_numpy(X_data), pt.from_numpy(y_data)  # convert numpy to tensor for training data
		# for i in range(tor_X.size(0)):
		# 	tor_X[i] = self.AdjReNormalize(tor_X[i])
		dataset_ = Data.TensorDataset(tor_X, tor_y)
		dataloader_ = Data.DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True)
		return dataloader_

	def Tensor_to_Numpy(self,input_tensor):
		out_numpy = input_tensor.numpy()
		return out_numpy

	def lda(self, X, label, n_component):
		n_feature = X.shape[1]
		temp = 0
		for index in range(n_feature-n_component):
			idx = n_feature-index
			if index == 0:
				temp = self.LDA.fit(X[:,idx-2:idx], label).transform(X[:,idx-2:idx])
			else:
				temp = np.concatenate((X[:,idx-2].reshape(-1,1),temp), axis=1)
				temp = self.LDA.fit(temp, label).transform(temp)
		X_new = np.concatenate((X[:,:n_component-1], temp), axis=1)
		return X_new

	def one_hot_coding(self,input_array):
		max_label = max(input_array)
		input_array[np.argwhere(input_array != max_label)[:,0]] = -1
		input_array[np.argwhere(input_array == max_label)[:,0]] = 1
		return input_array

	def get_raw_AdjMatrix(self,dataset):
		dim_AdjMatrix_max = dataset.data.edge_index.max() + 1
		raw_AdjMatrix = np.zeros(shape=[len(dataset), dim_AdjMatrix_max, dim_AdjMatrix_max])
		for index, data in enumerate(dataset):
			if not data.x is None:
				x = self.Tensor_to_Numpy(data.x)
			else:
				x = np.ones(shape=[data.num_nodes, 1])
			x_attr = np.argwhere(x == 1)[:, 1]+1
			edge_index = self.Tensor_to_Numpy(data.edge_index)
			if not data.edge_attr is None:
				edge_attr = self.Tensor_to_Numpy(data.edge_attr)
			else:
				edge_attr = np.ones(shape=[edge_index.shape[1], 1])
			edge_weight = np.argwhere(edge_attr == 1)[:, 1]+1
			for index_edge, (row, col) in enumerate(zip(edge_index[0], edge_index[1])):
				raw_AdjMatrix[index][row, col] = edge_weight[index_edge]
			# raw_AdjMatrix[index] += np.eye(dim_AdjMatrix_max)
			# deg_mat = (raw_AdjMatrix[index]!=0).sum(axis=1).reshape(-1,1)
			# raw_AdjMatrix[index] = np.subtract(deg_mat,raw_AdjMatrix[index])
			node_index = np.unique(edge_index[0])
			for i, node_idx in enumerate(node_index):
				raw_AdjMatrix[index][node_idx, node_idx] = x_attr[i]  # node_label
		return raw_AdjMatrix
	
	def removeClass(self, X, y):
		retain_index = np.where(y != 1)
		y_ = y[retain_index]
		X_ = X[retain_index]
		return X_, y_
	
	def get_processed_AdjMatrix(self, dataset, dim_AdjMatrix):
		dim_AdjMatrix_allowed = math.ceil(math.sqrt(len(dataset))) - 1
		dim_AdjMatrix = int(dim_AdjMatrix_allowed * 0.8)
		self.new_dim = dim_AdjMatrix
		if self.use_reduce_flag == 'yes':
			if dim_AdjMatrix > dim_AdjMatrix_allowed:
				raise Exception("DimensionError: dimension entered is too big! please ensure it between 1 and %s but got %s" %(dim_AdjMatrix_allowed,dim_AdjMatrix)) # input data out of range specified
		y = self.Tensor_to_Numpy(dataset.data.y)
		raw_AdjMatrix = self.get_raw_AdjMatrix(dataset=dataset)
		if dataset.name == "COLLAB":
			raw_AdjMatrix, y = self.removeClass(raw_AdjMatrix, y)
		y_label = self.one_hot_coding(y)
		dim_AdjMatrix_allowed = math.ceil(math.sqrt(len(y_label))) - 1
		pca = PCA(n_components=dim_AdjMatrix_allowed ** 2 - 10)
		# x_and_y = x_and_y.tolist()
		# x_and_y.sort(key=lambda ele:ele[-1],reversed())
		raw_dim = raw_AdjMatrix.shape[1]  # shape: [188,28,28]
		raw_AdjMatrix = raw_AdjMatrix.reshape(len(y_label), -1)  # [188,28**2]
		# Visualization(raw_AdjMatrix, y_label, title="the distribution of raw data before reduction")
		if self.use_reduce_flag == 'yes':
			temp_data = pca.fit_transform(raw_AdjMatrix)
			# Visualization(temp_data, y_label, title="the distribution of raw data after PCA")
			temp_data = self.lda(X=temp_data, label=y_label, n_component=dim_AdjMatrix ** 2)
			# Visualization(temp_data, y_label, title="the distribution of raw data after LDA")
		else:
			self.new_dim = raw_dim
			temp_data = raw_AdjMatrix
		processed_AdjMatrix = temp_data
		# print("max_dim: {:d}" .format(dim_AdjMatrix_allowed))
		return processed_AdjMatrix, y_label
	
	def save_test_data(self, test_data, filename):
		with open('test_data/' + filename + '_test_data.txt', 'w') as f:
			for i in range(test_data.shape[0]):
				for index, value in enumerate(test_data[i]):
					f.write(str(value))
					if index != len(test_data[i]) - 1:
						f.write(' ')
				if i != test_data.shape[0] - 1:  # create a new line if not the last line
					f.write('\n')
		return None
	
	def read_test_data(self, test_data_file, new_dim):
		with open(test_data_file, 'r') as f:
			all_line = f.readlines()
		test_data = np.zeros(shape=[len(all_line), new_dim ** 2 + 1])
		for index, value in enumerate(all_line):
			if not (value.startswith('#')):
				L = value.strip()  # remove flag like '\n' or ' '
				k = L.split(' ')
				# print(len(k))
				temp = [float(value) for value in k]  # convert the type of each element in k to int
				test_data[index, :] = np.array(temp)
		# print(index)
		return test_data
	
	def AdjReNormalize(self, adj:pt.Tensor):
		deg_mat = adj.sum(dim=1).diag().float()
		inv_deg_mat = deg_mat.inverse().sqrt()
		new_adj = pt.mm(pt.mm(inv_deg_mat, adj), inv_deg_mat)
		return new_adj
	
	def P_DataLoader(self, dataset_name, test_number, new_dim):
		r"""
		:param dataset_name: name of expected dataset
		:param test_number: the current dataset order if it consists of many data sets
		:param new_dim: the dimension of reduced graph
		:return:
		"""
		path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', self.category_expected_dataset[dataset_name],
		                self.expected_dataset_name[dataset_name][test_number])
		dataset = TUDataset(path, name=self.expected_dataset_name[dataset_name][0]).shuffle()
		size_dataset = len(dataset)
		ratio = (size_dataset // 10 + size_dataset % 10 - size_dataset % 100 // 10) / size_dataset
		self.batch_size = int((size_dataset - size_dataset * ratio) / self.n_split)
		# print('Loading {} dataset ...' .format(dataset.name))
		start_time = time.time()
		processed_AdjMatrix, label = self.get_processed_AdjMatrix(dataset=dataset, dim_AdjMatrix=new_dim)
		# Visualization(processed_AdjMatrix,label,title="the distribution of raw data")
		X_Data = preprocessing.scale(processed_AdjMatrix)  # normalize data according to [mean ==0 ,std ==1]
		X_Data = processed_AdjMatrix
		X_train, X_test, y_train, y_test = train_test_split(X_Data, label, test_size=ratio, random_state=42,
		                                                    shuffle=True)  # 按5：1的比例将数据集划分为训练集与测试集
		test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)  # concatenate test_data and label at column
		# self.save_test_data(test_data, filename=dataset.name)
		X_train_KFold = self.skf.split(X_train, y_train)
		# print('finish loading! || time consume: %.3f s' %(time.time()-start_time))
		return X_train_KFold, test_data, X_train, y_train


if __name__ == "__main__":
	preprocessor = Preprocess()
	dataset_group = ['MUTAG', 'PTC', 'PROTEINS', 'COLLAB', 'IMDB-B']
	for dataset in dataset_group:
		X_train_KFold, test_data, X_train, y_train = preprocessor.P_DataLoader(dataset_name=dataset, test_number=0,
		                                                                       new_dim=5)
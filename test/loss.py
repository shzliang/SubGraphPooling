#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '2020/6/24'
"""
import math

import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class GCN_Loss(nn.Module):
	r"""
	loss of GCN for graph embedding
	"""
	def __init__(self):
		super(GCN_Loss,self).__init__()
	
	def InternalDistance(self,output:pt.Tensor, label:pt.Tensor):
		r"""
		:param output: output of gcn: shape [batch_size, n_embedding]
		:param label: node label
		:return: internal distance for all class
		"""
		data_class0 = output[label==-1]
		data_class1 = output[label==1]
		mean_vec0 = data_class0.mean(dim=0).reshape(1,-1)
		mean_vec1 = data_class1.mean(dim=0).reshape(1,-1)
		Sw0_temp = pt.matmul(pt.sub(data_class0,mean_vec0),pt.sub(data_class0,mean_vec0).transpose(1,0))
		Sw0 = Sw0_temp.diag().sum()
		Sw1_temp = pt.matmul(pt.sub(data_class1, mean_vec1), pt.sub(data_class1, mean_vec1).transpose(1, 0))
		Sw1 = Sw1_temp.diag().sum()
		Sw = Sw0+Sw1
		return Sw
	
	def BetweenDistance(self, output:pt.Tensor, label:pt.Tensor):
		r"""
		:param output: output of gcn: shape [batch_size, n_embedding]
		:param label: node label
		:return: the total distance between two class
		"""
		Sw = self.InternalDistance(output,label)
		mean_vec = output.mean(dim=0).reshape(1,-1)
		St_temp = pt.matmul(pt.sub(output, mean_vec), pt.sub(output, mean_vec).transpose(1, 0))
		St = St_temp.diag().sum()
		Sb = St - Sw
		return Sb, Sw
	
	def forward(self, output, label):
		r"""
		:param output: output of gcn: shape [batch_size, n_embedding]
		:param label: node label
		:return: loss of GCN for classification
		"""
		Sb, Sw = self.BetweenDistance(output,label)
		loss = Sw/Sb
		return loss
	
class SVM_Loss(nn.Module):
	r"""
	loss for SVM classification
	"""
	def __init__(self, tar_mg, alpha, batch_num, batch_size):
		super(SVM_Loss, self).__init__()
		self.tar_mg = tar_mg
		self.alpha = alpha
		self.relax_factor = Parameter(pt.FloatTensor(batch_num, batch_size))
		self.mu = Parameter(pt.FloatTensor(batch_num, batch_size))
		self.reset_parameters()
	
	
	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.mu.size(0))
		self.relax_factor.data.uniform_(-stdv, stdv)
		self.mu.data.uniform_(-stdv, stdv)
	
	def hinge_loss(self, input):
		hinge_loss = F.relu(1 - input)
		return hinge_loss
	
	def forward(self, output, labels, svm_weight, batch_order, mu):
		"""
		:param svm_weight: weight of SVM classifier, shape: [out_features, in_features]
		:param output: output of SVM classifier, shape:[N,1]
		:param labels: shape:[N,]
		:return: loss of SVM
		"""
		delta_func = self.hinge_loss(pt.mul(labels, pt.squeeze(output)))
		svm_loss = mu * pt.norm(svm_weight, 2, dim=1).sum() + delta_func.sum()
		return svm_loss
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '2020/6/23'
"""
import math

import torch as pt
from torch import nn
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
	r"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(pt.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(pt.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input:pt.Tensor, adj:pt.Tensor):
		if input.dim()>2:
			weight_size = self.weight.size()
			new_weight = self.weight.reshape(1,weight_size[0],-1).expand(input.size(0),weight_size[0],-1)
			support = pt.bmm(input, new_weight)
			output = pt.bmm(adj, support)
		else:
			support = pt.mm(input, self.weight)
			output = pt.mm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'
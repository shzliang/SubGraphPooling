#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '7/11/2020'
"""

import argparse
import copy
import json
import time
import networkx as nx
import dgl
import torch
import numpy  as np
import torch.nn.functional as F
from models import Classifier
from parameters import parameter_parser
from torch.utils.data import DataLoader
from utils import load_data, cross_vavildation, evaluate, save_parameters, collate,GraphDataset

def train(args):
	"""
	train and evaluate the model
	:param args: the return value of parameter parser
	:return: None
	"""
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	batch_graph, graphs, labels = load_data(args.data_path,args.raw_data_path,args.dataset,seed=args.seed)
	data_fold = cross_vavildation(graphs,args.n_split)
	batch_size = len(graphs)//args.n_split
	node_attr_name, edge_attr_name= 'n_feat', 'e_feat'
	in_features = batch_graph.ndata[node_attr_name].size(1)
	edge_features = batch_graph.edata[edge_attr_name].size(1)
	# edge_dim = batch_graph.edata['e_feat'].size(1)
	n_class = int(labels.max())+1
	torch.manual_seed(args.seed)
	fnc_types = {'message_fnc':args.message_fnc, 'reduce_fnc': args.reduce_fnc, 'readout_fnc':args.readout_fnc}
	save_parameters(args,args.dataset)
	best_acc = []
	for order,(train_idxs, test_idxs) in enumerate(data_fold):
		dur = []
		update_epoch = 0
		best_train_acc, best_test_acc = 0, 0
		count_train, count_test = [], []
		model = Classifier(in_features=in_features,edge_dim=edge_features,hidden_dim=args.hidden_dim,
						   n_classes=n_class,node_attr_name=node_attr_name, edge_attr_name=edge_attr_name,
						   dropout=args.dropout,fnc_types=fnc_types,nheads=args.n_heads)
		
		optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		model = model.to(device)
		model.train()
		train_graphs, y_train = np.array(graphs)[train_idxs.tolist()].tolist(), labels[train_idxs.tolist()].tolist()
		train_loader = DataLoader(GraphDataset(train_graphs,y_train,args.dataset),batch_size=batch_size,
								  collate_fn=collate, shuffle=True)
		train_graphs, y_train = dgl.batch(np.array(graphs)[train_idxs.tolist()].tolist()).to(device), labels[train_idxs.tolist()].to(device)
		test_graphs, y_test = dgl.batch(np.array(graphs)[test_idxs.tolist()].tolist()).to(device), labels[test_idxs.tolist()].to(device)
		print("=================== %d-fold ===================" %(order+1))
		for epoch in range(args.Epochs):
			Loss = 0.
			if epoch >= 3:
				t0 = time.time()
			for iter, (bg, label) in enumerate(train_loader):
				bg, label = bg.to(device),label.to(device)
				optimizer.zero_grad()
				output = model(bg)
				loss = F.nll_loss(output, label)
				Loss += loss
				loss.backward()
				optimizer.step()
			Loss /= iter+1
			test_acc, count_1 = evaluate(model, test_graphs, y_test)
			train_acc, count_2 = evaluate(model, train_graphs, y_train,mode='train')
			if best_test_acc<test_acc:
				best_test_acc = test_acc
				count_test = count_1
				update_epoch = epoch
			if epoch-update_epoch>=args.patience:
				break
			if best_train_acc<train_acc:
				count_train = count_2
				best_train_acc=train_acc
			if epoch >= 3:
				dur.append(time.time() - t0)
			print("Epoch {:05d} | Loss {:.4f} | Train Acc {:.2f} % | Test Acc {:.2f} % | Time(s) {:.4f} | P: {:d}/{:d} N: {:d}/{:d} | best_acc: {:.2f} %".format(epoch,
			                                                                                                    Loss.item(),
			                                                                                                    train_acc * 100,
			                                                                                                    test_acc * 100,
			                                                                                                    np.mean(
			                                                                                                dur),count_1[2],count_1[0],count_1[3],count_1[1], best_test_acc*100))
		best_acc.append(copy.deepcopy(best_test_acc))
		print("best train accuracy: {:.2f} % P: {:d}/{:d} N: {:d}/{:d}" .format(best_train_acc*100,count_train[2],count_train[0],count_train[3],count_train[1]))
		print("best test accuracy: {:.2f} % P: {:d}/{:d} N: {:d}/{:d}".format(best_test_acc*100,count_test[2],count_test[0],count_test[3],count_test[1]))
		
	print("=================== finish training ===================")
	print(["{:.2f} % ".format(acc*100) for acc in best_acc])
	print("the test accuracy: {:.2f} ± {:.2f}".format(np.mean(best_acc)*100,np.std(best_acc)*100))
	print("============= the best average accuracy: {:.2f} % ============".format(np.mean(best_acc)*100))
	# print("the best train accuracy: {:.2f} %".format(best_train_acc*100))

if __name__ == "__main__":
	# dataset = 'PTC_FM'
	args = parameter_parser()
	# best_param = json.load(open('experiment_settings.json'))[dataset]
	# args.__dict__ = best_param
	train(args)
	
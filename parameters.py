#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '7/12/2020'
"""
import argparse

import dgl
import torch


def parameter_parser():
	r"""
	A method to parse up command line parameters.
	By default it gives an embedding of the partial NCI1 graph dataset.
	The default hyper parameters give a good quality representation without grid search.
	Representations are sorted by ID.
	"""
	parser = argparse.ArgumentParser(prog='train.py',description="Run ReducedGraph.")
	parser.add_argument("--dataset", type=str, default="MUTAG", help=" dataset name.")
	parser.add_argument("--data_path", default="./datasets/", help="the path of processed dataset.")
	parser.add_argument("--raw_data_path", default="./.dgl/", help="the path of raw dataset.")
	parser.add_argument("--n_split", type=float, default=10,
	                    help="the number of splits in the datsets.")
	parser.add_argument("--embedding_dim", type=int, default=10,
	                    help="the embedding dimension of input features.")
	parser.add_argument("--n_heads", type=int, default=2, help="the number of heads.")
	parser.add_argument("--Epochs", type=int, default=300,
	                    help="the total epoches in training.")
	parser.add_argument("--seed", type=int, default=32, help="the seed of generator of random module.")
	parser.add_argument("--patience", type=int, default=100, help="the patience for bad result on test dataset.")
	parser.add_argument("--hidden_dim", type=int, default=16 , help="the number of dimension at hidden layer")
	parser.add_argument("--dropout", type=float, default=0.3,
	                    help="the dropout ratio to prevent from over-fitting in training.")
	parser.add_argument("--lr", type=float, default=5e-2,
	                    help="the learning ratio used in gradient descent optimizer, e.g. Adam.")
	parser.add_argument("--weight_decay", type=float, default=1e-4, help="the initial weight decay, default is 1e-5.")
	parser.add_argument("--message_fnc", type=str, default='src_mul_edge', help="define the type of message function")
	parser.add_argument("--reduce_fnc", type=str, default='mean', help="define the type of reduce. function")
	parser.add_argument("--readout_fnc", type=str, default='mean', help="define the type of readout. function")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	# args = parameter_parser()
	pass
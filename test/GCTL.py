#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '2020/6/23'
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from test.utils import Preprocess
from test.models import GCN, SVM
from test.loss import SVM_Loss, GCN_Loss

class Train():
	def __init__(self, preprocessor):
		super(Train, self).__init__()
		self.optimal_accuracy = 0
		self.current_epoch, self.update_epoch = 0, 0
		self.batch_order = -1
		self.occur_time_optim = 0
		self.current_epoch = 0
		self.update_epoch = 0
		self.batch_order = -1
		self.Accuracy_val, self.Accuracy_train, self.Accuracy_test = [], [], []
		self.best_val_accuracy, self.best_test_accuracy = [], []
		self.optimal_model_svm = None
		self.optimal_model_analydict = None
		self.device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
		self.preprocessor = preprocessor
		self.set_seed()
	
	def initial(self, alpha, batch_size, batch_num, in_features, n_hidden, n_embedding, out_features, dropout_ratio):
		gcn = GCN(in_features,n_hidden,n_embedding,dropout_ratio)
		svm = SVM(n_embedding,out_features)
		lossfunc = SVM_Loss(1., alpha, batch_num, batch_size)
		gcn = gcn.to(self.device)
		svm = svm.to(self.device)
		gcn_loss = GCN_Loss()
		svm_loss = SVM_Loss(1.,alpha,batch_num,batch_size)
		gcn_optimizer = pt.optim.Adam([{'params': gcn.parameters(), 'weight_decay': 1e-5, 'lr': 0.002}])
		svm_optimizer = pt.optim.Adam([{'params': svm.parameters(), 'weight_decay': 1e-5, 'lr': 0.002}])
		return gcn, svm, gcn_loss, svm_loss, gcn_optimizer, svm_optimizer
	
	def set_seed(self):
		np.random.seed(42)
		pt.manual_seed(42)
	
	def Test(self, model, data_loader):
		correct_num = 0
		total_sample = 0
		for step, (X, y) in enumerate(data_loader):
			X, y = X.to(self.device), y.to(self.device)
			output = pt.squeeze(model(X))
			correct_num += sum(pt.sign(output) == y)
			total_sample += output.size(0)
		accuracy = int(correct_num) / total_sample
		return accuracy
	
	def validation(self, X_test_loader, X_train_loader, model, epoch=None):
		self.current_epoch += 1
		accuracy_val = self.Test(model, X_test_loader)
		accuracy_train = self.Test(model, X_train_loader)
		self.Accuracy_val.append(accuracy_val)
		self.Accuracy_train.append(accuracy_train)
		if accuracy_val >= self.optimal_accuracy:  # if current accuracy is better than ever, replace optimal accuracy by it
			if accuracy_val == self.optimal_accuracy:
				self.occur_time_optim += 1  # recording the times of the same optimal value
			else:
				self.occur_time_optim = 1
			self.optimal_accuracy = accuracy_val
			self.optimal_model_svm = copy.deepcopy(model)
			self.update_epoch = self.current_epoch
	
	# if epoch:
	# 	print("================= {:03d} =================" .format(epoch))
	# 	print("validation accuracy: {0:.2f} %" .format(accuracy_val*100))
	# 	print("train accuracy:      {0:.2f} %" .format(accuracy_train*100))
	
	def show_info(self,x,y,name):
		r"""
		:param X: data at X axis
		:param y: data at y axis
		:param name: curve name
		:return: None
		"""
		plt.plot(x,y,color='blue',label=name)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.show()
	
	def update_model(self,X_train_loader, optimizer,model,lossfunc,Loss,Epoch,epoch,mu=1.):
		r"""
		:param X_train_loader: training data loader
		:param optimizer: that can execute optimization on model
		:param model: model required optimization
		:param lossfunc: evaluate the quality of model
		:param Loss: record loss value in process
		:param Epoch: record epoch value in process
		:param epoch: current optimization step
		:return: None
		"""
		for step, (X, y) in enumerate(X_train_loader):
			optimizer.zero_grad()
			self.batch_order += 1
			X, y = X.to(self.device), y.to(self.device)
			if isinstance(model,GCN):
				output = model(X)
				loss = lossfunc(output,y)
			else:
				output = model(X)
				loss = lossfunc(output, y, model[1].SVM_w,self.batch_order,mu)
			Loss.append(loss.item())
			Epoch.append(epoch)
			loss.backward()
			optimizer.step()
		self.batch_order = -1
		
	def Training(self, parameters_dict, dataset_name='MUTAG'):
		alpha = parameters_dict['alpha']
		delta = parameters_dict['delta']
		Lambda = parameters_dict['Lambda']
		mu = parameters_dict['mu']
		new_dim = parameters_dict['new_dim']
		gamma = parameters_dict['gamma']
		X_KFold, Test_data, X_train, y_train = self.preprocessor.P_DataLoader(dataset_name=dataset_name, test_number=0,
		                                                                      new_dim=new_dim)
		new_dim = self.preprocessor.new_dim
		kfold_flag = 0
		Epochs = 300
		Loss_svm, Epoch_svm = [], []
		Loss_gcn, Epoch_gcn = [], []
		batch_size, batch_num = self.preprocessor.batch_size, self.preprocessor.batch_num
		test_data_loader = self.preprocessor.Dataloder(Test_data[:, :-1], Test_data[:, -1], len(Test_data), new_dim)
		for train_data, test_data in X_KFold:
			kfold_flag += 1
			X_train_data, y_train_data = X_train[train_data], y_train[train_data]
			X_test_data, y_test_data = X_train[test_data], y_train[test_data]
			X_train_loader = self.preprocessor.Dataloder(X_train_data, y_train_data, batch_size=batch_size,
			                                             new_dim=new_dim)
			X_test_loader = self.preprocessor.Dataloder(X_test_data, y_test_data, batch_size=len(y_test_data),
			                                            new_dim=new_dim)
			# Sparse_loss, AnalyDict_loss, SVM_loss, SynthesisDict_loss = [], [], [], []
			gcn, svm, gcn_loss, svm_loss, gcn_optimizer, svm_optimizer = self.initial(alpha, batch_size, batch_num,
			                                                                          new_dim,10,5,1,0.3)
			model = pt.nn.Sequential(gcn,svm)
			Loss_svm.clear(), Loss_gcn.clear()
			Epoch_svm.clear(), Epoch_gcn.clear()
			stop_flag = 'y'  # the flag active in each stage
			while stop_flag != 'n':
				for epoch in range(Epochs):
					self.update_model(X_train_loader,gcn_optimizer,gcn,gcn_loss,Loss_gcn,Epoch_gcn,epoch)
					# self.update_model(X_train_loader,svm_optimizer,model,svm_loss,Loss_svm,Epoch_svm,epoch,mu)
					self.validation(X_test_loader, X_train_loader, model, epoch)
					if self.current_epoch - self.update_epoch >= 35 or self.occur_time_optim >= 20 or epoch >= Epochs - 1:  # stop training if the optimal value not changed for too long (default: exceed 30 epoch)
						stop_flag = 'n'
						# print("the optimal accuracy is %f %% epoch: %d" %(self.optimal_accuracy*100,self.update_epoch))
						# print("================= complete the %dth training =================" %kfold_flag)
						self.best_val_accuracy.append(copy.deepcopy(self.optimal_accuracy))
						best_test_accuracy = self.Test(self.optimal_model_svm, test_data_loader)
						# print("the best test accuracy: {:.2f} %" .format(best_test_accuracy*100))
						self.best_test_accuracy.append(best_test_accuracy)
						self.optimal_accuracy = 0
						self.current_epoch = 0
						self.occur_time_optim = 0
						break
					# self.SaveModeltoFile(kfold_flag, kfold_flag, optimal_model=[self.optimal_model_svm, self.optimal_model_analydict],dataset_name=dataset_name)  # save the model by specified file name
					#  self.show_accuracy(self.Accuracy_val,self.Accuracy_train, kfold_flag, optimal_moment=self.update_epoch,dataset_name=dataset_name)
		# print("the best accuracy at validation dataset is {:.2f} %" .format(np.max(self.best_val_accuracy)*100))
		# print("the best accuracy at test dataset is {:.2f} %" .format(np.max(self.best_test_accuracy)*100))
		return np.mean(self.best_test_accuracy)


def SVM_Test(dataset, parameters):
	preprocessor = Preprocess()
	SVM_trainer = Train(preprocessor)
	best_accuracy = SVM_trainer.Training(parameters_dict=parameters, dataset_name=dataset)
	print("SVM best test accuracy: {:.2f} %".format(best_accuracy * 100))


if __name__ == "__main__":
	default_parameters = {'alpha': 0.4, 'delta': 1e1, 'Lambda': 1e1, 'mu': 1e1, 'new_dim': 10, 'gamma': 32}
	test_dataset = 'MUTAG'
	preprocessor = Preprocess()
	Trainer = Train(preprocessor)
	best_accuracy = Trainer.Training(parameters_dict=default_parameters, dataset_name=test_dataset)
	print("best accuracy: {:.2f} %".format(best_accuracy * 100))
	print('================= finish training ====================')


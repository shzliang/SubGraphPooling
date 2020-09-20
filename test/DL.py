#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__time__ = '7/6/2019'
"""

from torch.nn import Parameter
import math
import copy
import os
import numpy as np
import torch as pt
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as Func
from testfile.plot import plot
from testfile.FileInterface import ToFile
from matplotlib import pyplot
from scipy import interpolate
from test.utils import Preprocess
from visualization import Visualization


class SVM_Classifier(nn.Module):
	def __init__(self, col_synthesis_dict, col_sample, row_sample, num_sample, device):
		super(SVM_Classifier, self).__init__()
		self.col_synthesis_dict = col_synthesis_dict
		self.num_views = col_sample
		self.num_sample = num_sample
		self.row_sample = row_sample
		self.device = device
		self.SVM_w = Parameter(pt.FloatTensor(1,col_synthesis_dict*row_sample))
		self.SVM_b = Parameter(pt.FloatTensor(1))
		self.reset_parameters()
	
	def reset_parameters(self):
		stdv = 1./math.sqrt(self.SVM_w.size(0))
		self.SVM_w.data.uniform_(-stdv, stdv)
		self.SVM_b.data.uniform_(-stdv, stdv)
	
	def forward(self, X, Analysis_Dict): # compute the output of SVM
		"""
		:param X: train data, shape: [N,V,V]
		:param Analysis_Dict: analysis dictionary, shape: [K, V]
		:return:
		"""
		P = Analysis_Dict.expand(X.size(0),self.col_synthesis_dict,self.row_sample)
		new_X = pt.bmm(P,X) # new_X  =  PX   shape: [N,K,V]
		new_X = new_X.reshape(X.size(0),-1) # new_X, shape:[N,V,K]
		output = Func.linear(new_X, self.SVM_w, self.SVM_b) # output of SVM, shape: [N,V,1]
		return output

class P_AnalysisDict(nn.Module):
	def __init__(self,col_synthesis_dict, col_sample, row_sample, num_sample, device):
		super(P_AnalysisDict, self).__init__()
		self.col_synthesis_dict = col_synthesis_dict
		self.num_views = col_sample
		self.num_sample = num_sample
		self.row_sample = row_sample
		self.device = device
		self.Analysis_Dict = Parameter(pt.Tensor(col_synthesis_dict, row_sample))
		self.reset_parameters()
	
	def reset_parameters(self):
		stdv = 1./math.sqrt(self.Analysis_Dict.size(0))
		self.Analysis_Dict.data.uniform_(-stdv,stdv)
	
	def forward(self, X): #get the rebuilt value of sparse code
		P = self.Analysis_Dict.expand(X.size(0),self.col_synthesis_dict,self.row_sample)
		out = pt.bmm(P,X)
		return out

class P_Sparse(nn.Module):
	def __init__(self, col_synthesis_dict, col_sample, row_sample, num_sample, batch_num, device):
		super(P_Sparse, self).__init__()
		self.col_synthesis_dict = col_synthesis_dict
		self.num_views = col_sample
		self.num_sample = num_sample
		self.row_sample = row_sample
		self.device = device
		self.Sparse_Code = Parameter(pt.Tensor(batch_num,num_sample, col_synthesis_dict, col_sample))
		self.reset_parameters()
	
	def reset_parameters(self):
		stdv = 1./math.sqrt(self.Sparse_Code.size(0))
		self.Sparse_Code.data.uniform_(-stdv, stdv)
	
	def forward(self, Synthesis_Dict, batch_order): #caculate the rebuilt value of X
		B = Synthesis_Dict.expand(self.num_sample,self.row_sample,self.col_synthesis_dict)
		out = pt.bmm(B,self.Sparse_Code[batch_order])
		return out

class P_SynthesisDict(nn.Module):
	def __init__(self, col_SynthesisDict, col_sample, row_sample, num_sample, device):
		super(P_SynthesisDict, self).__init__()
		self.col_synthesis_dict = col_SynthesisDict
		self.num_views = col_sample
		self.num_sample = num_sample
		self.row_sample = row_sample
		self.device = device
		self.Synthesis_Dict = Parameter(pt.Tensor(row_sample, col_SynthesisDict))
		self.reset_parameters()
	
	def reset_parameters(self):
		stdv = 1./math.sqrt(self.Synthesis_Dict.size(0))  # axis=0计算每一列的标准差
		self.Synthesis_Dict.data.uniform_(-stdv, stdv)
	
	def forward(self, Sparse_Code): # obtain the rebuilt value of sample_x
		B = self.Synthesis_Dict.expand(self.num_sample,self.row_sample,self.col_synthesis_dict)
		out = pt.bmm(B,Sparse_Code)
		return out

class Sparse_Loss(nn.Module):
	def __init__(self, tar_mg, alpha):
		super(Sparse_Loss, self).__init__()
		"""" The definition of Sparse_Loss for gradient descent based optimization method. """
		self.tar_mg = tar_mg
		self.alpha = alpha
	
	def f_norm(self, weight, p):
		"""" The definition of F_norm """
		if p != 'f':
			f_norm = self.tar_mg * pt.Tensor.norm(weight, p) ** 2
		else:
			f_norm = self.tar_mg * pt.Tensor.sum(pt.Tensor.norm(weight, 2, 0) ** 2)
		return f_norm
	
	def hinge_loss(self, input):
		hinge_loss = Func.relu(1 - input)
		return hinge_loss
	
	def forward(self, X, Synthesis_, Sparse_, Analysis_, batch_order):
		"""" The calculation of Sparse_Loss """
		Sparse_out = Sparse_(Synthesis_.Synthesis_Dict, batch_order)
		# pow_Length_si = pt.Tensor.norm(Sparse_.Sparse_Code[batch_order], 2, dim=0) ** 2
		F_norm_X_sub_BS = (pt.norm(X - Sparse_out, dim=1) ** 2).sum(dim=1).sum()
		PX_sub_S = Analysis_(X) - Sparse_.Sparse_Code[batch_order]
		norm_PX_sub_S = (pt.norm(PX_sub_S, dim=1) ** 2).sum(dim=1).sum()
		Constr_norm_S = pt.norm(Sparse_.Sparse_Code[batch_order], 2, dim=1).sum(dim=1).sum()
		Phi_S = F_norm_X_sub_BS + self.alpha*(norm_PX_sub_S +Constr_norm_S)
		return Phi_S


class Analysis_Dict_Loss(nn.Module):
	def __init__(self, tar_mg, alpha):
		super(Analysis_Dict_Loss, self).__init__()
		"""" The definition of Analysis_Dict_Loss for gradient descent based optimization method. """
		self.tar_mg = tar_mg
		self.alpha = alpha
	
	def hinge_loss(self, input):
		hinge_loss = Func.relu(1 - input)
		return hinge_loss
	
	def f_norm(self, weight, p):
		"""" The definition of F_norm """
		if p != 'f':
			f_norm = self.tar_mg * pt.Tensor.norm(weight, p) ** 2
		else:
			f_norm = self.tar_mg * pt.Tensor.sum(pt.Tensor.norm(weight, 2, 0) ** 2)
		return f_norm
	
	def vec_mul_vec(self, vec1, vec2):
		out_vec = copy.deepcopy(vec1)
		for index, value in enumerate(vec2):
			out_vec[index] = pt.mul(vec1[index], value)
		return out_vec
	
	def forward(self, X, Analysis_, Sparse_, SVM_, Label, Lambda, batch_order,relax_factor):
		"""" The calculation of Analysis_Dict_Loss"""
		SVM_out = SVM_(X, Analysis_.Analysis_Dict)
		Analysis_Dict_out = Analysis_(X)
		pow_Length_pi = pt.Tensor.norm(Analysis_.Analysis_Dict, 2, dim=0) ** 2
		F_norm_PX_Sub_S = (pt.norm(pt.sub(Analysis_Dict_out, Sparse_.Sparse_Code[batch_order]), dim=1) ** 2).sum(dim=1).sum()
		delta_func = pt.mul(Label,pt.squeeze(SVM_out))
		delta_func = self.hinge_loss(delta_func)
		Psi_P = self.alpha*F_norm_PX_Sub_S + Lambda*delta_func.sum()
		return Psi_P

class SVM_Loss(nn.Module):
	def __init__(self, tar_mg, alpha, batch_num, batch_size):
		super(SVM_Loss, self).__init__()
		self.tar_mg = tar_mg
		self.alpha = alpha
		self.relax_factor = Parameter(pt.FloatTensor(batch_num, batch_size))
		self.mu = Parameter(pt.FloatTensor(batch_num,batch_size))
		self.reset_parameters()
	
	def reset_parameters(self):
		stdv = 1./math.sqrt(self.mu.size(0))
		self.relax_factor.data.uniform_(-stdv,stdv)
		self.mu.data.uniform_(-stdv,stdv)
	
	def hinge_loss(self, input):
		hinge_loss = Func.relu(1 - input)
		return hinge_loss
	
	def f_norm(self, weight, p):
		"""" The definition of F_norm """
		if p != 'f':
			f_norm = self.tar_mg * pt.Tensor.norm(weight, p) ** 2
		else:
			f_norm = self.tar_mg * pt.Tensor.sum(pt.Tensor.norm(weight, 2, 0) ** 2)
		return f_norm
	
	def vec_mul_vec(self, vec1, vec2):
		out_vec = copy.deepcopy(vec1)
		for index, value in enumerate(vec2):
			out_vec[index] = pt.mul(vec1[index], value)
		return out_vec
	
	def forward(self, output, labels, svm_weight, batch_order, mu):
		delta_func = pt.mul(labels,pt.squeeze(output))
		delta_func = self.hinge_loss(delta_func)
		svm_loss = mu*pt.norm(svm_weight,2,dim=1).sum()  + delta_func.sum()
		return svm_loss

class Synthesis_Dict_Loss(nn.Module):
	def __init__(self, tar_mg, alpha):
		super(Synthesis_Dict_Loss, self).__init__()
		self.tar_mg = tar_mg
		self.alpha = alpha
	
	def hinge_loss(self, input):
		hinge_loss = Func.relu(input - 1)
		return hinge_loss
	
	def forward(self, X, Synthesis_, Sparse_, delta, batch_order):
		Synthesis_Dict_out = Synthesis_(Sparse_.Sparse_Code[batch_order])
		F_norm_X_Sub_BS = (pt.norm(X - Synthesis_Dict_out, dim=1) ** 2).sum(dim=1).sum()
		pow_Length_bi = pt.Tensor.norm(Synthesis_.Synthesis_Dict, 2, dim=0) ** 2
		Constr_SysDict = self.hinge_loss(pow_Length_bi)
		pounds_B = F_norm_X_Sub_BS + delta*Constr_SysDict.sum()
		return pounds_B

class Train():
	def __init__(self,preprocessor):
		super(Train,self).__init__()
		self.optimal_accuracy=0
		self.current_epoch , self.update_epoch = 0, 0
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
	
	def set_seed(self):
		np.random.seed(42)
		pt.manual_seed(42)
	
	def initial(self,alpha,new_dim,gamma,sample_num,batch_num,batch_size):
		num_atom_synthesis = math.ceil(gamma*new_dim)
		self.Analysis_Dict_ = P_AnalysisDict(num_atom_synthesis, new_dim, new_dim, sample_num, self.device)
		self.SVM_ = SVM_Classifier(num_atom_synthesis, new_dim, new_dim, batch_size, self.device)
		self.Sparse_ = P_Sparse(num_atom_synthesis, new_dim, new_dim, sample_num, batch_num, self.device)
		self.SynthesisDict_ = P_SynthesisDict(num_atom_synthesis, new_dim, new_dim, sample_num, self.device)
		self.lossfunc_svm = SVM_Loss(1., alpha,batch_num,batch_size)
		self.lossfunc_AnalyDict = Analysis_Dict_Loss(1., alpha)
		self.lossfunc_Sparse = Sparse_Loss(1., alpha)
		self.lossfunc_SynthDict = Synthesis_Dict_Loss(1., alpha)
		self.SVM_ = self.SVM_.to(self.device)
		self.Analysis_Dict_ = self.Analysis_Dict_.to(self.device)
		self.Sparse_ = self.Sparse_.to(self.device)
		self.SynthesisDict_ = self.SynthesisDict_.to(self.device)
		self.lossfunc_svm = self.lossfunc_svm.to(self.device)
		self.lossfunc_AnalyDict = self.lossfunc_AnalyDict.to(self.device)
		self.lossfunc_Sparse = self.lossfunc_Sparse.to(self.device)
		self.lossfunc_SynthDict = self.lossfunc_SynthDict.to(self.device)
		# 优化器使用Adam
		self.optimizer_svm = pt.optim.Adam([{'params': self.SVM_.parameters(), 'weight_decay': 0, 'lr': 0.002}])
		self.optimizer_AnalyDict = pt.optim.Adam([{'params': self.Analysis_Dict_.parameters(), 'weight_decay': 0, 'lr': 0.002}])
		self.optimizer_Sparse = pt.optim.Adam([{'params': self.Sparse_.parameters(), 'weight_decay': 0, 'lr': 0.002}])
		self.optimizer_SynthDict = pt.optim.Adam([{'params': self.SynthesisDict_.parameters(), 'weight_decay': 0, 'lr': 0.002}])
		self.optimal_model_svm = copy.deepcopy(self.SVM_)
		self.optimal_model_analydict = copy.deepcopy(self.Analysis_Dict_)
		return None
	
	def Test(self, SVM_, Analysis_Dict_,Test_data):
		correct_num = 0
		total_sample = 0
		correct_pos, correct_neg = 0, 0
		for step, (X,y) in enumerate(Test_data):
			X, y = X.to(self.device), y.to(self.device)
			output = SVM_(X,Analysis_Dict_.Analysis_Dict)
			output = pt.squeeze(output[:X.shape[0]])
			comp_results = pt.sign(output) == y
			correct_num += sum(comp_results)
			# tmp_result = y[comp_results==True]
			# correct_pos += (tmp_result==1).sum()
			# correct_neg += (tmp_result==-1).sum()
			total_sample += output.size(0)
		# print("positive: %d  negative: %d" %(correct_pos,correct_neg))
		accuracy = int(correct_num)/total_sample
		return accuracy
	
	def validation(self, X_test_loader, X_train_loader,new_dim, epoch=None):
		self.current_epoch += 1
		accuracy_val = self.Test(self.SVM_, self.Analysis_Dict_, X_test_loader)
		accuracy_train = self.Test(self.SVM_, self.Analysis_Dict_, X_train_loader)
		self.Accuracy_val.append(accuracy_val)
		self.Accuracy_train.append(accuracy_train)
		if accuracy_val >= self.optimal_accuracy: # if current accuracy is better than ever, replace optimal accuracy by it
			if accuracy_val == self.optimal_accuracy:
				self.occur_time_optim += 1  # recording the times of the same optimal value
			else:
				self.occur_time_optim = 1
			self.optimal_accuracy = accuracy_val
			self.optimal_model_svm = copy.deepcopy(self.SVM_)
			self.optimal_model_analydict = copy.deepcopy(self.Analysis_Dict_)
			self.update_epoch = self.current_epoch
		# if epoch:
		# 	print("================= {:03d} =================" .format(epoch))
		# 	print("validation accuracy: {0:.2f} %" .format(accuracy_val*100))
		# 	print("train accuracy:      {0:.2f} %" .format(accuracy_train*100))
		# best_test_accuracy = self.Test(self.optimal_model_svm, self.optimal_model_analydict, test_data_loader)
		# print("test accuracy: {:.2f} %" .format(best_test_accuracy*100))
		return None
	
	def SaveModeltoFile(self, SVM_order, AnalyDict_order, optimal_model,dataset_name):
		os.makedirs('./Models/'+dataset_name, exist_ok=True)
		pt.save(optimal_model[0], 'Models/'+dataset_name+'/SVM_'+dataset_name+str(SVM_order)+'_.pt')
		pt.save(optimal_model[1], 'Models/'+dataset_name+'/Analysis_Dict_'+dataset_name+str(AnalyDict_order)+'_.pt')
		return None
	
	def update_sparse(self,cyclic, optimizer_Sparse, Analysis_Dict_, lossfunc_Sparse, SynthesisDict_, Sparse_, Sparse_loss,X_train_loader):
		# update sparse_code
		loss_sparse_ = 0
		for epoch in range(cyclic):
			for step, (X, y) in enumerate(X_train_loader):
				X, y = X.to(self.device), y.to(self.device)
				self.batch_order += 1
				optimizer_Sparse.zero_grad()
				loss_sparse = lossfunc_Sparse(X, SynthesisDict_, Sparse_, Analysis_Dict_, self.batch_order)
				Sparse_loss.append(loss_sparse / X.shape[0])
				loss_sparse_ = loss_sparse_ + loss_sparse / X.shape[0]
				loss_sparse.backward()
				optimizer_Sparse.step()
			self.batch_order = -1
		# print('sparse_loss: %f' % (loss_sparse_ / self.preprocessor.batch_num))
		return None
	
	def update_analysis(self,cyclic, optimizer_AnalyDict, SVM_, Analysis_Dict_, lossfunc_AnalyDict, Sparse_, AnalyDict_loss,Lambda,X_train_loader):
		global batch_order
		# update analysis dict
		loss_AnalyDict_ = 0
		for epoch in range(cyclic):
			for step, (X, y) in enumerate(X_train_loader):
				X, y = X.to(self.device), y.to(self.device)
				self.batch_order += 1
				optimizer_AnalyDict.zero_grad()
				loss_AnalyDict = lossfunc_AnalyDict(X, Analysis_Dict_, Sparse_, SVM_, y, Lambda, self.batch_order,self.lossfunc_svm.relax_factor)
				AnalyDict_loss.append(loss_AnalyDict / X.shape[0])
				loss_AnalyDict_ = loss_AnalyDict_ + loss_AnalyDict / X.shape[0]
				loss_AnalyDict.backward()
				optimizer_AnalyDict.step()
			self.batch_order = -1
		# print('analydict_loss: %f' % (loss_AnalyDict_ / self.preprocessor.batch_num))
		return None
	
	def update_synthesis(self,cyclic, optimizer_SynthDict, SynthesisDict_, Sparse_, lossfunc_SynthDict, SynthesisDict_loss,delta,X_train_loader):
		global batch_order
		# update synthesis dict
		loss_SynthDict_ = 0
		for epoch in range(cyclic):
			for step, (X, y) in enumerate(X_train_loader):
				X, y = X.to(self.device), y.to(self.device)
				self.batch_order += 1
				optimizer_SynthDict.zero_grad()
				loss_SynthDict = lossfunc_SynthDict(X, SynthesisDict_, Sparse_, delta, self.batch_order)
				SynthesisDict_loss.append(loss_SynthDict / X.shape[0])
				loss_SynthDict_ = loss_SynthDict_ + loss_SynthDict / X.shape[0]
				loss_SynthDict.backward()
				optimizer_SynthDict.step()
			self.batch_order = -1
		# print('synthdict_loss: %f' % (loss_SynthDict_ / self.preprocessor.batch_num))
		return None
	
	def update_svm(self,cyclic, optimizer_svm, Analysis_Dict_, lossfunc_svm, SVM_, SVM_loss,mu,X_train_loader):
		# update parameters of svm
		loss_svm_ = 0
		for epoch in range(cyclic):
			for step, (X, y) in enumerate(X_train_loader):
				X, y = X.to(self.device), y.to(self.device)
				self.batch_order +=1
				optimizer_svm.zero_grad()
				output = SVM_(X, Analysis_Dict_.Analysis_Dict)
				loss_svm = lossfunc_svm(output, y, SVM_.SVM_w, self.batch_order,mu)
				SVM_loss.append(loss_svm / X.shape[0])
				loss_svm_ = loss_svm_ + loss_svm / X.shape[0]
				loss_svm.backward()
				optimizer_svm.step()
			self.batch_order = -1
		# print('svm_loss: %f' % (loss_svm_ / self.preprocessor.batch_num))
		return None
	
	def show_result(self,Sparse_loss, AnalyDict_loss, SVM_loss, SynthesisDict_loss, EPOCH, name,dataset_name):
		os.makedirs('./results/'+dataset_name, exist_ok=True)
		pyplot.figure()
		plot(EPOCH, Sparse_loss, 'Sparse_loss')
		plot(EPOCH, AnalyDict_loss, 'AnalyDict_loss')
		plot(EPOCH, SVM_loss, 'SVM_loss')
		plot(EPOCH, SynthesisDict_loss, 'SynthesisDict_loss')
		pyplot.savefig("results/" + dataset_name + "/Loss_" + dataset_name + str(name) + "_.png")
		plt.show()
		return None
	
	def Interpolate(self,x, y, kind_interpolate):
		x_new = np.linspace(x[0], x[-1], 5 * len(x))
		f_insert = interpolate.interp1d(x, y, kind=kind_interpolate)
		y_new = f_insert(x_new)
		return x_new, y_new
	
	def show_accuracy(self,accuracy_test, accuracy_train, name, optimal_moment,dataset_name):
		pyplot.figure()  # create a colorful canvas
		test_x, test_y = self.Interpolate(x=np.arange(len(accuracy_test)), y=accuracy_test, kind_interpolate="slinear")
		train_x, train_y = self.Interpolate(x=np.arange(len(accuracy_train)), y=accuracy_train, kind_interpolate="slinear")
		pyplot.plot(test_x, test_y, color='g', mec='b', mfc='w', label="test_accuracy")
		pyplot.plot(train_x, train_y, color='b', mec='b', mfc='w', label="train_accuracy")
		optimal_value = np.array(accuracy_test)
		optimal_value[:] = optimal_value[optimal_moment - 1]
		pyplot.plot(np.arange(optimal_value.__len__()), optimal_value, color='y', mec='b', mfc='w', linestyle='--',
		            label='optimal_line\noptimal: %f %%' % (optimal_value[0] * 100))  # plot a line standing optimal level
		# 设置横坐标说明
		pyplot.xlabel('Epoch')
		# 设置纵坐标说明
		pyplot.ylabel('accuracy [100%]')
		# 让图例生效
		plt.legend()
		pyplot.savefig(os.path.join("results", dataset_name, "Accuracy_"+dataset_name+str(name)+"_.png"))
		ToFile(filename=os.path.join("results/", dataset_name, "test_accuracy"+str(name)+".txt"), data=test_y)
	
	def Training(self,parameters_dict,dataset_name='MUTAG'):
		alpha=parameters_dict['alpha']
		delta=parameters_dict['delta']
		Lambda=parameters_dict['Lambda']
		mu=parameters_dict['mu']
		new_dim=parameters_dict['new_dim']
		gamma=parameters_dict['gamma']
		X_KFold, Test_data, X_train, y_train = self.preprocessor.P_DataLoader(dataset_name=dataset_name,test_number=0,new_dim=new_dim)
		new_dim = self.preprocessor.new_dim
		kfold_flag = 0
		Epochs = 300
		batch_size, batch_num = self.preprocessor.batch_size, self.preprocessor.batch_num
		test_data_loader = self.preprocessor.Dataloder(Test_data[:,:-1], Test_data[:,-1],len(Test_data),new_dim)
		cyclic = 1
		for train_data, test_data in X_KFold:
			kfold_flag += 1
			X_train_data, y_train_data = X_train[train_data], y_train[train_data]
			X_test_data, y_test_data = X_train[test_data], y_train[test_data]
			X_train_loader = self.preprocessor.Dataloder(X_train_data, y_train_data, batch_size=batch_size, new_dim=new_dim)
			X_test_loader = self.preprocessor.Dataloder(X_test_data, y_test_data, batch_size=len(y_test_data), new_dim=new_dim)
			Sparse_loss, AnalyDict_loss, SVM_loss, SynthesisDict_loss = [], [], [], []
			self.initial(alpha=alpha,new_dim=new_dim,gamma=gamma,sample_num=batch_size,batch_num=batch_num,batch_size=batch_size)
			stop_flag = 'y' # the flag active in each stage
			while stop_flag != 'n':
				for epoch in range(Epochs):
					# update sparse_code
					self.update_sparse(cyclic, self.optimizer_Sparse, self.Analysis_Dict_, self.lossfunc_Sparse, self.SynthesisDict_, self.Sparse_, Sparse_loss,X_train_loader)
					# update synthesis_dict
					self.update_synthesis(cyclic, self.optimizer_SynthDict, self.SynthesisDict_, self.Sparse_, self.lossfunc_SynthDict, SynthesisDict_loss,delta,X_train_loader)
					# update analysis_dict
					self.update_analysis(cyclic, self.optimizer_AnalyDict, self.SVM_, self.Analysis_Dict_, self.lossfunc_AnalyDict, self.Sparse_, AnalyDict_loss,Lambda,X_train_loader)
					# update weights of SVM classifier
					self.update_svm(cyclic, self.optimizer_svm, self.Analysis_Dict_, self.lossfunc_svm, self.SVM_, SVM_loss,mu,X_train_loader)
					# print("================= Epoch %s ===================" %Epoch)
					self.validation(X_test_loader, X_train_loader,new_dim,epoch) # validate the model in each epoch
					if self.current_epoch - self.update_epoch >=35 or self.occur_time_optim>=20 or epoch>=Epochs-1: # stop training if the optimal value not changed for too long (default: exceed 30 epoch)
						stop_flag = 'n'
						# print("the optimal accuracy is %f %% epoch: %d" %(self.optimal_accuracy*100,self.update_epoch))
						# print("================= complete the %dth training =================" %kfold_flag)
						self.best_val_accuracy.append(copy.deepcopy(self.optimal_accuracy))
						best_test_accuracy = self.Test(self.optimal_model_svm, self.optimal_model_analydict, test_data_loader)
						print("the best test accuracy: {:.2f} %" .format(best_test_accuracy*100))
						self.best_test_accuracy.append(best_test_accuracy)
						self.optimal_accuracy = 0
						self.current_epoch = 0
						self.occur_time_optim = 0
						break
				data, labels  = X_train_data, y_train_data
				tor_data = pt.from_numpy(data).reshape(data.shape[0],new_dim,new_dim).to(self.device).float()
				new_tor_data = self.Analysis_Dict_(tor_data)
				new_data = new_tor_data.detach().cpu().numpy().reshape(data.shape[0],-1)
				Visualization(data, labels, title="embeddings of graphs before training")
				Visualization(new_data, labels, title="embeddings of graphs after training")
				# self.SaveModeltoFile(kfold_flag, kfold_flag, optimal_model=[self.optimal_model_svm, self.optimal_model_analydict],dataset_name=dataset_name)  # save the model by specified file name
				# self.show_result(Sparse_loss, AnalyDict_loss, SVM_loss, SynthesisDict_loss, np.arange(len(Sparse_loss)), kfold_flag,dataset_name)
				# self.show_accuracy(self.Accuracy_val,self.Accuracy_train, kfold_flag, optimal_moment=self.update_epoch,dataset_name=dataset_name)
		# print("the best accuracy at validation dataset is {:.2f} %" .format(np.max(self.best_val_accuracy)*100))
		# print("the best accuracy at test dataset is {:.2f} %" .format(np.max(self.best_test_accuracy)*100))
		return np.mean(self.best_test_accuracy)

def DLSVM_Test(dataset, parameters):
	preprocessor = Preprocess()
	DLSVM_trainer = Train(preprocessor)
	best_accuracy = DLSVM_trainer.Training(parameters_dict=parameters,dataset_name=dataset)
	print("DLSVM best test accuracy: {:.2f} %" .format(best_accuracy*100))

if __name__ == "__main__":
	default_parameters = {'alpha':0.4, 'delta':1e1, 'Lambda':1e1,
	                      'mu':1e1, 'new_dim':5, 'gamma':32}
	test_dataset = 'PTC'
	preprocessor = Preprocess()
	Trainer = Train(preprocessor)
	best_accuracy = Trainer.Training(parameters_dict=default_parameters,dataset_name=test_dataset)
	print("best accuracy: {:.2f} %" .format(best_accuracy*100))
	print('================= finish training ====================')
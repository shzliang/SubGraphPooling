#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '2020/6/26'
"""
import numpy as np
from time import time
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

def Visualization(data, label, title, fig, n_component=2,dynamic=True):
	r"""
	:param data: raw data with high dimension
	:param label: label of raw data
	:param n_component
	:param dynamic
	:param title: image title
	:return: None
	"""
	if dynamic is True:
		plt.ion()
		if n_component>2:
			plt.cla()
		else:
			plt.clf()
	# 创建了一个figure，标题为"Manifold Learning with 1000 points, 10 neighbors"
	plt.suptitle(title, fontsize=14)
	'''t-SNE'''
	t0 = time()
	Tsne = manifold.TSNE(n_components=n_component, init='pca', random_state=0)
	Y = Tsne.fit_transform(data)
	t1 = time()
	print("t-SNE: %.2g sec" % (t1 - t0))
	if n_component == 3:
		fig = plt.figure(figsize=(8, 8))
		ax = Axes3D(fig)
		ax.scatter(Y[:, 0], Y[:, 1], Y[:,2], c=label)
	else:
		plt.scatter(Y[:, 0], Y[:, 1], c=label)
	plt.title("t-SNE (%.2g sec)" % (t1 - t0))
	if dynamic is False:
		plt.ioff()

if __name__ == "__main__":
	t0 = time.time()
	# fig = plt.figure(figsize=(8, 8))
	# Visualization(X,color,fig=fig,title="T-SNE reduction and Visualization")
	plt.ion()
	x=[]
	while time.time() - t0<=100:
		plt.clf()
		x.append(time.time()-t0)
		plt.plot(x,np.sin(x),color='blue')
		plt.pause(1)
	plt.ioff()
	plt.close()
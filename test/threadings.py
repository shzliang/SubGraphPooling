#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '10307'
__mtime__ = '8/3/2020'
"""
import threading
import time
import matplotlib.pyplot as plt

class myThread(threading.Thread):
	def __init__(self, ID, name):
		threading.Thread.__init__(self)
		self.threadID = ID
		self.threadName = name
		self.change_flag = False
		self.data_list = []
		self.stop_flag = False
	
	def run(self):
		plt.ion()
		fig = plt.figure(1)
		print("start to show data in realtime!")
		while True:
			if self.change_flag is True:
				x,y,label = self.data_list[0], self.data_list[1], self.data_list[2]
				plt.clf()
				plt.scatter(x,y,color=label)
				self.change_flag = False
				plt.pause(0.2)
			if self.stop_flag:
				plt.ion()
				plt.close(fig)
				break
		print(" thread {} is terminated by user!".format(self.name))


def dynamic_display(x, y, label, name):
	""" show data dynamically """
	plt.ion()
	fig = plt.figure(1)
	plt.plot(x, y, color=label)

import pickle
import csv
import torch
from copy import copy
import os
import time

class Logger_Kernel():
	def __init__(self) -> None:
		self.name = ""
		self.logfile_info = {}
		self.extra_info = {}
		self.kernel_history = []
	
	def update_base_info(self, name):
		self.name = name
	
	def update_logfile_info(self, **logfile_info):
		self.logfile_info.update(logfile_info)
	
	def update_extra_info(self, **extra_info):
		self.extra_info.update(extra_info)
	
	def update_history(self, timestamp):
		self.kernel_history.append(timestamp)

	def save(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(filename):
		with open(filename, 'rb') as f:
			return pickle.load(file=f)

	@staticmethod
	def save(kernel_logger, filename):
		kernel_logger.save(filename)
	
class CSV_Operator():
	def __init__(self) -> None:
		self.buffer = []
	
	def add(self, data):
		self.buffer.append(data)
	
	def clear(self):
		self.buffer = []
	
	def save(self, file_path):
		with open(file_path, "a", newline='') as f:
			csv_writter = csv.writer(f)
			csv_writter.writerows(self.buffer)
		self.clear()
	
class Net_Storager():
	def __init__(self) -> None:
		pass

	def save(self, net, file_path):
		torch.save(net.state_dict(), file_path)
	
	def load(self, net, file_path):
		state_dict = torch.load(file_path)
		net.load_state_dict(state_dict)

class ViT_Logger():
	def __init__(self, dir_path, net, timestamp=None) -> None:
		self.loss_logger = CSV_Operator()
		self.net_storager = Net_Storager()
		self.dir_path = dir_path
		self.net = net

		if not os.path.exists(self.dir_path):
			os.makedirs(self.dir_path)

		kernel_path = f"{self.dir_path}/kernel_{timestamp}.pkl"
		if os.path.exists(kernel_path):
			self.kernel = Logger_Kernel.load(kernel_path)
			net_path = f"{self.dir_path}/{self.kernel.logfile_info['net']}"
			self.net_storager.load(net, net_path)
		else:
			self.kernel = Logger_Kernel()
	
	def update_loss(self, loss):
		self.loss_logger.add(loss)
	
	def update_avg_loss(self, avg_loss):
		self.kernel.update_extra_info(avg_loss=avg_loss)
	
	def add_epoch(self, epochs=1):
		kernel_epoch = self.kernel.extra_info['epoch']
		self.kernel.update_extra_info(epoch= kernel_epoch + epochs)

	def save(self):
		timestamp = time.strftime("%Y%m%d%H%M%S")
		avg_loss = self.kernel.extra_info['avg_loss']
		epoch = self.kernel.extra_info['epoch']
		loss_filename = f"loss-{avg_loss}_epoch-{epoch}_timestamp-{timestamp}.csv"
		net_filename = f"loss-{avg_loss}_epoch-{epoch}_timestamp-{timestamp}.pth"
		kernel_filename = f"kernel_{timestamp}.pkl"

		loss_filepath = f"{self.dir_path}/{loss_filename}"
		net_filepath = f"{self.dir_path}/{net_filename}"
		kernel_filepath = f"{self.dir_path}/{kernel_filename}"

		self.loss_logger.save(loss_filepath)
		self.net_storager.save(self.net, net_filepath)

		self.kernel.update_logfile_info(loss=loss_filename, net=net_filename)
		self.kernel.update_history(timestamp)
		self.kernel.save(kernel_filepath)
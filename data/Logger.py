import pickle

class Logger_Kernel():
	def __init__(self) -> None:
		self.net_name = None
		self.epoch = None
		self.logfile_info = {}
		self.extra_info = {}
	
	def init_base_info(self, net_name, epoch):
		self.net_name = net_name
		self.epoch = epoch
	
	def init_logfile_info(self, **logfile_info):
		self.logfile_info = logfile_info
	
	def init_extra_info(self, **extra_info):
		self.extra_info = extra_info
	
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
	
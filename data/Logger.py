import pickle
import csv
import numpy as np
import torch
from copy import copy
import os
import time
import glob
import torchvision
from matplotlib import pyplot as plt

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

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(file=f)

    @staticmethod
    def save(kernel_logger, filename):
        with open(filename, 'wb') as f:
            pickle.dump(kernel_logger, f, pickle.HIGHEST_PROTOCOL)
    
class CSV_Operator():
    def __init__(self) -> None:
        self.buffer = []
        self.head_list = None
    
    def add(self, data):
        self.buffer.append(data)
    
    def clear(self):
        self.buffer = []

    def set_head(self, head_list, focus=False):
        if focus or self.head_list is None:
            self.head_list = head_list
    
    def save(self, file_path):
        with open(file_path, "a", newline='') as f:
            csv_writter = csv.writer(f)
            csv_writter.writerow(self.head_list)
            csv_writter.writerows(self.buffer)
        self.clear()
    
    def read(self, file_path):
        with open(file_path, "r", newline='') as f:
            csv_reader = csv.reader(f)
            l = list(csv_reader)
            name = l[0]
            data = l[1:]
        return (name, data)

class Net_Storager():
    def __init__(self) -> None:
        pass

    def save(self, net, file_path):
        torch.save(net.state_dict(), file_path)
    
    def load(self, net, file_path, dev="cpu"):
        state_dict = torch.load(file_path, map_location=dev)
        net.load_state_dict(state_dict)


class Logger():
    def __init__(self, dir_path, net, timestamp=None, load_newest=False) -> None:
        '''
        If timestamp is not, create new logger kernel, else load kernel and data
        '''

        assert not (timestamp is not None and load_newest)

        self.train_logger = CSV_Operator()
        self.eval_logger = CSV_Operator()
        self.plot_logger = CSV_Operator()
        self.net_storager = Net_Storager()
        self.dir_path = dir_path
        self.net = net
        self.kernel = None

        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        kernel_path = f"{self.dir_path}/kernel_{timestamp}.pkl"
        if timestamp is not None and os.path.exists(kernel_path):
            self.load_kernel(timestamp)
            print(f'[I] Loaded kernel, timestamp: {timestamp}')
            self.print_info()
        elif load_newest:
            net_filePathList = glob.glob(f"{self.dir_path}/kernel_*.pkl")
            max_timestamp = 0
            for net_filePath in net_filePathList:
                file_timestamp = int(net_filePath[-18: -4])
                if (file_timestamp > max_timestamp):
                    max_timestamp = file_timestamp
            if (max_timestamp > 0):
                self.load_kernel(max_timestamp)
                print(f'[I] Loaded kernel, timestamp: {max_timestamp}')
                self.print_info()
            else:
                print("[W] Cannot find any kernel, new a logger kernel")
                self.new_kernel()
        else:
            self.new_kernel()
            print("[I] New a logger kernel")
    
    def load_kernel(self, timestamp):
        kernel_path = f"{self.dir_path}/kernel_{timestamp}.pkl"
        self.kernel = Logger_Kernel.load(kernel_path)
        net_path = f"{self.dir_path}/{self.kernel.logfile_info['net']}"
        self.net_storager.load(self.net, net_path)
    
    def new_kernel(self):
        self.kernel = Logger_Kernel()
        self.kernel.update_extra_info(epoch=0, avg_loss=-1.0)
    
    def print_info(self):
        epoch = self.kernel.extra_info["epoch"]
        epoch_avg_loss = self.kernel.extra_info["avg_loss"]
        print(f'epoch: {epoch}, epoch_avg_loss: {epoch_avg_loss}')
    
    def log_train(self, epoch, loss, acc):
        self.train_logger.set_head(["epoch", "loss", "acc"])
        self.train_logger.add([epoch, loss, acc])
    
    def log_eval(self, epoch, loss, acc):
        self.eval_logger.set_head(["epoch", "loss", "acc"])
        self.eval_logger.add([epoch, loss, acc])
    
    def update_avg_loss(self, avg_loss):
        self.kernel.update_extra_info(avg_loss=avg_loss)
    
    def add_epoch(self, epochs=1):
        if 'epoch' in self.kernel.extra_info:
            kernel_epoch = self.kernel.extra_info['epoch']
        else:
            kernel_epoch = 0
        self.kernel.update_extra_info(epoch= kernel_epoch + epochs)
    
    def get_epoch(self):
        return self.kernel.extra_info["epoch"]
    
    def get_avg_loss(self):
        return self.kernel.extra_info["avg_loss"]

    def save(self):
        timestamp = time.strftime("%Y%m%d%H%M%S")
        epoch = self.kernel.extra_info['epoch']
        train_filename = f"loss_log_{timestamp}_epoch-{epoch}.csv"
        eval_filename = f"eval_log_{timestamp}_epoch-{epoch}.csv"
        net_filename = f"net_{timestamp}_epoch-{epoch}.pth"
        kernel_filename = f"kernel_{timestamp}.pkl"

        train_filepath = f"{self.dir_path}/{train_filename}"
        eval_filepath = f"{self.dir_path}/{eval_filename}"
        net_filepath = f"{self.dir_path}/{net_filename}"
        kernel_filepath = f"{self.dir_path}/{kernel_filename}"

        self.train_logger.save(train_filepath)
        self.eval_logger.save(eval_filepath)
        self.net_storager.save(self.net, net_filepath)

        self.kernel.update_logfile_info(train_log=train_filename, net=net_filename, eval_log=eval_filename)
        self.kernel.update_history(timestamp)
        Logger_Kernel.save(self.kernel, kernel_filepath)
    
    def get_log(self, log_type):
        assert log_type in ["eval", "train"]
        name_array, data_array = [], []
        kernel_history_timestamp_array = self.kernel.kernel_history
        for history_timestamp in kernel_history_timestamp_array:
            kernel_path = f"{self.dir_path}/kernel_{history_timestamp}.pkl"
            history_kernel = Logger_Kernel.load(kernel_path)
            history_log_filename = history_kernel.logfile_info[f"{log_type}_log"]
            history_log_path = f"{self.dir_path}/{history_log_filename}"
            (name, data) = self.plot_logger.read(history_log_path)
            name_array = name
            data_array = data_array + data
        data_np = np.array(data_array, dtype=float).T
        return (name_array, data_np)
    
    def plot_log(self, log_type, plot_method, save_path=None, figsize=(12, 6)):
        assert log_type in ["eval", "train"]
        assert plot_method in ["show", "save"]
        name_array, data_np = self.get_log(log_type)
        [name_epoch, name_loss, name_acc] = name_array
        [data_epoch, data_loss, data_acc] = data_np

        plt.figure(figsize=figsize)

        plt.subplot(121)
        plt.plot(data_epoch, data_loss)
        plt.xlabel(name_epoch)
        plt.ylabel(name_loss)
        plt.title(f"{log_type} {name_loss}")
        plt.grid(True)

        plt.subplot(122)
        plt.plot(data_epoch, data_acc)
        plt.xlabel(name_epoch)
        plt.ylabel(name_acc)
        plt.title(f"{log_type} {name_acc}")
        plt.grid(True)

        if (plot_method == "show"):
            plt.show()
        elif (plot_method == "save"):
            assert save_path is not None
            plt.savefig(save_path)
        

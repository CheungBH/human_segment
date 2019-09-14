# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:31:22 2019

@author: CHENG Ming
"""

from keras.utils import Sequence
from keras.utils import np_utils
import numpy as np
import os

class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args: 
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """
    def __init__(self, directory, batch_size=1, shuffle=True):
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data() 
        # Print basic statistics information
        self.print_stats()
        
    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        for i,folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory,folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file)
                # append the each file path, and keep its label  
                X_path.append(file_path)
                Y_dict[file_path] = i  
        return X_path, Y_dict
    
    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files,self.n_classes))
        for i,label in enumerate(self.dirs):
            one_hot = np_utils.to_categorical(i, self.n_classes).astype('int')
            print('%10s : '%(label),one_hot)
    
    def load_data(self, path):
        # load data with any format, please fix the code here
        data = np.load(path)
        return data

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np_utils.to_categorical(batch_y, self.n_classes)
        return batch_x, batch_y

        
def NumpyDataGenerator(directory, batch_size=16):
    X_path = []
    Y_dict = {}
    
    dirs = sorted(os.listdir(directory))
    for i,folder in enumerate(dirs):
        folder_path = os.path.join(directory,folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file)
            X_path.append(file_path)
            Y_dict[file_path] = i
     
    n_classes = len(dirs)
    steps_per_epoch = int(np.ceil(len(X_path)/batch_size))
    
    while True:
        np.random.shuffle(X_path)
        for i in range(steps_per_epoch):
            # extract data path of current batch
            batch_path = X_path[i*batch_size : (i+1)*batch_size]
            # load data and their labels 
            batch_x = [np.load(x) for x in batch_path]
            batch_y = [Y_dict[x] for x in batch_path]
            # transfer data format
            batch_x = np.array(batch_x)
            batch_y = np_utils.to_categorical(batch_y, n_classes)
            yield batch_x, batch_y
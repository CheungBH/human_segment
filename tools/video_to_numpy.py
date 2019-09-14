# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:11:14 2019

@author: CHENG Ming
"""

import cv2  
import numpy as np
import matplotlib.pyplot as plot
import os 
from tqdm import tqdm



def Video2Array(file_path, target_frames=16, resize=(112,112)):
    # 读取视频
    cap = cv2.VideoCapture(file_path)
    len_frames = cap.get(7)
    interval = np.ceil(len_frames/target_frames)
    
    samples = []
    for i in range(int(len_frames)):
        _, frame = cap.read()
        if i % interval == 0:
            frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
            samples.append(frame)
        else:
            pass
        
    len_samples = len(samples)
    if len_samples < target_frames:
        padding = samples[-1]
        for i in range(target_frames-len_samples):
            samples.append(padding)
    
    return np.array(samples)
    



def Save2Dir(file_path, save_path):
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    videos = os.listdir(file_path)
    for i,v in tqdm(enumerate(videos)):
        video_path = os.path.join(file_path, v)
        array = Video2Array(file_path=video_path, target_frames=16, resize=(112,112))
        if array.shape == (16,112,112,3):
            path = os.path.join(save_path, str(i)+'.npy') 
            np.save(path, array)
        else:
            print("Error Dim: ", file_path)
       
    return None

    
Save2Dir(file_path='video_data/fight', save_path='np_data/fight')
Save2Dir(file_path='video_data/nofight', save_path='np_data/nofight')
    

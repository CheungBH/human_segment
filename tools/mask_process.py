# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 22:36:31 2019

@author: CHENG Ming
"""

import cv2
import numpy as np
import os
from tqdm import tqdm



    
def mask_enhance(img_dir=None,mask_dir=None):
    """Mask Enhancement - Transfer the value '1' in original mask to '255', 
    which can be easily seen in plot.
    Args:
        img_dir: the path of saved images
        mask_dir: the path of masks after processing
    Returns:
        None
    """
    # Create the folder
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    # Extract path of images
    images = sorted(os.listdir(img_dir))
    # process images step by step
    for i,path in tqdm(enumerate(images)):
        # load data
        img_path = os.path.join(img_dir,path)
        img = cv2.imread(img_path)
        # binarize and resize
        ret,img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
        img = cv2.resize(img,(224,224),cv2.INTER_AREA)
        # save to dist path
        dist = os.path.join(mask_dir,str(0+i)+'.jpg')
        cv2.imwrite(dist,img)
        
    return None
        

def rename_imgs(img_dir=None, dist_dir=None, start_num=0):
    """Move images to another path, and rename them from 0 to ..., which can be 
    matching with their masks correctly.
    Args:
        img_dir: the path of saved images
        dist_dir: the path of images after renaming
        start_num: the start number of renaming 
    Returns:
        None
    """
    if not os.path.exists(dist_dir):
        os.mkdir(dist_dir)  
        images = sorted(os.listdir(img_dir))
        
    for i,path in tqdm(enumerate(images)):
        
        img_path = os.path.join(img_dir,path)
        img = cv2.imread(img_path)
        
        img = cv2.resize(img,(224,224),cv2.INTER_AREA)
        
        dist = os.path.join(dist_dir,str(start_num+i)+'.jpg')
        cv2.imwrite(dist,img)
        
#mask_enhance(img_dir='./msk', mask_dir='./masks')
#rename_imgs(img_dir='./imgs', dist_dir='./images',start_num=0)






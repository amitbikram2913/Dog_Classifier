# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:45:41 2018

@author: User
"""

import os
import pandas as pd
import glob
import tensorflow as tf
root_path = 'C:/Users/User/Downloads/Capstone Project/train1/'
labels = pd.read_csv("C:/Users/User/Downloads/Capstone Project/labels.csv")

label_uni = list(set(labels['breed']))

for folder in label_uni:

    os.mkdir(os.path.join(root_path, folder))

image_dir = 'C:/Users/User/Downloads/Capstone Project/train/'
filenames = glob.glob(image_dir + "/*.jpg")

files = os.listdir(image_dir)

for index, row in labels.iterrows():
    if not tf.gfile.Exists('C:/Users/User/Downloads/Capstone Project/train1/%s'%(row['breed'])):
        tf.gfile.MkDir('C:/Users/User/Downloads/Capstone Project/train1/%s'%(row['breed']))
#     print('train/%s.jpg'%(row['id']),'dog_train/%s'%(row['breed']))
    tf.gfile.Copy('C:/Users/User/Downloads/Capstone Project/train/%s.jpg'%(row['id']),'C:/Users/User/Downloads/Capstone Project/train1/%s/%s.jpg'%(row['breed'],row['id']),True)


#fnam = [x.split('/')[5].split('.')[0].split('\\')[1] for x in filenames]
#
#fnam1 = [item for item in fnam if item in list(labels['id'])] 
#
#img_paths = []
#for i in range(len(fnam1)):
#    img_paths.append(glob.glob(image_dir + fnam1[i] + ".jpg"))
#
#img_paths1 = pd.DataFrame(img_paths).iloc[:,0].tolist()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts as opts
from dataloader_pad import *

import json

opt = opts.parse_opt()



# Deal with feature things before anything
opt.batch_size = 1
numbers = [113287,5000,5000]
splits = ['train','val','test']
loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
opt.seq_length = loader.seq_length

isExists=os.path.exists( opt.input_attr_label_dir)
if not isExists:
    os.makedirs(opt.input_attr_label_dir) 
isExists=os.path.exists( opt.input_subsequent_label_dir)
if not isExists:
    os.makedirs(opt.input_subsequent_label_dir)
    
attr_label_path = opt.input_attr_label_dir+'/'
subsequent_label_path = opt.input_subsequent_label_dir+'/'

idx = 0
markov_mat = np.zeros([1001,1001]) #add '0' for treating start\end token as another attribute
attr_cnt = np.zeros(1001)
for split in splits:
    number = numbers[idx]
    idx+=1
    loader.reset_iterator(split)
    for i in range(number):
        data = loader.get_batch(split)
    
        labels = data['labels']
        img_id = str(data['infos'][0]['id'])
        
        #save attr_label
        attr_label = np.zeros([1000])
        for j in range(5):
            for k in range(17):
                if(labels[j][k+1] <1001 and labels[j][k+1]>0):
                    attr_label[labels[j][k+1]-1] += 1
        attr_label = (attr_label>0).astype(np.float32)    
        np.save(attr_label_path+img_id+'.npy',attr_label)
        
        #save subsequent_label
        subsequent_label = np.zeros([5,18]).astype(np.int32)
        for j in range(5):
            previous = 0
            for k in range(1,18):
                if(labels[j][-k] <1001 and labels[j][-k]>0):
                    subsequent_label[j,-k] = labels[j][-k]
                    previous = labels[j][-k]
                else:
                    subsequent_label[j,-k] = previous
 
        np.save(subsequent_label_path+img_id+'.npy',subsequent_label)        
        
        #using training data to form the markov matrix
        if(split == 'train'):
            for j in range(5):
                previous = 0
                for k in range(1,18):
                    if(subsequent_label[j][k] != previous):
                        markov_mat[previous,subsequent_label[j][k]] += 1
                        attr_cnt[subsequent_label[j][k]] += 1
                        previous = subsequent_label[j][k]        
        if(i%1000 == 0):
            print(i)
attr_cnt = np.log(attr_cnt) #inverse frequency
markov_mat = markov_mat/attr_cnt
sum_markov = np.sum(markov_mat,axis=1) 
markov_mat = (markov_mat.T/sum_markov).T #normalize
debug = 1
np.save('data/markov_mat.npy',markov_mat)


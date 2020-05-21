from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts as opts
import models
from dataloader import *
import eval_utils
import argparse
import misc.utils as utils
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
model_id = 'MADSAP0516'  #the id of the model to eval
model_index = '00163'           #checkpoint number
subsequent_mat = np.load('data/markov_mat.npy').astype(np.float32)
subsequent_mat = torch.from_numpy(subsequent_mat).cuda(device=0)

#most frequently modifed options
parser.add_argument('--beam', type=int, default=1,
                help='whether beam search')
parser.add_argument('--batch_size', type=int, default=100,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--selected_num', type=int, default=15,
                        help='num of selected attributes')
#model information
parser.add_argument('--model', type=str, default='checkpoints/'+model_id+'/model'+model_id+model_index+'.pth',
                help='path to model to evaluate')
parser.add_argument('--infos_path', type=str, default='checkpoints/'+model_id+'/infos_'+model_id+model_index+'.pkl',
                help='path to infos to evaluate')


# Basic options
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc_36',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att_36',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_box_dir', type=str, default='data/cocobu_box_36',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_attr_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='data/cocotalk_attr.json', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_loss', type=int, default=1, 
                help='if we need to calculate loss.')
parser.add_argument('--verbose', type=int, default=1,
                help='if we need to print out all beam search beams.')
                      


opt = parser.parse_args()

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_box_dir = infos['opt'].input_box_dir
    
    opt.input_attr_label_dir = infos['opt'].input_attr_label_dir
    opt.input_subsequent_label_dir = infos['opt'].input_subsequent_label_dir  
    
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id

ignore = ["id", "batch_size", "start_from", "language_eval", 'model','selected_num']
for k in vars(infos['opt']).keys():   
    if k != 'model':
        if k not in ignore:
            if k in vars(opt):
                #assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
                pass
            else:
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

#modify
opt.drop_prob_lm = 0.0




# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()

loader = DataLoader(opt)
loader.ix_to_word = infos['vocab']


# Set sample options
loss, split_predictions, lang_stats,precision,recall = eval_utils.eval_split(model, loader,subsequent_mat, vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))


isExists=os.path.exists('eval_results')
if not isExists:
    os.makedirs('eval_results') 

with open('eval_results/res'+opt.id+'.txt',"aw") as text_file:
    text_file.write('{0}\n'.format(opt.model))
    text_file.write('selected_num {0}\n'.format(opt.selected_num))
    text_file.write('{0}\n'.format(lang_stats))
    text_file.write('P:{} R:{}\n'.format(precision,recall))

json.dump(split_predictions, open('eval_results/'+opt.id+'_results.json', 'w'))

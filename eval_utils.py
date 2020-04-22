from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

infos = json.load(open('data/cocotalk_attr.json'))
vocab = infos['ix_to_word']


def array_to_str(arr):
    out = []
    for i in range(len(arr)):
        out.append(str(arr[i]))
    return out

def index_to_attr(arr):
    out = []
    for i in range(len(arr)):
        if(arr[i] == -1):
            break
        out.append(vocab[str(arr[i]+1)])
    return out

def index_to_lable(arr):
    out = []
    for i in range(len(arr)):
        if(arr[i+1] == 0):
            break
        out.append(vocab[str(arr[i+1])])
    return out


def language_eval(preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out, imgToEval

def eval_split(model, loader, subsequent_mat,eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 1)
    model_id = eval_kwargs.get('id','MAD_SAP')
    selected_num = eval_kwargs.get('selected_num',15)
    beam = eval_kwargs.get('beam',0)
    checkpoint_path = eval_kwargs.get('checkpoint_path','checkpoints')
    # Make sure in the evaluation mode
    model.eval()
    print('starting evaluation !')
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    
    precision = 0.0
    recall = 0.0
    

    
    predictions = []

    text_file = open(checkpoint_path+'/logs/cap_'+model_id+'.txt', "aw")
    text_file.close()
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
 
        if data.get('labels', None) is not None and verbose_loss:
            
            #for heh in range(5):
            #    print(index_to_lable(data['labels'][heh]))
            
            fc_feats = None
            att_feats = None


            tmp = [data['fc_feats'], data['labels'], data['masks'],data['att_feats'],data['attr_labels']]
            tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, labels, masks,att_feats,attr_labels = tmp
           

            loss = 0
            with torch.no_grad():
                _,loss,_ = model(fc_feats, att_feats, labels,masks,attr_labels,None,None,subsequent_mat)                
                loss = loss.mean()

            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats = None
        att_feats = None

        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['attr_labels'][np.arange(loader.batch_size) * loader.seq_per_img]]
               
        tmp = [torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats,attr_labels = tmp


        # forward the model to also get generated samples for each image
        
        with torch.no_grad():
            if(beam == 0):
                seq,_,attr_index = model(fc_feats, att_feats, attr_labels,subsequent_mat,mode='sample')
            else:
                seq,attr_index = model(fc_feats, att_feats, attr_labels,subsequent_mat,mode='sample_beam')
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        seq = seq.cpu().numpy()

    
                
        #compute precision and recall
        attr_labels = attr_labels.cpu().numpy()     
        for k in range(loader.batch_size):
            DAS = attr_index[k].cpu().numpy()
            detect_attribute = np.zeros([1,1000])
            detect_attribute[:,DAS] = 1
            TP = np.sum(detect_attribute * attr_labels[k])
            tp_p = TP/selected_num
            tp_r = TP/max(1,np.sum(attr_labels[k]))
            precision += tp_p
            recall += tp_r           
        

            
        #sents = sents_save_temp
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            print('image %s: %s' %(entry['image_id'], entry['caption']))
            predictions.append(entry)
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
      
    lang_stats = None
    if lang_eval == 1:
        lang_stats, scores_each = language_eval(predictions, model_id, split)

        if verbose:
            text_file = open(checkpoint_path+'/logs/cap_'+model_id+'.txt', "aw")
            for img_id in scores_each.keys():
                #print('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                #print('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n')
            text_file.close()

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats,precision/n,recall/n

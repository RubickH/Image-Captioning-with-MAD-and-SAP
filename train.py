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
import models
from dataloader import *
import eval_utils
import misc.utils as utils
import traceback
import copy
import json
from collections import OrderedDict

import multiprocessing

import sys
sys.path.append("cider-master")
from pyciderevalcap.ciderD.ciderD import CiderD

CiderD_scorer = None

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)
def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(gen_result,greedy_res,data_gts):

    batch_size = gen_result.size(0) # batch_size = sample_size * seq_per_img
    seq_per_img = 5 #default value of MSCOCO
    
    # get greedy decoding baseline
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]
    
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
  
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    

    scores = cider_scores
    print('Cider scores(sample,argmax):', np.mean(scores[:batch_size]),np.mean(scores[batch_size:2*batch_size]))
    scores = scores[:batch_size] - scores[batch_size:2*batch_size]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards



def train(opt):
    # setup dataloader
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    
    
    #set the checkpoint path
    opt.checkpoint_path =  os.path.join(opt.checkpoint_path,opt.id)
    isExists=os.path.exists(opt.checkpoint_path)
    if not isExists:
        os.makedirs(opt.checkpoint_path) 
        os.makedirs(opt.checkpoint_path+'/logs') 
        print(opt.checkpoint_path +' creating !')
    else:
        print(opt.checkpoint_path +' already exists!')
   
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + format(int(opt.start_from),'04') + '.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "att_feat_size", "rnn_size", "input_encoding_size"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + format(int(opt.start_from),'04') + '.pkl')):
            with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + format(int(opt.start_from),'04') + '.pkl')) as f:
                histories = cPickle.load(f)
      
        

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    word_loss_history = histories.get('word_loss_history', {})
    MAD_loss_history = histories.get('MAD_loss_history', {})
    SAP_loss_history = histories.get('SAP_loss_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    
    lr_history = histories.get('lr_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    #set up model, assure in training mode
    threshold = opt.threshold
    sc_flag = False
    num_gpu = opt.num_gpu
    
    model = models.setup(opt).cuda(device=0)
    model.train()
    update_lr_flag = True
    dp_model = torch.nn.parallel.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(
            os.path.join(opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from),'04')+'.pth')):
        optimizer.load_state_dict(torch.load(os.path.join(
            opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from),'04')+'.pth')))
      
    if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
        frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
        opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
        model.ss_prob = opt.ss_prob 
        
    optimizer.zero_grad()
    accumulate_iter = 0
    train_loss = 0
  
    subsequent_mat = np.load('data/markov_mat.npy')
    subsequent_mat = torch.from_numpy(subsequent_mat).cuda(device=0).float()
    subsequent_mat_all = subsequent_mat.clone()
    # for multi-GPU training
    for i in range(opt.num_gpu-1):
        subsequent_mat_all = torch.cat([subsequent_mat_all,subsequent_mat],dim=0)    
    
    
    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            for group in optimizer.param_groups:
                group['lr'] = opt.current_lr     
            
    
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
    
            # If start self critical training
            if sc_flag == False and opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                print('initializing CIDEr scorer...')
                s = time.time()
                global CiderD_scorer
                if(CiderD_scorer is None):
                    CiderD_scorer = CiderD(df=opt.cached_tokens) 
                    #takes about 30s
                    print('initlizing CIDEr scorers in {:3f}s'.format(time.time()-s))                
                sc_flag = True
                opt.learning_rate_decay_every = opt.learning_rate_decay_every*2 #default 5 for xe, 10 for scst
    
            update_lr_flag = False  
    
        print('current_lr is {}'.format(opt.current_lr))
        start = time.time()
        data = loader.get_batch('train',opt.batch_size)
        
    
        torch.cuda.synchronize()
        
    
        fc_feats = None
        att_feats = None
    
        tmp = [data['fc_feats'], data['labels'], data['masks'],data['att_feats'],data['attr_labels'],data['subsequent_labels']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda(device=0) for _ in tmp]
        fc_feats, labels, masks,att_feats,attr_labels,subsequent_labels = tmp
        
        #convert 1-1000 to 0-999 (perhaps done in preprocessing)
        subsequent_labels = subsequent_labels-1
        subsequent_mask = (subsequent_labels[:,1:]>=0).float()
        subsequent_labels = torch.where(subsequent_labels>0,subsequent_labels,torch.zeros_like(subsequent_labels).int().cuda(device=0))
        
        print('Read and process data:', time.time() - start)
        
        if not sc_flag:
            SAP_loss,word_loss,MAD_loss = dp_model(fc_feats, att_feats, labels,masks, attr_labels,subsequent_labels,subsequent_mask,subsequent_mat_all)
            SAP_loss = SAP_loss.mean()
            word_loss = word_loss.mean()
            MAD_loss = MAD_loss.mean()
            accumulate_iter = accumulate_iter + 1
            loss = (word_loss + 0.2*SAP_loss + 0.2*MAD_loss)/opt.accumulate_number  
            loss.backward()
        else:
            st = time.time()          
            sm = torch.zeros([num_gpu,1]).cuda(device=0) #indexs for sampling by probabilities
            gen_result,sample_logprobs,_ = dp_model(fc_feats, att_feats, attr_labels,subsequent_mat_all,sm, mode='sample')
            dp_model.eval()
            with torch.no_grad():
                greedy_res,_,_ = dp_model(fc_feats, att_feats,attr_labels,subsequent_mat_all, mode='sample')
            dp_model.train()
            ed = time.time()
            print('GPU time is : {}s'.format(ed-st))
            reward = get_self_critical_reward(gen_result,greedy_res,data['gts'])
            word_loss = dp_model(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda(),mode='scst_forward')
            word_loss = word_loss.mean()
       
            loss = word_loss
            
            #forward to minimize SAP loss and MAD loss
            SAP_loss, _, MAD_loss = dp_model(fc_feats, att_feats, labels,masks, attr_labels,subsequent_labels,subsequent_mask,subsequent_mat_all)
            SAP_loss = SAP_loss.mean()
            MAD_loss = MAD_loss.mean()
            loss = loss + 0.2*SAP_loss+0.2*MAD_loss
            loss.backward()             
            accumulate_iter = accumulate_iter + 1
    
        if accumulate_iter % opt.accumulate_number == 0:              
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            
            iteration += 1
            accumulate_iter = 0
            train_loss = loss.item() * opt.accumulate_number
            end = time.time()
            #you can record the training log if you need
            #text_file = open(opt.checkpoint_path+'/logs/train_log_'+opt.id+'.txt', "aw")
            if not sc_flag:
                print("iter {} (epoch {}), SAP_loss = {:.3f}, word_loss = {:.3f}, MAD_loss = {:.3f} time/batch = {:.3f}" \
                      .format(iteration, epoch,SAP_loss, word_loss,MAD_loss, end - start))
                #text_file.write("iter {} (epoch {}),SAP_loss = {:.3f}, word_loss {:.3f}, MAD_loss {:.3f},time/batch = {:.3f}\n" \
                #      .format(iteration, epoch,SAP_loss, word_loss, MAD_loss, end - start))
           
            else:
                print("iter {} (epoch {}),SAP_loss = {:.3f}, avg_reward = {:.3f},MAD_loss = {:.3f} time/batch = {:.3f}" \
                      .format(iteration, epoch,SAP_loss,np.mean(reward[:, 0]),MAD_loss, end - start))
                #text_file.write("iter {} (epoch {}), avg_reward = {:.3f} MAD_loss ={:.3f}, time/batch = {:.3f}\n" \
                #      .format(iteration, epoch, np.mean(reward[:, 0]), MAD_loss, end - start))       
            #text_file.close()
        torch.cuda.synchronize()
    
        # Update the iteration and epoch
    
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True           
    
        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0) and (accumulate_iter % opt.accumulate_number == 0):
            add_summary_value(tb_summary_writer, 'word_loss', word_loss.item(), iteration)
            add_summary_value(tb_summary_writer, 'MAD_loss', MAD_loss.item(), iteration)
            add_summary_value(tb_summary_writer, 'SAP_loss', SAP_loss.item(), iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:, 0]), iteration)
    
            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:, 0])
            word_loss_history[iteration] = word_loss.item()
            SAP_loss_history[iteration] = SAP_loss.item()
            MAD_loss_history[iteration] = MAD_loss.item()
            
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob
            
    
        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0) and (accumulate_iter % opt.accumulate_number == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json,
                           'num_images': -1,
                           'index_eval':1,
                           'id':opt.id,
                           'beam':opt.beam,
                           'verbose_loss':1,
                           'checkpoint_path':opt.checkpoint_path
                           }
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats,precision,recall = eval_utils.eval_split(dp_model,loader,subsequent_mat_all, eval_kwargs)
    
            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
            
            #save lang stats
            f_lang = open(opt.checkpoint_path+'/logs/lang_'+opt.id+'.txt','aw')
            f_lang.write(str(iteration)+' '+str(iteration/opt.save_checkpoint_every)+'\n')
            f_lang.write('val loss '+str(val_loss)+'\n')
            for key_lang in lang_stats:
                f_lang.write(key_lang+' '+str(lang_stats[key_lang])+'\n') 
            f_lang.write('precision '+str(precision)+' recall '+str(recall)+'\n')
            f_lang.close()                
          
            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss
    
            best_flag = False
            save_id = iteration/opt.save_checkpoint_every
            
            if best_val_score is None or current_score > best_val_score or current_score>threshold:
                best_val_score = current_score
                best_flag = True
                
                ##only save the improved models or when the CIDEr-D is larger than a given threshold
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model'+opt.id+format(int(save_id),'04')+'.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer'+opt.id+format(int(save_id),'04')+'.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                
                #record the lang stats for saved mdoel
                f_lang = open(opt.checkpoint_path+'/logs/Best_lang_'+opt.id+'.txt','aw')
                f_lang.write(str(iteration)+' '+str(iteration/opt.save_checkpoint_every)+'\n')
                f_lang.write('val loss '+str(val_loss)+'\n')
                for key_lang in lang_stats:
                    f_lang.write(key_lang+' '+str(lang_stats[key_lang])+'\n')                    
                f_lang.write('precision '+str(precision)+' recall '+str(recall)+'\n')
                f_lang.close()                
    
            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
    
            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['word_loss_history'] = loss_history
            histories['MAD_loss_history'] = MAD_loss_history
            histories['SAP_loss_history'] = SAP_loss_history
            
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history
            
            
            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+format(int(save_id),'04')+'.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+format(int(save_id),'04')+'.pkl'), 'wb') as f:
                cPickle.dump(histories, f)
    
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break
    

opt = opts.parse_opt()
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id
train(opt)

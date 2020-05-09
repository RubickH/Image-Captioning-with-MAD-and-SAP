from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os
#import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from .CaptionModel import CaptionModel

import json

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
class AttModel_MAD_SAP(CaptionModel):
    def __init__(self, opt):
        super(AttModel_MAD_SAP, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
       
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img

        self.selected_num = getattr(opt,'selected_num',15)

        self.lstm_core = TopDownCore_MADSAP(opt)
        

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = torch.nn.Embedding(self.vocab_size+1,self.input_encoding_size)
        
        

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.drop_prob_lm))
        self.p_att_emd = nn.Linear(self.rnn_size, self.att_hid_size)
        
        
        self.embed_det = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.feats_det = nn.Linear(self.fc_feat_size, self.att_hid_size)
        
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        
        

        self.crit = LabelSmoothing(smoothing=0.2) #LanguageModelCriterion() using label smoothing performs a little better
        self.scst_crit = RewardCriterion()
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p,0)
        return
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        return fc_feats, att_feats

    def multimodal_detector(self,att_feats, attr_labels,selected_num):
        sig = torch.nn.Sigmoid()
   
        attr_emd = self.embed_det(self.embed(torch.Tensor([range(1,1001)]).long().cuda()).detach()).squeeze(0)  #1000*512 '0' is the start/end token
        feats_emd = self.feats_det(att_feats)        #bs*max*512
        
        #compute the similarity
        b_attr =  attr_emd.t().unsqueeze(0).expand(feats_emd.shape[0],attr_emd.shape[1],attr_emd.shape[0])
        logits = torch.bmm(feats_emd,b_attr) #bs*max*1000
        p_raw = torch.log(1.0 - sig(logits)+1e-7)
        
        #merge the probability
        p_merge = torch.sum(p_raw,dim=1,keepdim=False) #bs*1000
        p_final = 1.0 - torch.exp(p_merge)
        p_final = torch.clamp(p_final,0.01,0.99)
        #print(p_final)
        top_prob,attr_index = torch.topk(p_final,selected_num,dim=1)
        
        if(attr_labels is not None and attr_labels.shape[0] == p_final.shape[0]):
        #if(attr_labels is not None):
            alpha_factor = torch.tensor(0.95).cuda()
            gamma = torch.tensor(2.0).cuda()
            alpha_factor = torch.where(torch.eq(attr_labels, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(attr_labels, 1.), 1. - p_final, p_final)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
             
            bce = -(attr_labels * torch.log(p_final) + (1.0 - attr_labels) * torch.log(1.0 - p_final))
             
            cls_loss = focal_weight * bce
             
            focal_loss = torch.sum(cls_loss)/torch.max(torch.tensor(1.0).cuda(),torch.sum(attr_labels))
        else:
            focal_loss = torch.tensor(0.0).cuda()
        

        self.p_det = p_final
        
        return focal_loss,attr_index
    
    def _forward(self, fc_feats, att_feats, seq,seq_mask, attr_labels,subsequent_labels=None,subsequent_mask=None,subsequent_mat=None):
        """

        :param fc_feats:
        :param att_feats:
        :param seq:
        :param att_masks:
        :param rela_data:
        :param ssg_data:
        :param training_mode: when this is 0, using sentence sg and do not use memory,
               when this is 1, using sentence sg and write data into memory,
               when this is 2, using image sg and read data from memory
        :return:
        """
        att_feats, att_masks = self.clip_att(att_feats, None)

        
        MAD_loss,attr_index = self.multimodal_detector(att_feats, attr_labels, self.selected_num)
    
        
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

 
        subsequent_probs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, 1000)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)
        fc_feats, att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
        
        #if training_mode >= 0:
        att_feats_mem = att_feats
        p_att_feats_mem = self.p_att_emd(att_feats_mem)  #map
        
        
        previous_attr = torch.zeros_like(seq[:, 0]).long()
        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob

                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())
                    it.index_copy_(0, sample_ind,
                                        torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            
            previous_attr = torch.where(((it<1001) * (it>0)),it,previous_attr)

            if i >= 1 and seq[:, i].sum() == 0:
                break

            subsequent_prob, output, state = self.get_logprobs_state(
                it, fc_feats, att_feats_mem, p_att_feats_mem,subsequent_mat,None, state,previous_attr)
           
            subsequent_probs[:, i] = subsequent_prob
            outputs[:, i] = output
        if(seq is not None):
            word_loss = self.crit(outputs,seq[:,1:],seq_mask[:,1:])
        else:      
            word_loss = torch.Tensor([0.0]).cuda()
        if(subsequent_mask is not None):
            SAP_loss = self.crit(subsequent_probs,subsequent_labels[:,1:].long(),subsequent_mask)
        else:
            SAP_loss = torch.Tensor([0.0]).cuda()
        return SAP_loss.unsqueeze(0), word_loss.unsqueeze(0), MAD_loss.unsqueeze(0)

    def get_logprobs_state(self, it, fc_feats, att_feats_mem, p_att_feats_mem, subsequent_mat,prob,state,previous_attr):
        # 'it' contains a word index

        xt = self.embed(it)

        if(prob is None):
            prob = self.p_det
        output,subsequent_prob, state = self.lstm_core(xt, fc_feats, att_feats_mem, p_att_feats_mem,previous_attr,\
                                                    self.embed(torch.Tensor([range(1001)]).long().cuda()).detach().squeeze(0),prob,state,subsequent_mat)

        logprobs = F.log_softmax(self.logit(output), dim=1)
     
        return subsequent_prob,logprobs,state


    def _sample_beam(self, fc_feats, att_feats,attr_labels,subsequent_mat):
        #for multi-GPU training, we remove opt={}
        # the beam size here should be the same to that in CaptionModel.py
        beam_size = 3
        batch_size = fc_feats.size(0)
        _,attr_index = self.multimodal_detector(att_feats, attr_labels, self.selected_num)        
        
        fc_feats, att_feats = self._prepare_feature(fc_feats, att_feats, None)


        att_feats_mem = att_feats
        p_att_feats_mem = self.p_att_emd(att_feats_mem)  #map


        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        subsequent_probs_all = torch.FloatTensor(self.seq_length,batch_size,1000)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
                       
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats_mem = att_feats_mem[k:k + 1].expand(*((beam_size,) + att_feats_mem.size()[1:])).contiguous()
            tmp_p_att_feats_mem = p_att_feats_mem[k:k + 1].expand(
                *((beam_size,) + p_att_feats_mem.size()[1:])).contiguous()
            tmp_previous_attr = fc_feats.new_zeros(beam_size,dtype=torch.long)
            tmp_prob = self.p_det[k:k+1].expand(beam_size, 1000)
            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                subsequent_prob,logprobs,state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats_mem, tmp_p_att_feats_mem,subsequent_mat,tmp_prob,state,tmp_previous_attr)

            self.done_beams[k] = self.beam_search(
                state, logprobs,subsequent_prob,tmp_fc_feats, tmp_att_feats_mem, tmp_p_att_feats_mem,
                subsequent_mat,tmp_prob)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
            subsequent_probs_all[:,k,:] = self.done_beams[k][0]['attr']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1).cuda(), attr_index

    def _sample(self, fc_feats, att_feats,attr_labels,subsequent_mat,sm=1,opt={}):
        att_feats, att_masks = self.clip_att(att_feats, None)
        MAD_loss,attr_index = self.multimodal_detector(att_feats, attr_labels, self.selected_num)        
             
        output = fc_feats.new_zeros(fc_feats.shape[0], self.seq_length, self.vocab_size+1)
              
        
        sample_max = sm#opt.get('sample_max', 1)
        temperature = 1.0#opt.get('temperature', 1.0)
        decoding_constraint = 0#opt.get('decoding_constraint', 0)
 

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        fc_feats, att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
        #att_feats_det = self.feats_det(att_feats)
  

        att_feats_mem = att_feats
        p_att_feats_mem = self.p_att_emd(att_feats_mem)  #map


        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        subsequent_probs = fc_feats.new_zeros(batch_size, self.seq_length)
        subsequent_probs_all = fc_feats.new_zeros(batch_size,self.seq_length,1000)
        
        previous_attr = fc_feats.new_zeros(batch_size, dtype=torch.long)
        
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing
                
            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                previous_attr = torch.where((it<1001) * (it>0),it,previous_attr)
                seq[:,t-1] = it
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)
                subsequent_probs[:,t-1] = subsequent_prob.gather(1,torch.argmax(subsequent_prob,dim=1).unsqueeze(1)).view(-1)  #tobe moditfied when using topk loss
                subsequent_probs_all[:,t-1] = subsequent_prob
                
            subsequent_prob,logprobs, state = self.get_logprobs_state(it, fc_feats, att_feats_mem, p_att_feats_mem,
                 subsequent_mat,None,state,previous_attr)
           

            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

        return seq, seqLogprobs,attr_index
    def _scst_forward(self,sample_logprobs,gen_result,reward):   
        word_loss = self.scst_crit(sample_logprobs, gen_result, reward)
        return  word_loss.unsqueeze(0)
    
    def multimodal_detector_ENS(self,att_feats):
        sig = torch.nn.Sigmoid()


        attr_emd = self.embed_det(self.embed(torch.Tensor([range(1,1001)]).long().cuda()).detach()).squeeze(0)  #1000*512 '0' is the start/end token
        feats_emd = self.feats_det(att_feats)        #bs*max*512

        #compute the similarity
        b_attr =  attr_emd.t().unsqueeze(0).expand(feats_emd.shape[0],attr_emd.shape[1],attr_emd.shape[0])
        logits = torch.bmm(feats_emd,b_attr) #bs*max*1000
        p_raw = torch.log(1.0 - sig(logits)+1e-7)

        #merge the probability
        p_merge = torch.sum(p_raw,dim=1,keepdim=False) #bs*1000
        p_final = 1.0 - torch.exp(p_merge)
        p_final = torch.clamp(p_final,0.01,0.99)
        #print(p_final)

        return p_final



class subsequent_attribute_predictor(nn.Module):
    def __init__(self,opt):
        super(subsequent_attribute_predictor,self).__init__()
        self.selected_num = opt.selected_num
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm        

  
        self.f2logit = nn.Linear(2*self.rnn_size,self.rnn_size)
                                 
        self.attr_fc = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                 nn.LeakyReLU(0.1,inplace=True),
                                 nn.Dropout(self.drop_prob_lm))
        self.attr_fc2 = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                 nn.LeakyReLU(0.1,inplace=True),
                                 nn.Dropout(self.drop_prob_lm))
        self.attr_fc3 = nn.Linear(self.rnn_size,1000)        
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p,0)
        return        
    def forward(self,word_embedding,prob_det,h_att,previous_attr,subsequent_mat):
        x1 = torch.mm(subsequent_mat, word_embedding)
        x2 = self.attr_fc(x1)
        x3 = torch.mm(subsequent_mat,x2)
        x4 = self.attr_fc2(x3)
        x5 = self.attr_fc3(x4)        
        susequent_attr_embedding = x5[previous_attr]
        
        input_feat = torch.cat([h_att,susequent_attr_embedding],dim=1)
        logits = prob_det*self.f2logit(input_feat)
        subsequent_probs = F.log_softmax(logits,dim=-1)
        final_prob,final_index = torch.topk(subsequent_probs,self.selected_num)
        
        attr_embedding = word_embedding[final_index+1]

        return subsequent_probs,attr_embedding


class TopDownCore_MADSAP(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore_MADSAP, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        #self.att_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        #self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.attention_attr = Attention(opt)
        
        self.SAP = subsequent_attribute_predictor(opt)
        self.attr_emd = nn.Sequential(nn.Linear(opt.input_encoding_size, opt.rnn_size),
                                      nn.ReLU(inplace=True),
                                    nn.Dropout(opt.drop_prob_lm))
        self.p_attr_emd = nn.Linear(opt.rnn_size, opt.att_hid_size)
        
        self.dropout = nn.Dropout(self.drop_prob_lm)
        
        self.init_weight()
        
    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p,0)
        return        
    def forward(self, xt, fc_feats, att_feats, p_att_feats,previous_attr,word_embedding,prob_det,state,subsequent_mat):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([self.dropout(prev_h), fc_feats, xt], 1)
        #att_lstm_input = torch.cat([prev_h, xt], 1)
        #att_lstm_input = torch.cat([fc_feats, xt], 1)

        # state[0][0] means the hidden state c in first lstm
        # state[1][0] means the cell h in first lstm
        # state[0] means hidden state and state[1] means cell, state[0][i] means
        # the i-th layer's hidden state
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        
        #select the guiding attributes
        subsequent_probs,attr_embedding = self.SAP(word_embedding,prob_det,h_att,previous_attr,subsequent_mat)
        attr_embedding = self.attr_emd(attr_embedding)
        p_attr_emd = self.p_attr_emd(attr_embedding)
        
        
        att = self.attention(h_att, att_feats, p_att_feats, None)
        attr = self.attention_attr(h_att,attr_embedding,p_attr_emd, None)
        
        lang_lstm_input = torch.cat([att, attr,self.dropout(h_att)], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = self.dropout(h_lang)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output,subsequent_probs, state


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p,0)
        return
    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
       # if(att_size == 10):
       #     print(weight)
       #     debug = 1
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class LSTM_MAD_SAP(AttModel_MAD_SAP):
    def __init__(self, opt):
        super(LSTM_MAD_SAP, self).__init__(opt)
        self.num_layers = 2


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
    
class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self,input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)
    
        return output

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None
        
    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()

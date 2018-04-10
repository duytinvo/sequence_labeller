#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:41:43 2018

@author: dtvo
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
use_cuda = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Embs(nn.Module):
    """
    This module builds an embeddings layer with BiLSTM model,
    which can be used at both character and word level
    """
    def __init__(self, HPs):
        super(Embs, self).__init__()
        [size, dim, pre_embs, hidden_dim, dropout, layers, bidirect] = HPs
        self.layers = layers
        self.bidirect = bidirect
        self.hidden_dim = hidden_dim // 2 if bidirect else hidden_dim
            
        self.embeddings = nn.Embedding(size, dim)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))

        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(dim, self.hidden_dim, num_layers=layers, batch_first=True, bidirectional=bidirect)
        
        self.att_layer = nn.Linear(hidden_dim,1, bias=False)
        self.softmax = nn.Softmax(-1)
        if use_cuda:
            self.embeddings = self.embeddings.cuda()
            self.drop = self.drop.cuda()
            self.lstm = self.lstm.cuda()
            self.att_layer = self.att_layer.cuda()
            self.softmax = self.softmax.cuda()
    
    def forward(self, inputs, input_lengths):
        return self.get_all_hiddens(inputs, input_lengths)
        
    def get_last_hiddens(self, inputs, input_lengths):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        # set zero vector for padding, unk, eot, sot
        self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        hc_0 = self.initHidden(batch_size)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
        return  h_n

    def get_last_atthiddens(self, inputs, input_lengths=None):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        # set zero vector for padding, unk, eot, sot
        self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        hc_0 = self.initHidden(batch_size)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
                
        #(batch_size, seq_length, 1)
        att_features = F.relu(self.att_layer(rnn_out))
         #(batch_size, seq_length)
        att_features.squeeze_()
        alpha = self.softmax(att_features)
        att_embs = embs_drop*alpha.view(batch_size,seq_length,1)
        att_h = att_embs.sum(1)
        features = torch.cat([h_n, att_h], -1)
        return  features

    def get_all_hiddens(self, inputs, input_lengths=None):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        # set zero vector for padding, unk, eot, sot
        self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        hc_0 = self.initHidden(batch_size)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        return  rnn_out

    def get_all_atthiddens(self, inputs, input_lengths=None):
        """
            input:  
                input: Variable(batch_size, word_length)
                seq_lengths: numpy array (batch_size,  1)
            output: 
                Variable(batch_size, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        # set zero vector for padding, unk, eot, sot
        self.set_zeros([0,1,2,3])
        batch_size = inputs.size(0)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        hc_0 = self.initHidden(batch_size)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
        
        #(batch_size,seq_len,1)
        att_features = F.relu(self.att_layer(rnn_out))
        #(batch_size,seq_len)
        att_features.squeeze_()
        #(batch_size,seq_len)
        alpha = self.softmax(att_features)
        att_hidden = h_n.view(batch_size,1,-1)*alpha.view(batch_size,-1,1)
        features = torch.cat([rnn_out,att_hidden], -1)
        return  features
        
    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index,:] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs

    def initHidden(self, batch_size):
        d = 2 if self.bidirect else 1
        h = Variable(torch.zeros(self.layers*d, batch_size, self.hidden_dim))
        c = Variable(torch.zeros(self.layers*d, batch_size, self.hidden_dim))
        if use_cuda:
            return h.cuda(), c.cuda()
        else:
            return h,c
    
    def set_zeros(self,idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)
                 
class CW_bisltm(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism 
    to pass through another biLSTM for extracting final features for affine layers
    """
    def __init__(self, word_HPs, char_HPs=[], num_labels=None, drop_final=0.5, att=False):
        super(CW_bisltm, self).__init__()
        self.num_labels = num_labels
        [word_size, word_dim, word_pre_embs, word_hidden_dim, word_dropout, word_layers, word_bidirect] = word_HPs
        
        if char_HPs:
            self.use_char = True
            [char_size, char_dim, char_pred_embs, char_hidden_dim, char_dropout, char_layers, char_bidirect] = char_HPs
            self.char_embs_rnn = Embs(char_HPs)
            input_dim = char_hidden_dim + word_dim
            if att:
                input_dim += char_dim
        else:
            self.use_char = False
            input_dim = word_dim
            
        
        self.layers = word_layers
        self.bidirect = word_bidirect
        self.hidden_dim = word_hidden_dim // 2 if word_bidirect else word_hidden_dim

        self.embeddings = nn.Embedding(word_size, word_dim)
        if word_pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(word_pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(word_size, word_dim)))
            
        self.drop = nn.Dropout(word_dropout)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=word_layers, batch_first=True, bidirectional=word_bidirect)
        self.att_layer = nn.Linear(word_hidden_dim,1, bias=False)
        self.softmax = nn.Softmax(-1)    
        self.dropfinal = nn.Dropout(drop_final)

        if num_labels > 2:
            self.hidden2tag = nn.Linear(word_hidden_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss() 
        
        if use_cuda:
            self.embeddings = self.embeddings.cuda()
            self.drop = self.drop.cuda()
            self.lstm = self.lstm.cuda()
            self.att_layer = self.att_layer.cuda()
            self.softmax = self.softmax.cuda()
            self.dropfinal = self.dropfinal.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.lossF = self.lossF.cuda()            
    
    def forward(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        h_n = self.get_last_hiddens(word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover)
        label_score = self.hidden2tag(h_n)
        label_score = self.dropfinal(label_score)
        return label_score
    
    def NLL_loss(self, label_score, label_tensor):  
        if self.num_labels > 2:
            batch_loss = self.lossF(label_score, label_tensor)
        else:
            batch_loss = self.lossF(label_score, label_tensor.float().view(-1,1))
        return batch_loss 

    def inference(self, label_score):
        if self.num_labels > 2:
            label_prob, label_pred = label_score.data.max(1)
        else:
            label_prob = F.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred
            
    def get_last_hiddens(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        # set zero vector for padding, unk, eot, sot
        self.set_zeros([0,1,2,3])
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)
        word_embs = self.embeddings(word_inputs)
        if self.use_char:
            char_embs = self.char_embs_rnn.get_last_hiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length, -1)
            
            word_embs = torch.cat([char_embs, word_embs], -1)
            
        embs_drop = self.drop(word_embs)
        hc_0 = self.initHidden(word_batch)
        pack_input = pack_padded_sequence(embs_drop, word_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0) 
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
        return h_n

    def get_last_atthiddens(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        # set zero vector for padding, unk, eot, sot
        self.set_zeros([0,1,2,3])
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)
        word_embs = self.embeddings(word_inputs)
        if self.use_char:
            char_embs = self.char_embs_rnn.get_last_hiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length, -1)
            
            word_embs = torch.cat([char_embs, word_embs], -1)
            
        embs_drop = self.drop(word_embs)
        hc_0 = self.initHidden(word_batch)
        pack_input = pack_padded_sequence(embs_drop, word_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0) 
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)

        #(batch_size, seq_length, 1)
        att_features = F.relu(self.att_layer(rnn_out))
         #(batch_size, seq_length)
        att_features.squeeze_()
        alpha = self.softmax(att_features)
        att_embs = embs_drop*alpha.view(word_batch,seq_length,1)
        att_h = att_embs.sum(1)
        features = torch.cat([h_n, att_h], -1)    
        return features

    def get_all_hiddens(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        # set zero vector for padding, unk, eot, sot
        self.set_zeros([0,1,2,3])
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)
        word_embs = self.embeddings(word_inputs)
        if self.use_char:
            char_embs = self.char_embs_rnn.get_last_hiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length,-1)
            word_embs = torch.cat([char_embs, word_embs], -1)
        
        embs_drop = self.drop(word_embs)
        hc_0 = self.initHidden(word_batch)
        pack_input = pack_padded_sequence(embs_drop, word_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0) 
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        return rnn_out

    def get_all_atthiddens(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        # set zero vector for padding, unk, eot, sot
        self.set_zeros([0,1,2,3])
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)
        word_embs = self.embeddings(word_inputs)
        if self.use_char:
            char_embs = self.char_embs_rnn.get_last_atthiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length, -1)
            word_embs = torch.cat([char_embs, word_embs], -1)
            
        embs_drop = self.drop(word_embs)
        hc_0 = self.initHidden(word_batch)
        pack_input = pack_padded_sequence(embs_drop, word_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0) 
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)

        #(batch_size,seq_len,1)
        att_features = F.relu(self.att_layer(rnn_out))
        #(batch_size,seq_len)
        att_features.squeeze_()
        #(batch_size,seq_len)
        alpha = self.softmax(att_features)
        att_hidden = h_n.view(word_batch,1,-1)*alpha.view(word_batch,-1,1)
        word_features = torch.cat([rnn_out,att_hidden], -1)
        return word_features
         
    def initHidden(self, batch_size):
        d = 2 if self.bidirect else 1
        h = Variable(torch.zeros(self.layers*d, batch_size, self.hidden_dim))
        c = Variable(torch.zeros(self.layers*d, batch_size, self.hidden_dim))
        if use_cuda:
            return h.cuda(), c.cuda()
        else:
            return h,c
        
    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index,:] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs

    def set_zeros(self,idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)

class Deep_bisltm(nn.Module):
    """
    The model builds character biLSTM, concatenated by word biLSTM with attentional mechanism 
    to pass through another biLSTM for extracting final features for affine layers
    """
    def __init__(self, word_HPs, char_HPs=[], num_labels=None, drop_final=0.5, att=False):
        super(Deep_bisltm, self).__init__()
        self.num_labels = num_labels
        [word_size, word_dim, word_pre_embs, word_hidden_dim, word_dropout, word_layers, word_bidirect] = word_HPs
        
        word_HPs = [word_size, word_dim, word_pre_embs, word_dim, word_dropout, word_layers, word_bidirect]
        self.word_embs_rnn = Embs(word_HPs)
        
        if char_HPs:
            self.use_char = True
            [char_size, char_dim, char_pred_embs, char_hidden_dim, char_dropout, char_layers, char_bidirect] = char_HPs
            self.char_embs_rnn = Embs(char_HPs)
            input_dim = char_hidden_dim + word_dim
            if att:
                input_dim += char_dim
                input_dim += word_dim
        else:
            self.use_char = False
            input_dim = word_dim
            
        
        self.layers = word_layers
        self.bidirect = word_bidirect
        self.hidden_dim = word_hidden_dim // 2 if word_bidirect else word_hidden_dim
            
        self.drop = nn.Dropout(word_dropout)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=word_layers, batch_first=True, bidirectional=word_bidirect)
        self.att_layer = nn.Linear(word_hidden_dim,1, bias=False)
        self.softmax = nn.Softmax(-1)    
        self.dropfinal = nn.Dropout(drop_final)

        if num_labels > 2:
            self.hidden2tag = nn.Linear(word_hidden_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss() 
        
        if use_cuda:
            self.drop = self.drop.cuda()
            self.lstm = self.lstm.cuda()
            self.att_layer = self.att_layer.cuda()
            self.softmax = self.softmax.cuda()
            self.dropfinal = self.dropfinal.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.lossF = self.lossF.cuda()            
    
    def forward(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        h_n = self.get_last_hiddens(word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover)
        label_score = self.hidden2tag(h_n)
        label_score = self.dropfinal(label_score)
        return label_score
    
    def NLL_loss(self, label_score, label_tensor):  
        if self.num_labels > 2:
            batch_loss = self.lossF(label_score, label_tensor)
        else:
            batch_loss = self.lossF(label_score, label_tensor.float().view(-1,1))
        return batch_loss 

    def inference(self, label_score):
        if self.num_labels > 2:
            label_prob, label_pred = label_score.data.max(1)
        else:
            label_prob = F.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred
            
    def get_last_hiddens(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)

        word_embs = self.word_embs_rnn.get_all_hiddens(word_inputs, word_lengths)
        if self.use_char:
            char_embs = self.char_embs_rnn.get_last_hiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length, -1)
            
            word_embs = torch.cat([char_embs, word_embs], -1)
            
        embs_drop = self.drop(word_embs)
        hc_0 = self.initHidden(word_batch)
        pack_input = pack_padded_sequence(embs_drop, word_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0) 
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)
        return h_n

    def get_last_atthiddens(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)

        word_embs = self.word_embs_rnn.get_all_atthiddens(word_inputs, word_lengths)
        if self.use_char:
            char_embs = self.char_embs_rnn.get_last_hiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length, -1)
            
            word_embs = torch.cat([char_embs, word_embs], -1)
            
        embs_drop = self.drop(word_embs)
        hc_0 = self.initHidden(word_batch)
        pack_input = pack_padded_sequence(embs_drop, word_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0) 
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)

        #(batch_size, seq_length, 1)
        att_features = F.relu(self.att_layer(rnn_out))
         #(batch_size, seq_length)
        att_features.squeeze_()
        alpha = self.softmax(att_features)
        att_embs = embs_drop*alpha.view(word_batch,seq_length,1)
        att_h = att_embs.sum(1)
        features = torch.cat([h_n, att_h], -1)    
        return features

    def get_all_hiddens(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)

        word_embs = self.word_embs_rnn.get_all_hiddens(word_inputs, word_lengths)
        if self.use_char:
            char_embs = self.char_embs_rnn.get_last_hiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length,-1)
            word_embs = torch.cat([char_embs, word_embs], -1)
        
        embs_drop = self.drop(word_embs)
        hc_0 = self.initHidden(word_batch)
        pack_input = pack_padded_sequence(embs_drop, word_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0) 
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        return rnn_out

    def get_all_atthiddens(self, word_inputs, word_lengths, char_inputs, char_lengths, char_seq_recover):
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)

        word_embs = self.word_embs_rnn.get_all_atthiddens(word_inputs, word_lengths)
        if self.use_char:
            char_embs = self.char_embs_rnn.get_last_atthiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length, -1)
            word_embs = torch.cat([char_embs, word_embs], -1)
            
        embs_drop = self.drop(word_embs)
        hc_0 = self.initHidden(word_batch)
        pack_input = pack_padded_sequence(embs_drop, word_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.lstm(pack_input, hc_0) 
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        h_n = torch.cat([hc_n[0][0,:,:], hc_n[0][1,:,:]],-1)

        #(batch_size,seq_len,1)
        att_features = F.relu(self.att_layer(rnn_out))
        #(batch_size,seq_len)
        att_features.squeeze_()
        #(batch_size,seq_len)
        alpha = self.softmax(att_features)
        att_hidden = h_n.view(word_batch,1,-1)*alpha.view(word_batch,-1,1)
        word_features = torch.cat([rnn_out,att_hidden], -1)
        return word_features
         
    def initHidden(self, batch_size):
        d = 2 if self.bidirect else 1
        h = Variable(torch.zeros(self.layers*d, batch_size, self.hidden_dim))
        c = Variable(torch.zeros(self.layers*d, batch_size, self.hidden_dim))
        if use_cuda:
            return h.cuda(), c.cuda()
        else:
            return h,c
        
    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index,:] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs
            
class BiLSTM(nn.Module):
    def __init__(self, word_HPs=None, num_labels = None, drop_final=0.5):
        super(BiLSTM, self).__init__()
        [word_size, word_dim, wd_embeddings, word_hidden_dim, word_dropout, word_layers, word_bidirect] = word_HPs
        self.num_labels = num_labels
        self.lstm = Embs(word_HPs)
        self.dropfinal = nn.Dropout(drop_final)
        if num_labels > 2:
            self.hidden2tag = nn.Linear(word_hidden_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag = nn.Linear(word_hidden_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()            

        if use_cuda:
            self.dropfinal = self.dropfinal.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.lossF = self.lossF.cuda()
    
    def forward(self, word_tensor, word_lengths):  
        word_h_n = self.lstm.get_last_hiddens(word_tensor, word_lengths)
        label_score = self.hidden2tag(word_h_n)
        label_score = self.dropfinal(label_score)
        return label_score

    def NLL_loss(self, label_score, label_tensor):  
        if self.num_labels > 2:
            batch_loss = self.lossF(label_score, label_tensor)
        else:
            batch_loss = self.lossF(label_score, label_tensor.float().view(-1,1))
        return batch_loss  

    def inference(self, label_score):
        if self.num_labels > 2:
            label_prob, label_pred = label_score.data.max(1)
        else:
            label_prob = F.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred
            
from crf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, word_HPs, char_HPs, num_labels=None, drop_final=0.5):
        super(BiLSTM_CRF, self).__init__()
        [word_size, word_dim, word_pre_embs, word_hidden_dim, word_dropout, word_layers, word_bidirect] = word_HPs
        if char_HPs:
            [char_size, char_dim, char_pred_embs, char_hidden_dim, char_dropout, char_layers, char_bidirect] = char_HPs
       
        self.lstm = CW_bisltm(word_HPs, char_HPs, num_labels)
        # add two more labels for CRF
        self.crf = CRF(num_labels+2, use_cuda)
        ## add two more labels to learn hidden features for start and end transition 
        self.hidden2tag = nn.Linear(word_hidden_dim, num_labels+2)
        self.dropfinal = nn.Dropout(drop_final)
        if use_cuda:
            self.hidden2tag = self.hidden2tag.cuda()
            self.dropfinal = self.dropfinal.cuda()


    def NLL_loss(self, label_score, mask_tensor, label_tensor):
        batch_loss = self.crf.neg_log_likelihood_loss(label_score, mask_tensor, label_tensor)
        return batch_loss

    def inference(self, label_score, mask_tensor):
        label_prob, label_pred = self.crf._viterbi_decode(label_score, mask_tensor)
        return label_prob, label_pred

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        # (batch_size,sequence_len,hidden_dim)
        rnn_out = self.lstm.get_all_hiddens(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # (batch_size,sequence_len,num_labels+2)
        label_score = self.hidden2tag(rnn_out)
        label_score = self.dropfinal(label_score)
        return label_score

class attBiLSTM_CRF(nn.Module):
    def __init__(self, word_HPs, char_HPs, num_labels=None, drop_final=0.5):
        super(attBiLSTM_CRF, self).__init__()
        [word_size, word_dim, word_pre_embs, word_hidden_dim, word_dropout, word_layers, word_bidirect] = word_HPs
        if char_HPs:
            [char_size, char_dim, char_pred_embs, char_hidden_dim, char_dropout, char_layers, char_bidirect] = char_HPs
       
        self.lstm = CW_bisltm(word_HPs, char_HPs, num_labels, att=True)
        # add two more labels for CRF
        self.crf = CRF(num_labels+2, use_cuda)
        ## add two more labels to learn hidden features for start and end transition 
        self.hidden2tag = nn.Linear(2*word_hidden_dim, num_labels+2)
        self.dropfinal = nn.Dropout(drop_final)
        if use_cuda:
            self.hidden2tag = self.hidden2tag.cuda()
            self.dropfinal = self.dropfinal.cuda()


    def NLL_loss(self, label_score, mask_tensor, label_tensor):
        batch_loss = self.crf.neg_log_likelihood_loss(label_score, mask_tensor, label_tensor)
        return batch_loss

    def inference(self, label_score, mask_tensor):
        label_prob, label_pred = self.crf._viterbi_decode(label_score, mask_tensor)
        return label_prob, label_pred

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        # (batch_size,sequence_len,hidden_dim)
        rnn_out = self.lstm.get_all_atthiddens(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # (batch_size,sequence_len,num_labels+2)
        label_score = self.hidden2tag(rnn_out)
        label_score = self.dropfinal(label_score)
        return label_score


class deepBiLSTM_CRF(nn.Module):
    def __init__(self, word_HPs, char_HPs, num_labels=None, drop_final=0.5):
        super(deepBiLSTM_CRF, self).__init__()
        [word_size, word_dim, word_pre_embs, word_hidden_dim, word_dropout, word_layers, word_bidirect] = word_HPs
        if char_HPs:
            [char_size, char_dim, char_pred_embs, char_hidden_dim, char_dropout, char_layers, char_bidirect] = char_HPs
       
        self.lstm = Deep_bisltm(word_HPs, char_HPs, num_labels, att=True)
        # add two more labels for CRF
        self.crf = CRF(num_labels+2, use_cuda)
        ## add two more labels to learn hidden features for start and end transition 
        self.hidden2tag = nn.Linear(2*word_hidden_dim, num_labels+2)
        self.dropfinal = nn.Dropout(drop_final)
        if use_cuda:
            self.hidden2tag = self.hidden2tag.cuda()
            self.dropfinal = self.dropfinal.cuda()


    def NLL_loss(self, label_score, mask_tensor, label_tensor):
        batch_loss = self.crf.neg_log_likelihood_loss(label_score, mask_tensor, label_tensor)
        return batch_loss

    def inference(self, label_score, mask_tensor):
        label_prob, label_pred = self.crf._viterbi_decode(label_score, mask_tensor)
        return label_prob, label_pred

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        # (batch_size,sequence_len,hidden_dim)
        rnn_out = self.lstm.get_all_atthiddens(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # (batch_size,sequence_len,num_labels+2)
        label_score = self.hidden2tag(rnn_out)
        label_score = self.dropfinal(label_score)
        return label_score
    
if __name__ == "__main__":
    from data_utils import Data2tensor, Vocab, seqPAD, CoNLLDataset
    train_file='/media/data/NER/conll03/conll03/train.bmes'
    dev_file='/media/data/NER/conll03/conll03/dev.bmes'
    test_file='/media/data/NER/conll03/conll03/test.bmes'
    vocab = Vocab(cutoff=1, wl_th=None, cl_th=None, w_lower=False, w_norm=False, c_lower=False, c_norm=False)
    vocab.build([train_file, dev_file, test_file])
    
    
    word2idx = vocab.wd2idx(vocab_words=vocab.w2i, vocab_chars=vocab.c2i, allow_unk=True, start_end=True)
    tag2idx = vocab.tag2idx(vocab_tags=vocab.l2i, start_end=True)
    train_data = CoNLLDataset(train_file, word2idx=word2idx, tag2idx=tag2idx)
    train_iters = Vocab.minibatches(train_data, batch_size=10)
    data=[]
    label_ids = []
    for words, labels in train_iters:
        char_ids, word_ids = zip(*words)
        data.append(words)
        word_ids, sequence_lengths = seqPAD.pad_sequences(word_ids, pad_tok=0, wthres=1024, cthres=32)
        char_ids, word_lengths = seqPAD.pad_sequences(char_ids, pad_tok=0, nlevels=2, wthres=1024, cthres=32)
        label_ids, label_lengths = seqPAD.pad_sequences(labels, pad_tok=0, wthres=1024, cthres=32)
    
    w_tensor=Data2tensor.idx2tensor(word_ids)
    c_tensor=Data2tensor.idx2tensor(char_ids)
    y_tensor=Data2tensor.idx2tensor(label_ids)
    
    data_tensor = Data2tensor.sort_tensors(label_ids, word_ids, sequence_lengths, char_ids, word_lengths, volatile_flag=False)
    label_tensor, word_tensor, sequence_lengths, word_seq_recover, char_tensor, word_lengths, char_seq_recover = data_tensor
    
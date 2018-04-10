#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import random
import argparse
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.metric import get_ner_fmeasure, recover_label
from utils.core_nns import attBiLSTM_CRF as fNN
from utils.other_utils import Progbar, Timer, SaveloadHP
from utils.data_utils import Vocab, Data2tensor, Embeddings, CoNLLDataset, seqPAD, NERchunks

use_cuda = torch.cuda.is_available()
seed_num = 12345
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
    
class Classifier(object):
    def __init__(self, args=None):
        self.args = args  
        word_layers = 1
        word_bidirect = True
        word_HPs = [len(self.args.vocab.w2i), self.args.word_dim, self.args.word_pred_embs, self.args.word_hidden_dim, self.args.dropout, word_layers, word_bidirect]
        char_HPs = [len(self.args.vocab.c2i), self.args.char_dim, None, self.args.char_hidden_dim, self.args.dropout, word_layers, word_bidirect]
        
        
        self.model = fNN(word_HPs=word_HPs, char_HPs=char_HPs, num_labels=len(self.args.vocab.l2i), drop_final=args.drop_final)

        if args.optimizer.lower() == "adamax":
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        
        self.word2idx = self.args.vocab.wd2idx(vocab_words=self.args.vocab.w2i, vocab_chars=self.args.vocab.c2i, allow_unk=True, start_end=self.args.start_end)
        self.tag2idx = self.args.vocab.tag2idx(vocab_tags=self.args.vocab.l2i, start_end=self.args.start_end)
        
    def evaluate_batch(self, eva_data):
        wl = self.args.vocab.wl
        cl = self.args.vocab.cl    
        
        batch_size = self.args.batch_size  
         ## set model in eval model
        self.model.eval()
        correct_preds = 0.
        total_preds = 0.
        total_correct = 0.
        accs = []
        pred_results=[]
        gold_results=[]
        for i,(words, label_ids) in enumerate(self.args.vocab.minibatches(eva_data, batch_size=batch_size)):
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = seqPAD.pad_sequences(word_ids, pad_tok=0, wthres=wl, cthres=cl)
            char_ids, word_lengths = seqPAD.pad_sequences(char_ids, pad_tok=0, nlevels=2, wthres=wl, cthres=cl)
            label_ids, _ = seqPAD.pad_sequences(label_ids, pad_tok=0, wthres=wl, cthres=cl)
    
            data_tensors = Data2tensor.sort_tensors(label_ids, word_ids,sequence_lengths, char_ids,word_lengths, volatile_flag=True)
            label_tensor, word_tensor, sequence_lengths, word_seq_recover, char_tensor, word_lengths, char_seq_recover = data_tensors
            mask_tensor = word_tensor > 0
            
            label_score = self.model(word_tensor, sequence_lengths, char_tensor, word_lengths, char_seq_recover)

            label_prob, label_pred = self.model.inference(label_score, mask_tensor)
            
#            pred_label, gold_label = recover_label(label_pred, label_tensor, mask_tensor, self.args.vocab.l2i, word_seq_recover)
#            pred_results += pred_label
#            gold_results += gold_label
#        acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)
            
            label_pred = label_pred.cpu().data.numpy()
            label_tensor = label_tensor.cpu().data.numpy()
            sequence_lengths = sequence_lengths.cpu().data.numpy()
            
            for lab, lab_pred, length in zip(label_tensor, label_pred, sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]                
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]
    
                lab_chunks      = set(NERchunks.get_chunks(lab, self.args.vocab.l2i))
                lab_pred_chunks = set(NERchunks.get_chunks(lab_pred, self.args.vocab.l2i))
    
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
                
        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return acc, f

    def train_batch(self,train_data):
        wl = self.args.vocab.wl
        cl = self.args.vocab.cl 
        clip_rate = self.args.clip
        
        batch_size = self.args.batch_size
        num_train = len(train_data)
        total_batch = num_train//batch_size+1
        prog = Progbar(target=total_batch)
        ## set model in train model
        self.model.train()
        train_loss = []
        for i,(words, label_ids) in enumerate(self.args.vocab.minibatches(train_data, batch_size=batch_size)):
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = seqPAD.pad_sequences(word_ids, pad_tok=0, wthres=wl, cthres=cl)
            char_ids, word_lengths = seqPAD.pad_sequences(char_ids, pad_tok=0, nlevels=2, wthres=wl, cthres=cl)
            label_ids, _ = seqPAD.pad_sequences(label_ids, pad_tok=0, wthres=wl, cthres=cl)

            data_tensors = Data2tensor.sort_tensors(label_ids, word_ids,sequence_lengths, char_ids,word_lengths)
            label_tensor, word_tensor, sequence_lengths, word_seq_recover, char_tensor, word_lengths, char_seq_recover = data_tensors
            mask_tensor = word_tensor > 0
            
            label_score = self.model(word_tensor, sequence_lengths, char_tensor, word_lengths, char_seq_recover)

            batch_loss = self.model.NLL_loss(label_score, mask_tensor, label_tensor)

            train_loss.append(batch_loss.data.tolist()[0])
            self.model.zero_grad()
            batch_loss.backward()
            if clip_rate>0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), clip_rate)
            self.optimizer.step()
            
            prog.update(i + 1, [("Train loss", batch_loss.data.tolist()[0])])
        return np.mean(train_loss)

    def lr_decay(self, epoch):
        lr = self.args.lr/(1+self.args.decay_rate*epoch)
        print("INFO: - Learning rate is setted as: %f"%lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):            
        train_data = CoNLLDataset(self.args.train_file, word2idx=self.word2idx, tag2idx=self.tag2idx)
        dev_data = CoNLLDataset(self.args.dev_file, word2idx=self.word2idx, tag2idx=self.tag2idx)
        test_data = CoNLLDataset(self.args.test_file, word2idx=self.word2idx, tag2idx=self.tag2idx)
    
        max_epochs = self.args.max_epochs
        best_dev = -1
        nepoch_no_imprv = 0
        epoch_start = time.time()
        for epoch in xrange(max_epochs):
            self.lr_decay(epoch)
            print("Epoch: %s/%s" %(epoch,max_epochs))
            train_loss = self.train_batch(train_data)
            # evaluate on developing data
            acc_dev, f1_dev = self.evaluate_batch(dev_data)
            dev_metric = f1_dev
            if dev_metric > best_dev:
                nepoch_no_imprv = 0
                model_name = self.args.model_dir+ self.args.model_name
                torch.save(self.model.state_dict(), model_name)
                best_dev = dev_metric 
                print("UPDATES: - New improvement")
#                test_metric = self.evaluate_batch(test_data)
                print("         - Train loss: %4f"%train_loss)
                print("         - Dev acc: %2f"%(100*best_dev))
#                print("         - Test acc: %2f"%(100*test_metric))                
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.args.patience:
                    print("\nSUMMARY: - Early stopping after %d epochs without improvements"%(nepoch_no_imprv))
                    acc_test, f1_test = self.evaluate_batch(test_data)
                    test_metric = f1_test
                    print("         - Train loss: %4f"%train_loss)
                    print("         - Dev acc: %2f"%(100*best_dev))
                    print("         - Test acc: %2f"%(100*test_metric))
                    return

            epoch_finish = Timer.timeEst(epoch_start,(epoch+1)/max_epochs)
            print("\nINFO: - Trained time(Remained time): %s; - Dev acc: %.2f"%(epoch_finish,100*dev_metric))
            
            gc.collect()
        print("\nSUMMARY: - Completed %d epoches"%(max_epochs))
        acc_test, f1_test = self.evaluate_batch(test_data)
        test_metric = f1_test
        print("         - Train loss: %4f"%train_loss)
        print("         - Dev acc: %2f"%(100*best_dev))
        print("         - Test acc: %2f"%(100*test_metric))
        
        return 

    def predict(self, sent):
        numtags = len(self.args.vocab.l2i)
        wl = self.args.vocab.wl
        cl = self.args.vocab.cl            
         ## set model in eval model
        self.model.eval()
        
        words=self.args.vocab.process_seq(sent)  
        fake_label = [[0]*len(words)]
        
        words = [self.word2idx(word) for word in words]
        char_ids, word_ids = zip(*words)
        word_ids, sequence_lengths = seqPAD.pad_sequences([word_ids], pad_tok=0, wthres=wl, cthres=cl)
        char_ids, word_lengths = seqPAD.pad_sequences([char_ids], pad_tok=0, nlevels=2, wthres=wl, cthres=cl)
    
        data_tensors = Data2tensor.sort_tensors(fake_label, word_ids,sequence_lengths, char_ids,word_lengths, volatile_flag=True)    
        fake_label_tensor, word_tensor, sequence_lengths, word_seq_recover, char_tensor, word_lengths, char_seq_recover = data_tensors
        label_score = self.model(word_tensor, sequence_lengths, char_tensor, word_lengths, char_seq_recover)
        
        if numtags > 2:
            label_prob, label_pred = label_score.data.max(1)
        else:
            label_prob = F.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred 


def build_data(args):    
    print("Building dataset...")
    if not os.path.exists(args.model_dir): 
        os.mkdir(args.model_dir)
    vocab = Vocab(wl_th=args.word_thres, cl_th=args.char_thres, cutoff=args.cutoff,
                  w_lower=args.w_lower, c_lower=args.c_lower, w_norm=args.w_norm, c_norm=args.c_norm)
    vocab.build([args.train_file,args.dev_file,args.test_file])
    args.vocab = vocab    
    if args.pre_trained:
        scale = np.sqrt(3.0 / args.word_dim)
        args.word_pred_embs = Embeddings.get_W(args.emb_file, args.word_dim,vocab.w2i, scale)
    else:
        args.word_pred_embs = None  
    SaveloadHP.save(args, args.model_args)
    return args

if __name__ == '__main__':
    """
    CUDA_VISIBLE_DEVICES=1 screen python model.py --emb_file /media/data/embeddings/glove/glove.6B.100d.txt --word_dim 100 --word_hidden_dim 200 --char_dim 50 --char_hidden_dim 100 --dropout 0.5 --drop_final 0.5 --lr 0.015 --batch_size 10 --model_name a1.bilstm.m --model_args ./results/a1.bilstm.args.pkl --c_norm --w_norm
    """
    argparser = argparse.ArgumentParser(sys.argv[0])
    
    argparser.add_argument('--train_file', help='Trained file', default="/media/data/NER/conll03/conll03/train.bmes", type=str)
    
    argparser.add_argument('--dev_file', help='Developed file', default="/media/data/NER/conll03/conll03/dev.bmes", type=str)
    
    argparser.add_argument('--test_file', help='Tested file', default="/media/data/NER/conll03/conll03/test.bmes", type=str)
                        
    argparser.add_argument("--cutoff", type = int, default = 1, help = "prune words ocurring <= cutoff")  
    
    argparser.add_argument("--char_thres", type = int, default = None, help = "char threshold")
    
    argparser.add_argument("--word_thres", type = int, default = None, help = "word threshold")
    
    argparser.add_argument("--c_lower", action='store_true', default = False, help = "lowercase characters")
    
    argparser.add_argument("--w_lower", action='store_true', default = False, help = "lowercase words")

    argparser.add_argument("--c_norm", action='store_true', default = False, help = "number-norm characters")
    
    argparser.add_argument("--w_norm", action='store_true', default = False, help = "number-norm words")

    argparser.add_argument("--start_end", action='store_true', default = False, help = "start-end paddings")
    
    argparser.add_argument("--emb_file", type = str, default = "/media/data/embeddings/glove/glove.6B.50d.txt", help = "embedding file")
    
    argparser.add_argument("--pre_trained", type = int, default = 1, help = "Use pre-trained embedding or not")
    
    argparser.add_argument("--word_dim", type = int, default = 50, help = "word embedding size")
        
    argparser.add_argument("--word_hidden_dim", type = int, default = 100, help = "LSTM layers")

    argparser.add_argument("--char_dim", type = int, default = 50, help = "char embedding size")
        
    argparser.add_argument("--char_hidden_dim", type = int, default = 100, help = "char LSTM layers")
                
    argparser.add_argument("--dropout", type = float, default = 0.5, help = "dropout probability")
    
    argparser.add_argument("--drop_final", type = float, default = 0.5, help = "final dropout probability")
    
    argparser.add_argument("--patience", type = int, default = 64, help = "early stopping")
            
    argparser.add_argument("--optimizer", type = str, default = "SGD", help = "learning method (adagrad, sgd, ...)")
    
    argparser.add_argument("--lr", type = float, default = 0.015, help = "learning rate") 
    
    argparser.add_argument("--decay_rate", type = float, default = 0.05, help = "decay learning rate")
        
    argparser.add_argument("--max_epochs", type = int, default = 512, help = "maximum # of epochs")
    
    argparser.add_argument("--batch_size", type = int, default = 10, help = "mini-batch size")  
    
    argparser.add_argument('--clip', help='Clipping value', default=5, type=int)
    
    argparser.add_argument('--model_dir', help='Model dir', default="./results/", type=str)
    
    argparser.add_argument('--model_name', help='Model dir', default="bilstm.m", type=str)

    argparser.add_argument('--model_args', help='Model dir', default="./results/bilstm.args.pklz", type=str)
    
    args = argparser.parse_args()
    
    args = build_data(args)
    
    classifier = Classifier(args)    

    classifier.train() 
    
    

    
    






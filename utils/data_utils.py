#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 17:31:21 2018

@author: dtvo
"""
from __future__ import print_function
from __future__ import division

import csv
import sys
import torch
import itertools
import numpy as np
from torch.autograd import Variable
from collections import Counter
from other_utils import Encoder
    
PADc = u"<PADc>"
UNKc = u"<UNKc>"
SOc = u"<sc>"
EOc = u"</sc>"
PADw = u"<PADw>"
UNKw = u"<UNKw>"
SOw = u"<sw>"
EOw = u"</sw>"
NONE = u"O"

class Vocab(object):
    def __init__(self, wl_th=None, cl_th=None, cutoff=1, w_lower=False, c_lower=False, w_norm=False, c_norm=False):
        self.w2i = {}
        self.c2i = {}
        self.l2i = {}
        self.wl = wl_th
        self.cl = cl_th
        self.w_lower = w_lower
        self.c_lower = c_lower
        self.w_norm = w_norm
        self.c_norm = c_norm
        self.cutoff = cutoff
                        
    def build(self, files, cutoff=1, firstline=False):
        lcnt = Counter()
        wcnt = Counter()
        ccnt = Counter()
        print("Extracting vocabulary:")
        wl=0
        cl=0
        for fname in files:
            raw=CoNLLDataset(fname)  
            for seqs, labels in raw:
                w_seqs = Vocab.process_seq(seqs, self.w_lower, self.w_norm)
                wcnt.update(w_seqs)
                wl=max(wl,len(w_seqs))
                
                c_seqs = Vocab.process_seq(seqs, self.c_lower, self.c_norm)
                ccnt.update(u''.join(c_seqs))
                cl=max(cl,max([len(wd) for wd in c_seqs]))
                
                lcnt.update(labels)
                
        print("\t%d total words, %d total characters, %d total labels" % (sum(wcnt.values()),sum(ccnt.values()),sum(lcnt.values())))
        wlst=[x for x, y in wcnt.iteritems() if y >= cutoff]
        wlst = [PADw, UNKw, SOw, EOw] + wlst
        wvocab = dict([ (y,x) for x,y in enumerate(wlst) ])
        
        clst=[x for x, y in ccnt.iteritems() if y >= cutoff]
        clst = [PADc, UNKc, SOc, EOc] + clst
        cvocab = dict([ (y,x) for x,y in enumerate(clst) ])
        lvocab = dict([ (y,x) for x,y in enumerate(lcnt.keys()) ])
        print("\t%d unique words, %d unique characters, %d unique labels" % (len(wcnt),len(ccnt), len(lcnt)))
        print("\t%d unique words, %d unique characters appearing at least %d times" % (len(wvocab)-4,len(cvocab)-4, cutoff))
        self.w2i = wvocab
        self.c2i = cvocab
        self.l2i = lvocab
        if self.wl is None:
            self.wl = wl
        else:
            self.wl = min(wl, self.wl)
        
        if self.cl is None:
            self.cl = cl
        else:
            self.cl = min(cl, self.cl)
        
    @staticmethod    
    def process_seq(seq, lowercase=False, numnorm=False):
        if isinstance(seq, list):
            seq = u' '.join(seq)
        if lowercase:
            seq = seq.lower()
        if numnorm:
            seq = Vocab.norm_seq(seq)
        seq = seq.split()
        return seq
    
    @staticmethod 
    def norm_seq(seq):
        seq=u''.join([ u'0' if ch.isdigit() else ch for ch in seq])
        return seq

    def wd2idx(self, vocab_words=None, vocab_chars=None, allow_unk=True, start_end=False):
        '''
        Return a function to convert tag2idx or word/char2idx
        '''
        def f(seqs): 
            if vocab_chars is not None:
                c_seqs = self.process_seq(seqs, self.c_lower, self.c_norm)
                char_ids = []
                for tok in c_seqs: 
                    char_id = []
                    for char in tok:
                        # ignore chars out of vocabulary
                        if char in vocab_chars:
                            char_id += [vocab_chars[char]]
                        else:
                            if allow_unk:
                                 char_id += [vocab_chars[UNKc]]
                            else:
                                raise Exception("Unknow key is not allowed. Check that "\
                                                "your vocab (tags?) is correct") 
                    if start_end:
                        char_id = [vocab_chars[SOc]] + char_id + [vocab_chars[EOc]]
                    char_ids += [char_id]
                    
                if start_end:
                    char_ids = [[vocab_chars[SOc], vocab_chars[EOc]]] + char_ids + [[vocab_chars[SOc], vocab_chars[EOc]]]
            # 3. get id of word
            if vocab_words is not None:
                w_seqs = self.process_seq(seqs,self.w_lower, self.w_norm)
                word_ids = []
                for tok in w_seqs:
                    if tok in vocab_words:
                        word_ids += [vocab_words[tok]]
                    else:
                        if allow_unk:
                            word_ids += [vocab_words[UNKw]]
                        else:
                            raise Exception("Unknow key is not allowed. Check that "\
                                            "your vocab (tags?) is correct")
                if start_end:
                    word_ids = [vocab_words[SOw]] + word_ids + [vocab_words[EOw]]
            # 4. return tuple char ids, word id
            if vocab_chars is not None:
                return zip(char_ids, word_ids)
            else:
                return word_ids
        return f 

    @staticmethod     
    def tag2idx(vocab_tags=None, start_end=False):
        def f(seqs):
            tag_ids = []
            if vocab_tags is not None:
                for tag in seqs:
                    if tag in vocab_tags:
                        tag_ids += [vocab_tags[tag]]
                    else:
                        raise Exception("Check that your tags? is correct")  
            if start_end:
                tag_ids = [vocab_tags[NONE]] + tag_ids + [vocab_tags[NONE]]
            return tag_ids
        
        return f

    @staticmethod
    def minibatches(data, batch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)
    
        Yields:
            list of tuples
    
        """
        x_batch, y_batch = [], []
        for (x, y) in data:
            if len(x_batch) == batch_size:
                # yield a tuple of list ([wd_ch_i], [label_i])
                yield x_batch, y_batch
                x_batch, y_batch = [], []
            # if use char, decompose x into wd_ch_i=[([char_ids],...[char_ids]),(word_ids)]
            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]
    
        if len(x_batch) != 0:
            yield x_batch, y_batch

class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, word2idx=None, tag2idx=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:    
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break

                        if self.word2idx is not None:
                            words = self.word2idx(words)
                        if self.tag2idx is not None:
                            tags = self.tag2idx(tags)

                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    word = Encoder.str2uni(word)
                    tag = Encoder.str2uni(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
                              
class seqPAD:
    @staticmethod
    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
    
        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []
    
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded +=  [seq_]
            sequence_length += [min(len(seq), max_length)]
    
        return sequence_padded, sequence_length

    @staticmethod
    def pad_sequences(sequences, pad_tok, nlevels=1, wthres=1024, cthres=32):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
            nlevels: "depth" of padding, for the case where we have characters ids
    
        Returns:
            a list of list where each sublist has same length
    
        """
        if nlevels == 1:
            max_length = max(map(lambda x : len(x), sequences))
            max_length = min(wthres,max_length)
            sequence_padded, sequence_length = seqPAD._pad_sequences(sequences, pad_tok, max_length)
    
        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            max_length_word = min (cthres,max_length_word)
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # pad the character-level first to make the word length being the same
                sp, sl = seqPAD._pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]
            # pad the word-level to make the sequence length being the same
            max_length_sentence = max(map(lambda x : len(x), sequences))
            max_length_sentence = min(wthres,max_length_sentence)
            sequence_padded, _ = seqPAD._pad_sequences(sequence_padded, [pad_tok]*max_length_word, max_length_sentence)
            sequence_length, _ = seqPAD._pad_sequences(sequence_length, 1, max_length_sentence)
    
        return sequence_padded, sequence_length
    
    @staticmethod
    def pad_labels(y, nb_classes=None):
        '''Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy.
        '''
        if not nb_classes:
            nb_classes = max(y)+1
        Y=[[0]*nb_classes for i in xrange(len(y))]
        for i in range(len(y)):
            Y[i][y[i]] = 1
        return Y 
    
class Embeddings:
    @staticmethod
    def load_embs(fname):
        embs=dict()
        s=0
        V=0
        with open(fname,'rb') as f:
            for line in f: 
                p=line.strip().split()
                if len(p)==2:
                    V=int(p[0]) ## Vocabulary
                    s=int(p[1]) ## embeddings size
                else:
#                    assert len(p)== s+1
                    w=p[0]
                    e=[float(i) for i in p[1:]]
                    embs[w]=np.array(e,dtype="float32")
#        assert len(embs)==V
        return embs 
    
    @staticmethod
    def get_W(emb_file, wsize, vocabx, scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        print("Extracting pretrained embeddings:")
        word_vecs =Embeddings.load_embs(emb_file)
        print(('\t%d pre-trained word embeddings')%(len(word_vecs)))
        print('Mapping to vocabulary:')
        unk=0
        part=0
        W = np.zeros(shape=(len(vocabx), wsize),dtype="float32")            
        for word,idx in vocabx.iteritems():
            if word_vecs.get(word) is not None:
                W[idx]=word_vecs.get(word)
            else:
                if word_vecs.get(word.lower()) is not None:
                    W[idx]=word_vecs.get(word.lower())
                    part +=1
                else:
                    unk+=1
                    rvector=np.asarray(np.random.uniform(-scale,scale,(1,wsize)),dtype="float32")
                    W[idx]=rvector
        print('\t%d randomly unknown word vectors;'%unk)
        print('\t%d partially pre-trained word vectors.'%part)
        print('\t%d pre-trained word vectors.'%(len(vocabx)-unk-part))
        return W

    @staticmethod
    def init_W(wsize, vocabx, scale=0.25):
        """
        Randomly initial word vectors between [-scale, scale]
        """
        W = np.zeros(shape=(len(vocabx), wsize),dtype="float32")            
        for word,idx in vocabx.iteritems():
            if idx ==0:
                continue
            rvector=np.asarray(np.random.uniform(-scale,scale,(1,wsize)),dtype="float32")
            W[idx]=rvector
        return W

class Data2tensor:
    @staticmethod
    def zerotensor(shape, ttype="long", volatile_flag=False):
        var = Variable(torch.zeros(shape,volatile =  volatile_flag)).long()
        if ttype=="byte":
            var = var.byte()
        if torch.cuda.is_available():
            return var.cuda()
        else:
            return var

    @staticmethod
    def idx2tensor(indexes, volatile_flag=False):
        result = Variable(torch.LongTensor(indexes), volatile=volatile_flag)
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result

    @staticmethod
    def sort_tensors(label_ids, word_ids, sequence_lengths, char_ids, word_lengths, volatile_flag=False):        
        label_tensor=Data2tensor.idx2tensor(label_ids, volatile_flag)
        word_tensor=Data2tensor.idx2tensor(word_ids, volatile_flag)
        sequence_lengths = Data2tensor.idx2tensor(sequence_lengths, volatile_flag)
        
        sequence_lengths, word_perm_idx = sequence_lengths.sort(0, descending=True)
        
        word_tensor = word_tensor[word_perm_idx]
        label_tensor = label_tensor[word_perm_idx]
        
    
        char_tensor=Data2tensor.idx2tensor(char_ids, volatile_flag)
        word_lengths = Data2tensor.idx2tensor(word_lengths, volatile_flag)
        
        batch_size = len(word_ids)        
        max_seq_len = sequence_lengths.max()
        char_tensor = char_tensor[word_perm_idx].view(batch_size*max_seq_len.data[0],-1)
        word_lengths = word_lengths[word_perm_idx].view(batch_size*max_seq_len.data[0],)
       
        word_lengths, char_perm_idx = word_lengths.sort(0, descending=True)
        
        char_tensor = char_tensor[char_perm_idx]
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        return label_tensor, word_tensor, sequence_lengths, word_seq_recover, char_tensor, word_lengths, char_seq_recover

class NERchunks:
    @staticmethod
    def get_chunk_type(tok, idx_to_tag):
        """
        Args:
            tok: id of token, ex 4
            idx_to_tag: dictionary {4: "B-PER", ...}
    
        Returns:
            tuple: "B", "PER"
    
        """
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type

    @staticmethod
    def get_chunks(seq, tags):
        """Given a sequence of tags, group entities and their position
    
        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4
    
        Returns:
            list of (chunk_type, chunk_start, chunk_end)
    
        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]
    
        """
        default = tags[NONE]
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
    
            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = NERchunks.get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B" or tok_chunk_class == "S":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
    
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)
    
        return chunks
    
def batchify_with_label(labels, words, chars, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(labels)
    
    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = Variable(torch.zeros((batch_size, )),volatile =  volatile_flag).long()
    mask = Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [list(chars[idx]) + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [map(len, pad_char) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if torch.cuda.is_available():
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
#        char_seq_lengths = char_seq_lengths.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask
          
if __name__ == "__main__":
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
        data.append(words)
        chars, words = zip(*words)
        word_ids, sequence_lengths = seqPAD.pad_sequences(words, pad_tok=0, wthres=1024, cthres=32)
        char_ids, word_lengths = seqPAD.pad_sequences(chars, pad_tok=0, nlevels=2, wthres=1024, cthres=32)
        label_ids, label_lengths = seqPAD.pad_sequences(labels, pad_tok=0, wthres=1024, cthres=32)
    
    w_tensor=Data2tensor.idx2tensor(word_ids)
    c_tensor=Data2tensor.idx2tensor(char_ids)
    y_tensor=Data2tensor.idx2tensor(label_ids)
    
    data_tensor = Data2tensor.sort_tensors(label_ids, word_ids, sequence_lengths, char_ids, word_lengths, volatile_flag=False)
    label_tensor, word_tensor, sequence_lengths, word_seq_recover, char_tensor, word_lengths, char_seq_recover = data_tensor
    mask_tensor = word_tensor.gt(0)
    
    tensors = batchify_with_label(labels, words, chars)
    word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask = tensors

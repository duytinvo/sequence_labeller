# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:12:27 2016

@author: duytinvo
"""
from __future__ import print_function
from collections import Counter

class Encoder:
    @staticmethod
    def str2uni(text, encoding='utf8', errors='strict'):
        """Convert `text` to unicode.
    
        Parameters
        ----------
        text : str
            Input text.
        errors : str, optional
            Error handling behaviour, used as parameter for `unicode` function (python2 only).
        encoding : str, optional
            Encoding of `text` for `unicode` function (python2 only).
    
        Returns
        -------
        str
            Unicode version of `text`.
    
        """
        if isinstance(text, unicode):
            return text
        return unicode(text, encoding, errors=errors)

    @staticmethod
    def uni2str(text, errors='strict', encoding='utf8'):
        """Convert utf8 `text` to bytestring.
    
        Parameters
        ----------
        text : str
            Input text.
        errors : str, optional
            Error handling behaviour, used as parameter for `unicode` function (python2 only).
        encoding : str, optional
            Encoding of `text` for `unicode` function (python2 only).
    
        Returns
        -------
        str
            Bytestring in utf8.
    
        """
    
        if isinstance(text, unicode):
            return text.encode('utf8')
        # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
        return unicode(text, encoding, errors=errors).encode('utf8')

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

                        yield words
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

def sentgen(filnames, lowercase=False, numnorm=True):
    for filename in filnames:
        sents = CoNLLDataset(filename)
        for sent in sents:
            sent = Vocab.process_seq(sent,lowercase, numnorm)
            yield sent

def wordgen(filnames, lowercase=False, numnorm=True):
    for filename in filnames:
        sents = CoNLLDataset(filename)
        for sent in sents:
            sent = Vocab.process_seq(sent,lowercase, numnorm)
            words = []
            for word in sent:
                words += list(word)
            yield words
        
if __name__ == "__main__":
    train_file='/media/data/NER/conll03/conll03/train.bmes'
    dev_file='/media/data/NER/conll03/conll03/dev.bmes'
    test_file='/media/data/NER/conll03/conll03/test.bmes'
    data=[]
    sents = wordgen(train_file)
    for sent in sents:
        data.append(sent)

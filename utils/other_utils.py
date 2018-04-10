#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:32:36 2018

@author: dtvo
"""
from __future__ import print_function
from __future__ import division
import time
import gzip
import sys
import logging
import numpy as np
import cPickle as pickle
import math
    
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
    
class RWfile:
    @staticmethod
    def write_vocab(tok2idx, filename):
        """Writes a vocab to a file
    
        Writes one word per line.
    
        Args:
            vocab: iterable that yields word
            filename: path to vocab file
    
        Returns:
            write a word per line
    
        """
        print("Writing tokens into %s file: "%filename)
        with open(filename, "w") as f:
            for (word,idx) in tok2idx.iteritems():
                if idx != len(tok2idx) - 1:
                    f.write("{} {}\n".format(word, idx))
                else:
                    f.write("{} {}".format(word, idx))
        print("\t- Done: {} tokens.".format(len(tok2idx)))

    @staticmethod
    def load_vocab(filename):
        """Loads vocab from a file
    
        Args:
            filename: (string) the format of the file must be one word per line.
    
        Returns:
            d: dict[word] = index
    
        """
        try:
            print("Reading %s file:"%filename)
            d = dict()
            with open(filename) as f:
                for line in f:
                    word,idx = line.strip().split()
                    d[word] = idx
            print("\t- Done: %d tokens."%len(d))
        except IOError:
            raise Exception("please create the file first!")
        return d
    
######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#
class Timer:
    @staticmethod
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    @staticmethod
    def timeSince(since):
        now = time.time()
        s = now - since
        return '%s' % (Timer.asMinutes(s))

    @staticmethod
    def timeEst(since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (Timer.asMinutes(s), Timer.asMinutes(rs))
    
# Save and load hyper-parameters
class SaveloadHP:
    @staticmethod            
    def save(args,argfile='./results/model_args.pklz'):
        """
        argfile='model_args.pklz'
        """
        print("Writing hyper-parameters into %s"%argfile)
        with gzip.open(argfile, "wb") as fout:
            pickle.dump(args,fout,protocol = pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def load(argfile='./results/model_args.pklz'):
        print("Reading hyper-parameters from %s"%argfile)
        with gzip.open(argfile, "rb") as fin:
            args = pickle.load(fin)
        return args
    
def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
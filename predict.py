#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
from __future__ import print_function
from __future__ import division

import torch
from model import Classifier
from utils.other_utils import SaveloadHP, Encoder
use_cuda = torch.cuda.is_available()


def interactive_shell():
    """Creates interactive shell to play with model

    Args:
        model: instance of Classification

    """
    print("""
To exit, enter 'EXIT'.
Enter a sentence like 
input> wth is it????""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip()

        if words_raw == "EXIT":
            break
        
        words_raw = Encoder.str2uni(words_raw)
        label_prob, label_pred = predict(words_raw)
        if label_pred[0] == 0:
            print("OUTPUT> Subversive \t\t PROB> %.2f"%(100*(1-label_prob.data[0])))
        else:
            print("OUTPUT> None \t\t PROB> %.2f"%(100*label_prob.data[0]))
            
def predict(sent):
    args = SaveloadHP.load()
    print("Load Model from file: %s"%(args.model_dir+args.model_name))
    classifier = Classifier(args)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    if not use_cuda:
        classifier.model.load_state_dict(torch.load(args.model_dir+args.model_name), map_location=lambda storage, loc: storage)
        # classifier.model = torch.load(args.model_dir, map_location=lambda storage, loc: storage)
    else:
        classifier.model.load_state_dict(torch.load(args.model_dir+args.model_name))
        # classifier.model = torch.load(args.model_dir)
    
    label_prob, label_pred = classifier.predict(sent)
    return label_prob, label_pred

if __name__ == '__main__':

    interactive_shell()
    

    
    






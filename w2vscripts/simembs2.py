# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:21:18 2015

@author: duytinvo
"""
import gensim
import sys
import argparse

def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    print("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> wth is it????""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break
        lists = model.wv.most_similar(words_raw[0])
        for wd,prob in lists:
            print wd, prob

def loadw2v(args):
    print('loading w2v model ...')
    model=gensim.models.word2vec.Word2Vec.load(args.mod_file)
    interactive_shell(model)


if __name__ == '__main__':
    """
    python simembs2.py --mod_file ./results/twsamples.process.model
    """
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("--mod_file", help="Saved model file.",default='./results/twsamples.process.model')
    args = parser.parse_args()
    loadw2v(args)


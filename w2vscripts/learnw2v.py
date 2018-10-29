# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:21:18 2015

@author: duytinvo
"""
from gensim.models.word2vec import Word2Vec
import sys
import argparse

        
def trainw2v(args):
    sents=textgen([args.train_file, args.dev_file, args.test_file])
    if args.mode == "skipgram":
        sg=1
    else:
        sg=0
    model = Word2Vec(min_count=args.min_count, window=args.window, size=args.size,
                     alpha=args.lr, sg=sg, hs=args.hs, workers=args.worker)
    print('building vocab ...')
    model.build_vocab(sents)

    print('training w2v model ...')
    sents=textgen([args.train_file, args.dev_file, args.test_file],)
    model.train(sents, total_examples=model.corpus_count, epochs=args.iters)
    print('writing w2v vectors ...')
    model.wv.save_word2vec_format(args.emb_file)
    print("saving model ...")
    model.save(args.mod_file)

if __name__ == '__main__':
    """
    python learnw2v.py
    """
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--train_file', help='Trained file', default="/media/data/NER/conll03/conll03/train.bmes", type=str)
    parser.add_argument('--dev_file', help='Developed file', default="/media/data/NER/conll03/conll03/dev.bmes", type=str)
    parser.add_argument('--test_file', help='Tested file', default="/media/data/NER/conll03/conll03/test.bmes", type=str)
    parser.add_argument("--emb_file", help="Embedding file.",default='./results/ner.word.vec')
    parser.add_argument("--mod_file", help="Saved model file.",default='./results/ner.word.m')
    parser.add_argument("--emb_type", help="type of embeddings",default='word')
    parser.add_argument("--min_count", help="Min Count", type=int, default=1)
    parser.add_argument("--window", help="Window Width", type=int, default=5)
    parser.add_argument("--iters", help="number of iterations", type=int, default=100)
    parser.add_argument("--size", help="Embedding Size", type=int, default=50)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.05)
    parser.add_argument("--mode", help="model type",default='skipgram')
    parser.add_argument("--hs", help="hierarchial sampling", type=int, default=1)
    parser.add_argument("--worker", help="Number of threads",type=int, default=12)
    args = parser.parse_args()
    if args.emb_type=="word":
        from utils.data_utils import sentgen as textgen
    else:
        from utils.data_utils import wordgen as textgen
    trainw2v(args)

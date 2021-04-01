#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:57:37 2020

@author: Asep Fajar Firmansyah
"""
from __future__ import unicode_literals, print_function, division
import os
import os.path as path
import sys
import argparse
import torch
from gensim.models.keyedvectors import KeyedVectors

#from streamlit import caching
import visdom
import numpy as np

from data_loader import split_data, load_emb
from train import train_iter 
from find_best_result import find_best_topk
#caching.clear_cache()

IN_DBPEDIA_DIR = os.path.join(path.dirname(os.getcwd()), 'KGSUM-GAT/data/ESBM_benchmark_v1.2', 'dbpedia_data')
IN_LMDB_DIR = os.path.join(path.dirname(os.getcwd()), 'KGSUM-GAT/data/ESBM_benchmark_v1.2', 'lmdb_data')
FILE_N = 6
TOP_K = [5, 10]
DS_NAME = ['dbpedia', 'lmdb']
DEVICE = torch.device("cpu")

def main(mode, emb_model, loss_type,  ent_emb_dim, pred_emb_dim, hidden_layers, nheads, lr, dropout, reg, weight_decay, n_epoch, save_every, word_emb_model, word_emb_calc, use_epoch, concat_model): 
    if word_emb_model == "fasttext":
        #word_emb={}
        '''
        # This way is not working properly, which has error "killed" after executed the codes
        fname = "data/wiki-news-300d-1M.vec"
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        for line in fin:
            tokens = line.rstrip().split(' ')
            word_emb[tokens[0]] = map(float, tokens[1:])
        '''    
        
        '''
        #original code used word2vec format to load fasttext
        KeyedVectors.load_word2vec_format("data/wiki-news-300d-1M.vec")
        '''
        word_emb = KeyedVectors.load_word2vec_format("data/wiki-news-300d-1M.vec")
        
    elif word_emb_model=="Glove":
        word_emb = {}
        with open("data/glove.6B/glove.6B.300d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                word_emb[word] = vector
    else:
        print("please choose the correct word embedding model")
        sys.exit()
    
    if loss_type == "BCE":
        loss_function = torch.nn.BCELoss()
    elif loss_type == "MSE":
        loss_function = torch.nn.MSELoss()
    else:
        print("please choose the correct loss fucntion")
        sys.exit()
        
    if reg==True:
        weight_decay==weight_decay
    else:
        weight_decay==0
        
    print('Hyper paramters:')  
    print("Loss function: {}".format(loss_function))
    print("Learning rate: {}",format(lr))
    print("Dropout: {}",format(dropout))
    print("Weight Decay: {} (if regularization = True)", format(weight_decay))
    print("n Epochs: {}", format(n_epoch))
    print("Regularization: {}", format(reg))
    viz = visdom.Visdom()
    if mode == "train" or mode =="find" or mode=="find-test":
        for ds_name in DS_NAME:
            if ds_name == "dbpedia":
                db_dir = IN_DBPEDIA_DIR
            elif ds_name == "lmdb":
                db_dir = IN_LMDB_DIR
            else:
                raise ValueError("The database's name must be dbpedia or lmdb")
                sys.exit()
                
            print('Data loading ...')
            
            entity2vec, pred2vec, entity2ix, pred2ix = load_emb(ds_name, emb_model)
            entity_dict = entity2vec
            pred_dict = pred2vec
            pred2ix_size = len(pred2ix)
            entity2ix_size = len(entity2ix)
            hidden_size = ent_emb_dim + pred_emb_dim
            for topk in TOP_K:
                train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, test_adjs, test_facts, test_labels = split_data(ds_name, db_dir, topk, FILE_N) 
                if mode == "train":
                    train_iter(ds_name, train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, reg, n_epoch, save_every, DEVICE, entity_dict, \
                               pred_dict, loss_function, pred2ix_size, hidden_size, pred_emb_dim, ent_emb_dim, lr, dropout, entity2ix_size, hidden_layers, nheads, \
                               word_emb, db_dir, weight_decay, word_emb_calc, topk, FILE_N, viz, concat_model)
                if mode == "find" or mode=="find-test":
                    find_best_topk(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, \
                                     DEVICE, use_epoch, db_dir, dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, FILE_N, n_epoch, mode, concat_model)
    
    #if mode == "test":
    #    train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, test_adjs, test_facts, test_labels = split_data(ds_name, db_dir, topk, FILE_N) 
    #    generate_summary(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, \
    #                             DEVICE, use_epoch, db_dir, dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, FILE_N, concat_model)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KGSUMM: Knowledge Graph SUMMarization')
    parser.add_argument("--mode", type=str, default="all", help="train, test or all")
    parser.add_argument("--emb_model", type=str, default="ComplEx", help="use ComplEx/DistMult/ConEx")
    parser.add_argument("--loss_type", type=str, default="BCE", help="use which loss type to train the model with BCE or MSE")
    parser.add_argument("--ent_emb_dim", type=int, default=300, help="the embeddiing dimension of predicate")
    parser.add_argument("--pred_emb_dim", type=int, default=300, help="the embeddiing dimension of predicate")
    parser.add_argument("--hidden_layers", type=int, default=2, help="n hidden layer")
    parser.add_argument("--nheads", type=int, default=1, help="n heads")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--dropout", type=float, default='0.0', help="use dropout parameter")
    parser.add_argument("--weight_decay", type=float, default='1e-5', help="use weight decay parameter")
    parser.add_argument("--reg", type=bool, default=False, help="use regularization or not")
    parser.add_argument("--save_every", type=int, default=1, help="save model in every n epochs")
    parser.add_argument("--n_epoch", type=int, default=50, help="train model in total n epochs")
    parser.add_argument("--word_emb_model", type=str, default="fasttext", help="use which word embedding model (fasttext/Glove)")
    parser.add_argument("--word_emb_calc", type=str, default="SUM", help="SUM/AVG")
    parser.add_argument("--use_epoch", type=int, nargs='+', help="how many epochs to train the model")
    parser.add_argument("--concat_model", type=int, default=1, help="use which concatenation model (1 or 2). In which, 1 refers to KGE + Word embedding, and 2 refers to KGE + (KGE/Word embeddings) depends on the object value")
    
    args = parser.parse_args()
    main(args.mode, args.emb_model, args.loss_type, args.ent_emb_dim, args.pred_emb_dim, args.hidden_layers, args.nheads, args.lr, args.dropout, args.reg, args.weight_decay, \
         args.n_epoch, args.save_every, args.word_emb_model, args.word_emb_calc, args.use_epoch, args.concat_model)
    
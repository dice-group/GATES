#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:57:37 2020
"""
from __future__ import unicode_literals, print_function, division
import os
import os.path as path
import sys
import argparse
import torch
from gensim.models.keyedvectors import KeyedVectors
from graphviz import Digraph

import numpy as np
import time
import math

from data_loader import split_data, load_emb
from train import train_iter
from generate_summary import generate_summary
from model import GATES
from utils import tensor_from_data

IN_DBPEDIA_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/ESBM_benchmark_v1.2', 'dbpedia_data')
IN_LMDB_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/ESBM_benchmark_v1.2', 'lmdb_data')
OUT_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES')
IN_FACES_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/FACES', 'faces_data')
FILE_N = 6
TOP_K = [5, 10]
DS_NAME = ['faces']
DEVICE = torch.device("cpu")

def asHours(s):
	m = math.floor(s / 60)
	h = math.floor(m / 60)
	s -= m * 60
	m -= h * 60
	return '%dh %dm %ds' % (h, m, s)

def _read_epochs_from_log(ds_name, topk):
    log_file_path = os.path.join(OUT_DIR, 'GATES_log.txt')
    
    key = '{}-top{}'.format(ds_name, topk)
    epoch_list = None
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(key):
                epoch_list = list(eval(line.split('\t')[1]))
    return epoch_list
def get_embeddings(word_emb_model):
    if word_emb_model == "fasttext":
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
    return word_emb

def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph
    
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


def main(mode, emb_model, loss_type,  ent_emb_dim, pred_emb_dim, hidden_layers, nheads, lr, dropout, reg, weight_decay, n_epoch, save_every, word_emb_model, word_emb_calc, use_epoch, concat_model, weighted_edges_method): 
            
    word_emb = get_embeddings(word_emb_model)
    
    if loss_type == "BCE":
        loss_function = torch.nn.BCELoss()
    elif loss_type == "MSE":
        loss_function = torch.nn.MSELoss()
    elif loss_type == "NLLL":
        loss_function = torch.nn.NLLLoss()
    elif loss_type=="CE":
        loss_function=torch.nn.CrossEntropyLoss()
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
    if reg==True:
        print("Weight Decay: {}", format(weight_decay))
    print("n Epochs: {}", format(n_epoch))
    print("Regularization: {}", format(reg))
   
    if mode == "train" or mode =="test" or mode=="all":
        log_file_path = os.path.join(OUT_DIR, 'GATES_log.txt')
        if mode=="train" or mode=="all":
            with open(log_file_path,'w') as log_file:pass
        for ds_name in DS_NAME:
            if ds_name == "dbpedia":
                db_dir = IN_DBPEDIA_DIR
            elif ds_name == "lmdb":
                db_dir = IN_LMDB_DIR
            elif ds_name == "faces":
                db_dir = IN_FACES_DIR
            else:
                raise ValueError("The database's name must be dbpedia or lmdb")
                sys.exit()
            
            print('loading embeddings and dictionaries for {} ...'.format(ds_name))
            entity2vec, pred2vec, entity2ix, pred2ix = load_emb(ds_name, emb_model)
            entity_dict = entity2vec
            pred_dict = pred2vec
            pred2ix_size = len(pred2ix)
            entity2ix_size = len(entity2ix)
            hidden_size = ent_emb_dim + pred_emb_dim
            start = time.time()
            
            if mode=="train":
                print_to = 'model-training-{}.txt'.format(ds_name)
                with open(print_to, 'w+') as f:
                    f.write("Starting Training \n")
                    f.write("Training model is processed to {} ".format(ds_name))
                    
            if mode=="test":
                print_to = 'model-testing-dbpedia-lmdb.txt'
                if ds_name=='faces':
                   print_to = 'model-testing-{}.txt'.format(ds_name) 
                   with open(print_to, 'w+') as f:
                       f.write("dsFile:true {}\n".format(IN_FACES_DIR))
                       f.write("===============================================================================\n")
                       f.write("dataset:{}\n".format(ds_name))
                        
            for topk in TOP_K:
                train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, test_adjs, test_facts, test_labels = split_data(ds_name, db_dir, topk, FILE_N, weighted_edges_method) 
                if mode == "train" or mode=="all":
                    use_epoch = train_iter(ds_name, train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, reg, n_epoch, save_every, DEVICE, entity_dict, \
                               pred_dict, loss_function, pred2ix_size, hidden_size, pred_emb_dim, ent_emb_dim, lr, dropout, entity2ix_size, hidden_layers, nheads, \
                               word_emb, db_dir, weight_decay, word_emb_calc, topk, FILE_N, concat_model, print_to)
                    
                    with open(log_file_path,'a') as log_file:
                        line = '{}-top{} epoch:\t{}\n'.format(ds_name,topk, str(use_epoch))
                        log_file.write(line)
                        
                if mode == "test" or mode=="all":
                    print("Testing processes for {}@top{}".format(ds_name, topk))
                    use_epoch = _read_epochs_from_log(ds_name, topk) if mode=='test' or mode=='all' else []
                    generate_summary(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, \
                                 DEVICE, use_epoch, db_dir, dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, FILE_N, concat_model, print_to)
                    
                    model = GATES(pred2ix_size, entity2ix_size, pred_emb_dim, ent_emb_dim, DEVICE, dropout, hidden_layers, nheads)
                    pred_tensor, obj_tensor = tensor_from_data(concat_model, entity_dict, pred_dict, test_facts[0][0], word_emb, word_emb_calc)
                    input_tensor = [pred_tensor.to(DEVICE), obj_tensor.to(DEVICE)]
                    y = model(input_tensor, test_adjs[0][0])
                    g = make_dot(y, model.state_dict())
                    g.view()
            total_time = time.time()-start
            if mode=="train":
                print("Training processes time", asHours(total_time))
            elif mode=="test":
                print("Testing processes time", asHours(total_time))
            else:
                print("All processing time", asHours(total_time))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATES: Graph Attention Network for Entity Summarization')
    parser.add_argument("--mode", type=str, default="all", help="use which mode type: train/test/all")
    parser.add_argument("--kge_model", type=str, default="ComplEx", help="use ComplEx/DistMult/ConEx")
    parser.add_argument("--loss_function", type=str, default="BCE", help="use which loss type: BCE/MSE")
    parser.add_argument("--ent_emb_dim", type=int, default=300, help="the embeddiing dimension of entity")
    parser.add_argument("--pred_emb_dim", type=int, default=300, help="the embeddiing dimension of predicate")
    parser.add_argument("--hidden_layers", type=int, default=2, help="the number of hidden layers")
    parser.add_argument("--nheads", type=int, default=3, help="the number of heads attention")
    parser.add_argument("--lr", type=float, default=0.005, help="use to define learning rate hyperparameter")
    parser.add_argument("--dropout", type=float, default='0.0', help="use to define dropout hyperparameter")
    parser.add_argument("--weight_decay", type=float, default='1e-5', help="use to define weight decay hyperparameter if the regularization set to True")
    parser.add_argument("--regularization", type=bool, default=False, help="use to define regularization: True/False")
    parser.add_argument("--save_every", type=int, default=1, help="save model in every n epochs")
    parser.add_argument("--n_epoch", type=int, default=50, help="train model in total n epochs")
    parser.add_argument("--word_emb_model", type=str, default="Glove", help="use which word embedding model: fasttext/Glove")
    parser.add_argument("--word_emb_calc", type=str, default="AVG", help="use which method to compute textual form: SUM/AVG")
    parser.add_argument("--use_epoch", type=int, nargs='+', help="how many epochs to train the model")
    parser.add_argument("--concat_model", type=int, default=1, help="use which concatenation model (1 or 2). In which, 1 refers to KGE + Word embedding, and 2 refers to KGE + (KGE/Word embeddings) depends on the object value")
    parser.add_argument("--weighted_edges_method", type=str, default="", help="use which apply the initialize weighted edges method: tf-idf")
    
    args = parser.parse_args()
    main(args.mode, args.kge_model, args.loss_function, args.ent_emb_dim, args.pred_emb_dim, args.hidden_layers, args.nheads, args.lr, args.dropout, args.regularization, args.weight_decay, \
         args.n_epoch, args.save_every, args.word_emb_model, args.word_emb_calc, args.use_epoch, args.concat_model, args.weighted_edges_method)
    

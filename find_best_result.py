#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:37:20 2020

@author: Asep Fajar Firmansyah
"""
import os # to communicate with operation system 
import os.path as path
import torch
import numpy as np

from utils import tensor_from_data, tensor_from_weight, _eval_Fmeasure
from data_loader import get_data_gold
from model import KGSUM
from generate_summary import generate_summary
  
def find_best_topk(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, device, use_epoch, db_dir,  \
                     dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, file_n, n_epoch, mode, concat_model):
    directory = path.join("data/ESBM_benchmark_v1.2/output_summaries", ds_name)
    if not path.exists(directory):
        os.makedirs(directory)
  
    use_epoch=[]
    print("Model selection for {} top {} based on K-Fold".format(ds_name, topk))
    for num in range(5):
        best_value = 0
        best_epoch = 0
        for epoch in range(n_epoch):
            acc_list = []
            CHECK_DIR = path.join("kgsumm_checkpoint-{}-{}-{}".format(ds_name, topk, num))
            
            kgsumm = KGSUM(pred2ix_size, entity2ix_size, pred_emb_dim, ent_emb_dim, device, dropout, hidden_layers, nheads)
            checkpoint = torch.load(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(epoch)))
            kgsumm.load_state_dict(checkpoint["model_state_dict"])
            kgsumm.to(device)
            adj = test_adjs[num]
            edesc = test_facts[num]
            label = test_labels[num]
            for i in range(len(edesc)):
                eid = edesc[i][0][0]
                pred_tensor, obj_tensor = tensor_from_data(concat_model, entity_dict, pred_dict, edesc[i], word_emb, word_emb_calc)
                input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
                target_tensor = tensor_from_weight(len(edesc[i]), edesc[i], label[i]).to(device)
                output_tensor = kgsumm(input_tensor, adj[i])
              
                output_tensor = output_tensor.view(1, -1).cpu()
                target_tensor = target_tensor.view(1, -1).cpu()
                (label_top_scores, label_top) = torch.topk(target_tensor, topk)
                (output_top_scores, output_top) = torch.topk(output_tensor, topk)
              
                gold_list_top = get_data_gold(db_dir, eid, topk, file_n)
                top_list_output_top = output_top.squeeze(0).numpy().tolist()
              
                acc = _eval_Fmeasure(top_list_output_top, gold_list_top)
                acc_list.append(acc)
            avg_acc = np.mean(acc_list)
            if avg_acc >= best_value:
                best_value = avg_acc
                best_epoch = epoch
            if epoch == n_epoch-1:
                use_epoch.append(best_epoch)
        print('fold {}'.format(num), 'Best score of {} top{}:'.format(ds_name, topk), best_value, 'Best epoch', best_epoch) 
        
    if mode=="find-test":
        #use_epoch = [best_epoch, best_epoch, best_epoch, best_epoch, best_epoch]
        print('List of the best model based on K-Fold', use_epoch)
        generate_summary(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, \
                                 device, use_epoch, db_dir, dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, file_n, concat_model)
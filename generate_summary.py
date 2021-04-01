#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:24:18 2020

@author: Asep Fajar Firmansyah
"""

import os # to communicate with operation system 
import os.path as path
import torch
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

from utils import tensor_from_data, tensor_from_weight, _eval_Fmeasure
from data_loader import get_data_gold
from model import KGSUM

def generate_summary(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, device, use_epoch, db_dir,  \
                     dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, file_n, concat_model):
  directory = path.join("data/ESBM_benchmark_v1.2/output_summaries", ds_name)
  if not path.exists(directory):
    os.makedirs(directory)
  favg_top_all = []
  for num in tqdm(range(5)):
    favg_top_list = []
    CHECK_DIR = path.join("kgsumm_checkpoint-{}-{}-{}".format(ds_name, topk, num))
    
    kgsumm = KGSUM(pred2ix_size, entity2ix_size, pred_emb_dim, ent_emb_dim, device, dropout, hidden_layers, nheads)
    print(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(use_epoch[num])))    
    checkpoint = torch.load(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(use_epoch[num])))
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
      (output_rank_scores, output_rank) = torch.topk(output_tensor, len(edesc[i]))
      
      if not path.exists(path.join(directory, "{}".format(eid))):
        os.makedirs(path.join(directory, "{}".format(eid)))
      writer(db_dir, eid, directory, "top{}".format(topk), output_top)
      writer(db_dir, eid, directory, "rank", output_rank)
      
      #writer(db_dir, eid, directory, "rank", output_rank)

      gold_list_top = get_data_gold(db_dir, eid, topk, file_n)
      top_list_output_top = output_top.squeeze(0).numpy().tolist()
      
      favg_top = _eval_Fmeasure(top_list_output_top, gold_list_top)
      favg_top_list.append(favg_top)
      favg_top_all.append(favg_top)
      
      
    test_favg_top = np.mean(favg_top_list)
    print('\n')
    print('top {} test fold %d:'.format(topk) % num, test_favg_top)
    print('\n')
  test_favg_top_all = np.mean(favg_top_all)
  print('top {} test all:'.format(topk), test_favg_top_all)
  display_parameters(kgsumm, use_epoch[0], CHECK_DIR)
  if ds_name=="lmdb" and topk==10:
      os.system('java -jar evaluation/esummeval_v1.2.jar data/ESBM_benchmark_v1.2/ data/ESBM_benchmark_v1.2/output_summaries/')
  

def writer(db_dir, eid, directory, top_or_rank, output):
    with open(path.join(db_dir, 
            "{}".format(eid), 
            "{}_desc.nt".format(eid)),
            encoding="utf8") as fin, \
    open(path.join(directory,
            "{}".format(eid),
            "{}_{}.nt".format(eid, top_or_rank)),
            "w", encoding="utf8") as fout:
        if top_or_rank == "top5" or top_or_rank == "top10":
            top_list = output.squeeze(0).numpy().tolist()
            #print('top list', top_list)
            for t_num, triple in enumerate(fin):
                if t_num in top_list:
                    fout.write(triple)
        elif top_or_rank == "rank":
            rank_list = output.squeeze(0).numpy().tolist()
            #print(rank_list)
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
              try:
                  fout.write(triples[rank])
              except:
                  pass            
    return

def display_parameters(model, use_epoch, CHECK_DIR):
        print("Initializing model...")
        net = model
        checkpoint = torch.load(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(use_epoch)))
        net.load_state_dict(checkpoint["model_state_dict"])
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in net.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
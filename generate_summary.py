#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:24:18 2020
"""

import os # to communicate with operation system 
import os.path as path
import torch
import numpy as np
from tqdm import tqdm
from numpy import array
from numpy import argmax

from utils import tensor_from_data, tensor_from_weight, _eval_Fmeasure, _eval_ndcg_scores
from data_loader import get_data_gold
from model import GATES

def generate_summary(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, device, use_epoch, db_dir,  \
                     dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, file_n, concat_model, print_to, weighted_edges_method):
  directory = path.join("data/output_summaries", ds_name)
  if not path.exists(directory):
    os.makedirs(directory)
  favg_top_all = []
  ndcg_scores_all = []
  weighted_adjacency_matrix=False
  if weighted_edges_method=="tf-idf":
      weighted_adjacency_matrix = True
  for num in tqdm(range(5)):
    favg_top_list = []
    ndcg_scores = []
    CHECK_DIR = path.join("models", "gates_checkpoint-{}-{}-{}".format(ds_name, topk, num))
    gates = GATES(pred2ix_size, entity2ix_size, pred_emb_dim, ent_emb_dim, device, dropout, hidden_layers, nheads, weighted_adjacency_matrix)
    #print(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(use_epoch[num])))
    checkpoint = torch.load(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(use_epoch[num])))
    gates.load_state_dict(checkpoint["model_state_dict"])
    gates.to(device)
    adj = test_adjs[num]
    edesc = test_facts[num]
    label = test_labels[num]
    gates.eval()
    with torch.no_grad():
        for i in range(len(edesc)):
          eid = edesc[i][0][0]
          pred_tensor, obj_tensor = tensor_from_data(concat_model, entity_dict, pred_dict, edesc[i], word_emb, word_emb_calc)
          input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
          target_tensor = tensor_from_weight(len(edesc[i]), edesc[i], label[i]).to(device)
          output_tensor = gates(input_tensor, adj[i])
          
          output_tensor = output_tensor.view(1, -1).cpu()
          target_tensor = target_tensor.view(1, -1).cpu()
          (label_top_scores, label_top) = torch.topk(target_tensor, topk)
          (output_top_scores, output_top) = torch.topk(output_tensor, topk)
          (output_rank_scores, output_rank) = torch.topk(output_tensor, len(edesc[i]))
          
          if not path.exists(path.join(directory, "{}".format(eid))):
            os.makedirs(path.join(directory, "{}".format(eid)))
          writer(db_dir, eid, directory, "top{}".format(topk), output_top)
          writer(db_dir, eid, directory, "rank", output_rank)
          
          gold_list_top = get_data_gold(db_dir, eid, topk, file_n)
          top_list_output_top = output_top.squeeze(0).numpy().tolist()
          all_list_output_top = output_rank.squeeze(0).numpy().tolist()
          
          favg_top = _eval_Fmeasure(top_list_output_top, gold_list_top)
          favg_top_list.append(favg_top)
          favg_top_all.append(favg_top)
          
          ndcg_score = _eval_ndcg_scores(all_list_output_top, gold_list_top)
          ndcg_scores.append(ndcg_score)
          ndcg_scores_all.append(ndcg_score)
      
      
        test_favg_top = np.mean(favg_top_list)
        print('top {} of {} testing fold %d:'.format(topk, ds_name) % num, test_favg_top, np.average(ndcg_scores))
            
        test_favg_top_all = np.mean(favg_top_all)
  print("### Single Score ###")
  #if ds_name=='faces':
  print("dataset: {}".format(ds_name))
  print("############################################")
  print('Results{}@{}: F-measure={}, NDCG Score={}'.format(ds_name, topk, test_favg_top_all, np.average(ndcg_scores_all)))
  print("#######################################")
  print("\n")
  
  if ds_name=="faces":
      with open(print_to, 'a') as f:
            f.write("Results({}@top{})-single score: F-measure={}, NDCG Score={}\n".format(ds_name, topk, test_favg_top_all, np.mean(ndcg_scores_all)))    
  if ds_name=="lmdb" and topk==10:
      os.system('java -jar evaluation/esummeval_v1.2.jar data/ESBM_benchmark_v1.2/ data/output_summaries/ > {}'.format(print_to))
      os.system('java -jar evaluation/esummeval_v1.2.jar data/ESBM_benchmark_v1.2/ data/data/output_summaries/')
              

def ensembled_generating_summary(ds_name, test_adjs, test_facts, test_labels, pred_dict, entity_dict, pred2ix_size, pred_emb_dim, ent_emb_dim, device, use_epoch, db_dir,  \
                     dropout, entity2ix_size, hidden_layers, nheads, word_emb, word_emb_calc, topk, file_n, concat_model, print_to, weighted_edges_method):
    directory = path.join("data/output_summaries_ensembled", ds_name)
    if not path.exists(directory):
        os.makedirs(directory)
    favg_top_all = []  
    ndcg_scores_all = []
    #load models
    models = []
    weighted_adjacency_matrix=False
    if weighted_edges_method=="tf-idf":
        weighted_adjacency_matrix = True
            
    for num in tqdm(range(5)):
        CHECK_DIR = path.join("models", "gates_checkpoint-{}-{}-{}".format(ds_name, topk, num))
        gates = GATES(pred2ix_size, entity2ix_size, pred_emb_dim, ent_emb_dim, device, dropout, hidden_layers, nheads, weighted_adjacency_matrix)
        checkpoint = torch.load(path.join(CHECK_DIR, "checkpoint_epoch_{}.pt".format(use_epoch[num])))
        gates.load_state_dict(checkpoint["model_state_dict"])
        gates.to(device)
        models.append(gates)
    
    for num in tqdm(range(5)):
        print("Fold", num)
        favg_top_list = []
        ndcg_scores = []
        adj = test_adjs[num]
        edesc = test_facts[num]
        label = test_labels[num]
        with torch.no_grad():
            for i in range(len(edesc)):
              eid = edesc[i][0][0]
              pred_tensor, obj_tensor = tensor_from_data(concat_model, entity_dict, pred_dict, edesc[i], word_emb, word_emb_calc)
              input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
              target_tensor = tensor_from_weight(len(edesc[i]), edesc[i], label[i]).to(device)
              output_tensor = evaluate_n_members(models, num, input_tensor, adj[i])
              
              output_tensor = output_tensor.view(1, -1).cpu()
              target_tensor = target_tensor.view(1, -1).cpu()
              (label_top_scores, label_top) = torch.topk(target_tensor, topk)
              (output_top_scores, output_top) = torch.topk(output_tensor, topk)
              (output_rank_scores, output_rank) = torch.topk(output_tensor, len(edesc[i]))
              
              if not path.exists(path.join(directory, "{}".format(eid))):
                os.makedirs(path.join(directory, "{}".format(eid)))
              writer(db_dir, eid, directory, "top{}".format(topk), output_top)
              writer(db_dir, eid, directory, "rank", output_rank)
              
              gold_list_top = get_data_gold(db_dir, eid, topk, file_n)
              top_list_output_top = output_top.squeeze(0).numpy().tolist()
              all_list_output_top = output_rank.squeeze(0).numpy().tolist()
              
              favg_top = _eval_Fmeasure(top_list_output_top, gold_list_top)
              favg_top_list.append(favg_top)
              favg_top_all.append(favg_top)
              
              ndcg_score = _eval_ndcg_scores(all_list_output_top, gold_list_top)
              ndcg_scores.append(ndcg_score)
              ndcg_scores_all.append(ndcg_score)
          
            test_favg_top = np.mean(favg_top_list)
            print('top {} of {} testing fold %d:'.format(topk, ds_name) % num, test_favg_top, np.average(ndcg_scores))
                
            test_favg_top_all = np.mean(favg_top_all)
    print("\n")
    print("### Ensembled score ###")
    #if ds_name=='faces':
    print("dataset: {}".format(ds_name))
    print("############################################")
    print('Results{}@{}: F-measure={}, NDCG Score={}'.format(ds_name, topk, test_favg_top_all, np.average(ndcg_scores_all)))
    print("#######################################")
    print("\n")
      
    if ds_name=="faces":
        with open(print_to, 'a') as f:
            f.write("Results({}@top{})-ensembled score: F-measure={}, NDCG Score={}\n".format(ds_name, topk, test_favg_top_all, np.mean(ndcg_scores_all)))  
    if ds_name=="lmdb" and topk==10:
        os.system('java -jar evaluation/esummeval_v1.2.jar data/ESBM_benchmark_v1.2/ data/output_summaries_ensembled/ > {}'.format("model-testing-dbpedia-lmdb-ensembled.txt"))
        os.system('java -jar evaluation/esummeval_v1.2.jar data/ESBM_benchmark_v1.2/ data/output_summaries_ensembled/')
        
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, fold, input_tensor, adj):
    if fold==4:
        subset = [members[0],  members[4]]
    else:
        subset = [members[fold],  members[fold+1]]
    yhat = ensemble_predictions(subset, input_tensor, adj)
    return yhat

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, input_tensor, adj):
	# make predictions
    yhats = torch.stack([model(input_tensor, adj) for model in members])
    result = torch.sum(yhats, axis=0)
    return result

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
            for t_num, triple in enumerate(fin):
                if t_num in top_list:
                    fout.write(triple)
        elif top_or_rank == "rank":
            rank_list = output.squeeze(0).numpy().tolist()
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
              try:
                  fout.write(triples[rank])
              except:
                  pass            
    return
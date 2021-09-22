#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 10:19:58 2021

@author: asep
"""

import os
import os.path as path
from data_loader import get_all_data
import numpy as np
import math

ds = ["dbpedia", "lmdb", "faces"]
topk=[5, 10]
file_n=6


IN_ESBM_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data', 'ESBM_benchmark_v1.2')
IN_DBPEDIA_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/ESBM_benchmark_v1.2', 'dbpedia_data')
IN_LMDB_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/ESBM_benchmark_v1.2', 'lmdb_data')
IN_FACES_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/FACES', 'faces_data')
IN_FACES = os.path.join(path.dirname(os.getcwd()), 'GATES/data', 'FACES')
OUTPUT_DBPEDIA_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/output_summaries_ensembled', 'dbpedia')
OUTPUT_LMDB_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/output_summaries_ensembled', 'lmdb')
OUTPUT_FACES_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/output_summaries_ensembled', 'faces')

def get_topk_triples(db_path, num, top_n, triples_dict):
  triples=[]
  encoded_triples = []
  with open(path.join(db_path, "{}".format(num), "{}_rank.nt".format(num)), encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
        triple = triple.replace("\n", "").strip()
        triples.append(triple)
        
        encoded_triple = triples_dict[triple]
        encoded_triples.append(encoded_triple)
  return triples, encoded_triples

def getNDCGSCore(goldSummaries, algoRank):
    tripleGrade = {}
    for goldSum in goldSummaries:
        #print("gold sum",goldSum)
        for t in goldSum:
            if t not in tripleGrade:
                tripleGrade[t]=1
            else:
                tripleGrade[t]= tripleGrade[t]+1
    #print("dict", tripleGrade)
    gradeList = list(tripleGrade.values())
    #print("list", gradeList)
    gradeList.sort(reverse=True)
    #print("sort", gradeList)
    dcg = 0
    idcg = 0
    
    maxRankPos = len(algoRank)
    maxIdealPos = len(gradeList)
    
    for pos in range(1, maxRankPos+1):
        t = algoRank[pos-1]
        #print("t", t)
        try:
            rel = tripleGrade[t]
        except:
            rel=0
        dcgItem = rel/math.log(pos + 1, 2)
        dcg += dcgItem
        
        if (pos<=maxIdealPos):
            idealRel = gradeList[pos-1]
            #print("ideal", idealRel)
            idcg += idealRel/math.log(pos + 1, 2)
    
    ndcg = dcg/idcg
    return ndcg
    

for dataset in ds:
    if dataset == "dbpedia":
        IN_DATA = IN_DBPEDIA_DIR
        IN_SUMM = OUTPUT_DBPEDIA_DIR
        start = [0, 140]
        end   = [100, 165]
    elif dataset == "lmdb":
        IN_DATA = IN_LMDB_DIR
        IN_SUMM = OUTPUT_LMDB_DIR
        start = [100, 165]
        end   = [140, 175]
    else:
        IN_DATA = IN_FACES_DIR
        IN_SUMM = OUTPUT_FACES_DIR
        start = [0, 25]
        end   = [25, 50]
        
    for k in topk:
        all_ndcg_scores = []
        total_ndcg=0
        for i in range(start[0], end[0]):
            t = i+1
            gold_list_top, triples_dict, triple_tuples = get_all_data(IN_DATA, t, k, file_n)
            #print("############### Top-K Triples ################", t)
            topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
            #print(triples_dict)
            #print("total of gold summaries", len(gold_list_top))
            #print("topk", encoded_topk_triples)
            #ndcg_score = getNDCG(rel)
            ndcg_score = getNDCGSCore(gold_list_top, encoded_topk_triples)
            total_ndcg += ndcg_score
            all_ndcg_scores.append(ndcg_score)
        
        for i in range(start[1], end[1]):
            t = i+1
            gold_list_top, triples_dict, triple_tuples = get_all_data(IN_DATA, t, k, file_n)
            #print("############### Top-K Triples ################", t)
            topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
            #print(triples_dict)
            #print("total of gold summaries", len(gold_list_top))
            #print("topk", encoded_topk_triples)
            #ndcg_score = getNDCG(rel)
            ndcg_score = getNDCGSCore(gold_list_top, encoded_topk_triples)
            total_ndcg += ndcg_score
            all_ndcg_scores.append(ndcg_score)
        
        print("{}@top{}: NDCG={}".format(dataset, k, np.average(all_ndcg_scores)))
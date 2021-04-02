#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:04:19 2020

@author: Asep Fajar Firmansyah
"""
import numpy as np # The library is used for mathematical computation
import os # to communicate with operation system 
import re # is stand for regex expression
import os.path as path
from utils import _compact, _extract, build_dict, build_vec
import scipy.sparse as sp

IN_ESBM_DIR = os.path.join(path.dirname(os.getcwd()), 'KGSUM-GAT/data', 'ESBM_benchmark_v1.2')
IN_DBPEDIA_DIR = os.path.join(path.dirname(os.getcwd()), 'KGSUM-GAT/data/ESBM_benchmark_v1.2', 'dbpedia_data')
IN_LMDB_DIR = os.path.join(path.dirname(os.getcwd()), 'KGSUM-GAT/data/ESBM_benchmark_v1.2', 'lmdb_data')

# get data from ESBM benchmark v.1.2 for cross-validation - adapted by DeepLENS
def get_5fold_train_valid_test_elist(ds_name_str, esbm_dir=IN_ESBM_DIR):
  if ds_name_str == "dbpedia":
    split_path = path.join(esbm_dir, "dbpedia_split")
  elif ds_name_str == "lmdb":
    split_path = path.join(esbm_dir, "lmdb_split")
  else:
    raise ValueError("The database's name must be dbpedia or lmdb")

  trainList, validList, testList = [],[],[]
  for i in range(5): # 5-folds
    # read split eid files
    fold_path = path.join(split_path, 'Fold'+str(i))
    train_eids = _read_split(fold_path,'train')
    valid_eids = _read_split(fold_path,'valid')
    test_eids = _read_split(fold_path,'test')
    trainList.append(train_eids)
    validList.append(valid_eids)
    testList.append(test_eids)
  return trainList, validList, testList

# read split data from data split directories on ESBM benchmark v.1.2 -adapted by DeepLENS
def _read_split(fold_path, split_name):
	'''
	:param fold_path:
	:param split_name: 'train', 'valid', 'test'
	:return:
	'''
	split_eids = []
	with open(path.join(fold_path, "{}.txt".format(split_name)),encoding='utf-8') as f:
		for line in f:
			if len(line.strip())==0:
				continue
			eid = int(line.split('\t')[0])
			split_eids.append(eid)
	return split_eids

# Prepare data for per entity
def get_entity_desc(ds_name, db_path, num):
  data=list()
  with open(path.join(db_path, "{}".format(num), "{}_literal_status.txt".format(num)), 'r', encoding='utf-8') as f:
      for line in f:
          items = line.strip().split('\t')
          #print(items)
          edesc = (num, items[0], items[1], items[2], items[3], items[4])
          data.append(edesc)
  return data

# Build graph
def build_graph(db_path, num, weighted_edges_model):
  triples_idx=list()
  
  with open(path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), encoding="utf8") as reader:
    subjectList = list()
    relationList = list()
    objectList = list()
    for i, triple in enumerate(reader):
      sub, pred, obj, _ = parserline(triple)
      subjectList.append(sub)
      relationList.append(pred)
      objectList.append(obj)

    relations = relationList
    subjects = subjectList
    objects = objectList
    nodes = subjects + objects
    
    relations_dict = {}
    for relation in relations:
        if relation not in relations_dict:
            relations_dict[relation] = len(relations_dict)
            
    nodes_dict = {}
    for node in nodes:
      if node not in nodes_dict :
        nodes_dict[node] = len(nodes_dict)
 
  predicatesObjectsFreq = {} 
  weighted_edges = []
  triples_list=[]
  with open(path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), encoding="utf8") as reader:
    for i, triple in enumerate(reader):
      sub, pred, obj, _ = parserline(triple)
      triples = (sub, pred, obj)
      triple_tuple_idx = (nodes_dict[sub], relations_dict[pred], nodes_dict[obj])
      #print(triple_tuple_idx)
      triples_idx.append(triple_tuple_idx)
      triples_list.append(triples)
  
  for sub, pred, obj in triples_list:    
      if (sub, pred) not in predicatesObjectsFreq:
          predicatesObjectsFreq[(sub, pred)] = 1
      else:
          n = predicatesObjectsFreq[(sub, pred)]
          predicatesObjectsFreq[(sub, pred)] = n+1
                
      nqu=0
      for (s, p) in predicatesObjectsFreq.keys():
          nqu += predicatesObjectsFreq[(s, p)]
      tf = predicatesObjectsFreq[(sub, pred)]/nqu
      fPredOverGraph = 0
      for _, p, o in triples_list:
          if (s, pred) == (s, p) or (pred, o) == (p, o):
              fPredOverGraph +=1
      idf = np.log(len(nodes_dict)/fPredOverGraph)
      tfidf = np.multiply(tf, idf)
      weighted_edges.append(tfidf)
    
  triples_idx = np.array(triples_idx)
  
  if weighted_edges_model=="tf-idf":
      adj = sp.coo_matrix((weighted_edges, (triples_idx[:, 0], triples_idx[:, 2])),
                        shape=(triples_idx.shape[0], triples_idx.shape[0]),
                        dtype=np.float32)
  else:
      adj = sp.coo_matrix((np.ones(triples_idx.shape[0]), (triples_idx[:, 0], triples_idx[:, 2])),
                        shape=(triples_idx.shape[0], triples_idx.shape[0]),
                        dtype=np.float32)
                
  
   
  #print(triples_list)
  #print(triples_idx)
  
  #print(adj)
  return adj

def parserline(triple):
  literal = re.findall('\^\^', triple)
  if len(literal) > 0:
    components = re.findall('\^\^', triple)
  else:
    components = re.findall('<([^:]+:[^\s"<>]*)>', triple)
    
  if len(components) == 2:
    sub, pred = components
    remaining_triple = triple[triple.index(pred) + len(pred) + 2:]
    literal = remaining_triple[:-1]
    obj = literal
    if literal != '"" .':
        obj = re.sub(r'\\', '', obj)
        obj = re.sub(r'""', '"', obj)
    obj =  re.findall(r'"([^"]*)"', obj)[0]
    obj_literal = obj
    
  elif len(components) == 3:
    sub, pred, obj = components
    obj_literal = obj
  else:
    components = triple.split(" ")
    sub = components[0]
    pred = components[1]
    obj = components[2].split("^^")[0]
    obj =  re.findall(r'"([^"]*)"', obj)[0]
    obj_literal = obj

  sub = _compact(_extract(sub))
  pred = _extract(pred)
  obj = _compact(_extract(obj))
  if obj == '':
    obj = 'UNK'
  #print('obj', obj, 'obj literal', obj_literal)
  quad_tuple = (sub, pred, obj, obj_literal)
  return quad_tuple

def get_data_gold(db_path, num, top_n, file_n):
  triples_dict = {}
  with open(path.join(db_path, 
        "{}".format(num), 
        "{}_desc.nt".format(num)),
        encoding="utf8") as reader:
    for i, triple in enumerate(reader):
      triple_tuple = parserline(triple)
      if triple_tuple not in triples_dict:
        triples_dict[triple_tuple] = len(triples_dict)
  gold_list = []
  for i in range(file_n):
    with open(path.join(db_path, 
            "{}".format(num), 
            "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)),
            encoding="utf8") as reader:
      n_list = []
      for i, triple in enumerate(reader):
        triple_tuple = parserline(triple)
        gold_id = triples_dict[triple_tuple]
        n_list.append(gold_id)
      gold_list.append(n_list)
  return gold_list

# get data per entity id (provide data in graph and entity description)
def get_data(ds_name, data_eids, db_dir, weighted_edges_model):
  adj_data = list()
  edesc_data = list() 
  for eid in data_eids:
    adj = build_graph(db_dir, eid, weighted_edges_model)
    edesc = get_entity_desc(ds_name, db_dir, eid)
    adj_data.append(adj)
    edesc_data.append(edesc)
  return adj_data, edesc_data

# provide train, valid, and test data
def split_data(ds_name, db_dir, top_n, file_n, weighted_edges_model):
  if ds_name == "dbpedia":
    train_data, valid_data, test_data = get_5fold_train_valid_test_elist(ds_name, IN_ESBM_DIR) 
  elif ds_name == "lmdb":
    train_data, valid_data, test_data = get_5fold_train_valid_test_elist(ds_name, IN_ESBM_DIR)
  else:
    raise ValueError("The database's name must be dbpedia or lmdb")
  
  # prepare train data
  train_data_adjs = list()
  train_data_edescs = list()
  train_label = list()
  #print("loading training data")
  for train_eids in train_data:
    label = list()
    adjs, edescs = get_data(ds_name, train_eids, db_dir, weighted_edges_model)
    for train_eid in train_eids:
      per_entity_label_dict = prepare_label(ds_name, train_eid, top_n=top_n, file_n=file_n)
      label.append(per_entity_label_dict)
    train_label.append(label)
    train_data_adjs.append(adjs)
    train_data_edescs.append(edescs)

  # prepare valid data
  valid_data_adjs = list()
  valid_data_edescs = list()
  valid_label = list()
  #print("loading validation data")
  for valid_eids in valid_data:
    label = list()
    adjs, edescs = get_data(ds_name, valid_eids, db_dir, weighted_edges_model)
    for valid_eid in valid_eids:
      per_entity_label_dict = prepare_label(ds_name, valid_eid, top_n=top_n, file_n=file_n)
      label.append(per_entity_label_dict)
    valid_label.append(label)
    valid_data_adjs.append(adjs)
    valid_data_edescs.append(edescs)

  # prepare test data
  test_data_adjs = list()
  test_data_edescs = list()
  test_label = list()
  #print("loading testing data")
  for test_eids in test_data:
    label = list()
    adjs, edescs = get_data(ds_name, test_eids, db_dir, weighted_edges_model)
    for test_eid in test_eids:
      per_entity_label_dict = prepare_label(ds_name, test_eid, top_n=top_n, file_n=file_n)
      label.append(per_entity_label_dict)
    test_label.append(label)
    test_data_adjs.append(adjs)
    test_data_edescs.append(edescs)
  
  return train_data_adjs, train_data_edescs, train_label, valid_data_adjs, valid_data_edescs, valid_label, test_data_adjs, test_data_edescs, test_label

# provide label per entity id
def prepare_label(ds_name, num, top_n, file_n):
  if ds_name == "dbpedia":
    db_path = IN_DBPEDIA_DIR
  elif ds_name == "lmdb":
    db_path = IN_LMDB_DIR
  else:
    raise ValueError("The database's name must be dbpedia or lmdb")

  per_entity_label_dict = {}
  for i in range(file_n):
    with open(path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)), encoding="utf8") as reader:
      for i, triple in enumerate(reader):
        sub, pred, obj, _ = parserline(triple)
        counter(per_entity_label_dict, "{}++$++{}".format(pred, obj))
  return per_entity_label_dict

# dict counter
def counter(cur_dict, word):
    if word in cur_dict:
        cur_dict[word] += 1
    else:
        cur_dict[word] = 1
        
# entity dict, transe data required
def process_data(ds_name):
  if ds_name == "dbpedia":
    db_path = IN_DBPEDIA_DIR
    db_start, db_end = [1, 141], [101, 166]
  elif ds_name == "lmdb":
    db_path = IN_LMDB_DIR
    db_start, db_end = [101, 166], [141, 176]
  else:
    raise ValueError("The database's name must be dbpedia or lmdb")
  data = []
  for i in range(db_start[0], db_end[0]):
    #print('id triple', i)  
    quads = get_entity_desc(ds_name, db_path, i)  
    data.extend([[sub, pred, obj, obj_ori]for _, sub, pred, obj, obj_ori in quads])

  for i in range(db_start[1], db_end[1]):
    #print('id triple', i)
    quads = get_entity_desc(ds_name, db_path, i)  
    data.extend([[sub, pred, obj, obj_ori]for _, sub, pred, obj, obj_ori in quads])
  
  # entity dict
  entity2ix = {}
  for sub, _, obj, _ in data:
    if sub not in entity2ix:
      entity2ix[sub] = len(entity2ix)
    if obj not in entity2ix:
      entity2ix[obj] = len(entity2ix)

  # pred dict
  pred2ix = {}
  for _, pred, _, _ in data:
    if pred not in pred2ix:
      pred2ix[pred] = len(pred2ix)

  return data, entity2ix, pred2ix

# Load KGE embeddings
def load_emb(ds_name, emb_model):
    if ds_name == "dbpedia":
        directory = path.join(path.join("data/ESBM_benchmark_v1.2"), "dbpedia_embeddings")
    elif ds_name == "lmdb":
        directory = path.join(path.join("data/ESBM_benchmark_v1.2"), "lmdb_embeddings")
    else:
        raise ValueError("The database's name must be dbpedia or lmdb")
	
    entity2ix = build_dict(path.join(directory, "entities.dict"))
    pred2ix = build_dict(path.join(directory, "relations.dict"))
       
    if emb_model =="DistMult":
       embedding = np.load(path.join(path.join(directory, "DistMult_vec.npz")))
    elif emb_model == "ComplEx":
       embedding = np.load(path.join(path.join(directory, "ComplEx_vec.npz")))
    elif emb_model == "ConEx":
       embedding = np.load(path.join(path.join(directory, "ConEx_vec.npz")))   
    else:
       raise ValueError("Please choose KGE DistMult or ComplEx")
	
    entity_embedding = embedding["ent_embedding"]
    pred_embedding = embedding["rel_embedding"]

    entity2vec = build_vec(entity2ix, entity_embedding)
    pred2vec = build_vec(pred2ix, pred_embedding)
	
    return entity2vec, pred2vec, entity2ix, pred2ix

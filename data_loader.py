#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:04:19 2020
"""
import numpy as np # The library is used for mathematical computation
import os # to communicate with operation system 
import re # is stand for regex expression
import os.path as path
from utils import _compact, _extract, build_dict, build_vec
import scipy.sparse as sp
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON

IN_ESBM_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data', 'ESBM_benchmark_v1.2')
IN_DBPEDIA_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/ESBM_benchmark_v1.2', 'dbpedia_data')
IN_LMDB_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/ESBM_benchmark_v1.2', 'lmdb_data')
IN_FACES_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/FACES', 'faces_data')
IN_FACES = os.path.join(path.dirname(os.getcwd()), 'GATES/data', 'FACES')

# get data from ESBM benchmark v.1.2 for cross-validation - adapted by DeepLENS
def get_5fold_train_valid_test_elist(ds_name_str, esbm_dir=IN_ESBM_DIR):
  if ds_name_str == "dbpedia":
    split_path = path.join(esbm_dir, "dbpedia_split")
  elif ds_name_str == "lmdb":
    split_path = path.join(esbm_dir, "lmdb_split")
  elif ds_name_str == "faces":
    split_path = path.join(esbm_dir, "faces_split")
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
  with open(path.join(db_path, "{}".format(num), "{}_literal_status.txt".format(num)), encoding="utf8") as reader:
      for i, triple in enumerate(reader):
          #print(i, triple)
          sub, pred, obj, literal, _ = triple.split("\t")
          edesc = (num, sub, pred, obj, literal)
          #print(i, edesc)
          data.append(edesc)
          
  return data

# Build graph
def build_graph(db_path, num, weighted_edges_model):
  triples_idx=list()
  
  with open(path.join(db_path, "{}".format(num), "{}_literal_status.txt".format(num)), encoding="utf8") as reader:
    subjectList = list()
    relationList = list()
    objectList = list()
    for i, items in enumerate(reader):
      sub, pred, obj, _, _ = items.split("\t")
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
  with open(path.join(db_path, "{}".format(num), "{}_literal_status.txt".format(num)), encoding="utf8") as reader:
    for i, items in enumerate(reader):
      sub, pred, obj, _, _ = items.split("\t")
      triples = (sub, pred, obj)
      triple_tuple_idx = (nodes_dict[sub], relations_dict[pred], nodes_dict[obj])
      #print(triple_tuple_idx)
      triples_idx.append(triple_tuple_idx)
      triples_list.append(triples)
  #print(num, triples_list, len(triples_idx), triples_idx)
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
  return adj

def get_all_data(db_path, num, top_n, file_n):
  import glob
  triples_dict = {}
  triple_tuples = []
  ### Retrieve all triples of an entity based on eid
  with open(path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
      if len(triple)==1:
        continue  
      triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
      triple_tuples.append(triple_tuple)
      if triple_tuple not in triples_dict:
        triples_dict[triple_tuple] = len(triples_dict)
  gold_list = []
  ds_name = db_path.split("/")[-1].split("_")[0]
  
  ### Get file_n/ n files of ground truth summaries for faces dataset
  if ds_name=="faces":
      gold_files = glob.glob(path.join(db_path, "{}".format(num), "{}_gold_top{}_*".format(num, top_n).format(num)))
      #print(len(gold_files))
      if len(gold_files) != file_n:
          file_n = len(gold_files)
  
  ### Retrieve ground truth summaries of an entity based on eid and total of file_n  
  for i in range(file_n):
    with open(path.join(db_path, 
            "{}".format(num), 
            "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)),
            encoding="utf8") as reader:
      #print(path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)))
      n_list = []
      for i, triple in enumerate(reader):
        if len(triple)==1:
            continue
        triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
        gold_id = triples_dict[triple_tuple]
        n_list.append(gold_id)
      gold_list.append(n_list)
        
  return gold_list, triples_dict, triple_tuples

def get_data_gold(db_path, num, top_n, file_n):
  import glob
  triples_dict = {}
  with open(path.join(db_path, "{}".format(num), "{}_desc.nt".format(num)), encoding="utf8") as reader:   
    for i, triple in enumerate(reader):
      if len(triple)==1:
        continue  
      triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
      if triple_tuple not in triples_dict:
        triples_dict[triple_tuple] = len(triples_dict)
  gold_list = []
  ds_name = db_path.split("/")[-1].split("_")[0]
  if ds_name=="faces":
      gold_files = glob.glob(path.join(db_path, "{}".format(num), "{}_gold_top{}_*".format(num, top_n).format(num)))
      #print(len(gold_files))
      if len(gold_files) != file_n:
          file_n = len(gold_files)
  for i in range(file_n):
    with open(path.join(db_path, 
            "{}".format(num), 
            "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)),
            encoding="utf8") as reader:
      #print(path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)))
      n_list = []
      for i, triple in enumerate(reader):
        if len(triple)==1:
            continue
        triple_tuple = triple.replace("\n", "").strip()#parserline(triple)
        gold_id = triples_dict[triple_tuple]
        n_list.append(gold_id)
      gold_list.append(n_list)
  #print(len(gold_list))
  #print("num {}".format(num), gold_list[0])
  return gold_list

# get data per entity id (provide data in graph and entity description)
def get_data(ds_name, data_eids, db_dir, weighted_edges_model):
  adj_data = list()
  edesc_data = list()
  for eid in data_eids:
    #print("eid", eid)
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
  elif ds_name == "faces":
    train_data, valid_data, test_data = get_5fold_train_valid_test_elist(ds_name, IN_FACES)  
  else:
    raise ValueError("The database's name must be dbpedia or lmdb")
  
  # prepare train data
  train_data_adjs = list()
  train_data_edescs = list()
  train_label = list()
  
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
  import glob
  if ds_name == "dbpedia":
    db_path = IN_DBPEDIA_DIR
  elif ds_name == "lmdb":
    db_path = IN_LMDB_DIR
  elif ds_name == "faces":
    db_path = IN_FACES_DIR
  else:
    raise ValueError("The database's name must be dbpedia or lmdb")

  per_entity_label_dict = {}
  if ds_name=="faces":
      gold_files = glob.glob(path.join(db_path, "{}".format(num), "{}_gold_top{}_*".format(num, top_n).format(num)))
      #print(len(gold_files))
      if len(gold_files) != file_n:
          file_n = len(gold_files)
  for i in range(file_n):
    with open(path.join(db_path, "{}".format(num), "{}_gold_top{}_{}.nt".format(num, top_n, i).format(num)), encoding="utf8") as reader:
      for i, triple in enumerate(reader):
        sub, pred, obj, _, _ = parserline_get_literal(triple, False)
        counter(per_entity_label_dict, "{}++$++{}".format(pred, obj))
  return per_entity_label_dict

# dict counter
def counter(cur_dict, word):
    if word in cur_dict:
        cur_dict[word] += 1
    else:
        cur_dict[word] = 1
        
# entity dict
def process_data(ds_name):
  if ds_name == "dbpedia":
    db_path = IN_DBPEDIA_DIR
    db_start, db_end = [1, 141], [101, 166]
  elif ds_name == "lmdb":
    db_path = IN_LMDB_DIR
    db_start, db_end = [101, 166], [141, 176]
  elif ds_name == "faces":
    db_path = IN_FACES_DIR
    db_start, db_end = [1, 26], [26, 51]  
  else:
    raise ValueError("The database's name must be dbpedia or lmdb")
  data = []
  for i in range(db_start[0], db_end[0]):
    print('id triple', i)  
    quads = get_entity_desc(ds_name, db_path, i)  
    data.extend([[sub, pred, obj, obj_ori]for _, sub, pred, obj, obj_ori in quads])

  for i in range(db_start[1], db_end[1]):
    print('id triple', i)
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
    elif ds_name == "faces":
        directory = path.join(path.join("data/FACES"), "faces_embeddings")
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

def gen_literal(ds_name):
    
    if ds_name == "dbpedia":
        db_path = IN_DBPEDIA_DIR
        db_start, db_end = [1, 141], [101, 166]
    elif ds_name == "lmdb":
        db_path = IN_LMDB_DIR
        db_start, db_end = [101, 166], [141, 176]
    elif ds_name == "faces":
        db_path = IN_FACES_DIR
        db_start, db_end = [1, 26], [26, 51]
    else:
        raise ValueError("The database's name must be dbpedia or lmdb or faces")
    #print("stage 1")    
    for i in tqdm(range(db_start[0], db_end[0])):
        with open(path.join(db_path, "{}".format(i), "{}_literal_status.txt".format(i)), "w", encoding="utf-8") as f:
            with open(path.join(db_path, "{}".format(i), "{}_desc.nt".format(i)), encoding="utf8") as reader:
                for triple in reader:
                    sub, pred, obj, obj_literal, status = parserline_get_literal(triple, True)
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(sub, pred, obj, obj_literal, status))            
    
    #print("stage 2")
    for i in tqdm(range(db_start[1], db_end[1])):
        with open(path.join(db_path, "{}".format(i), "{}_literal_status.txt".format(i)), "w", encoding="utf-8") as f:
            with open(path.join(db_path, "{}".format(i), "{}_desc.nt".format(i)), encoding="utf8") as reader:
                for triple in reader:
                    sub, pred, obj, obj_literal, status = parserline_get_literal(triple, True)
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(sub, pred, obj, obj_literal, status))
                    
def parserline_get_literal(triple, getLabelFlag):
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
    status = "literal"
    
  elif len(components) == 3:
    sub, pred, obj = components
    #print(components)
    status = "resource"
    if getLabelFlag:
        uri = obj.split("/")
        if uri[2]=="data.linkedmdb.org":
            id_ = uri[-1]
            key= uri[-2]
            if key!="movie":
                keyword = "{}:{}".format(key, id_)
                obj_literal = get_label_of_entity_lmdb(keyword)
                if obj_literal == "None":
                    obj_literal = uri[-1]
            else:
                obj_literal = uri[-1]
        else:
            obj_literal = get_label_of_entity(obj)
            if obj_literal == "None":
                obj_literal = uri[-1]
                if obj_literal=="":
                    obj_literal = obj
    else:
      obj_literal = obj.split("/")[-1].replace("_", " ")  
  else:
    components = triple.split(" ")
    sub = components[0]
    pred = components[1]
    obj = components[2].split("^^")[0]
    obj =  re.findall(r'"([^"]*)"', obj)[0]
    obj_literal =  obj
    status = "literal"
    
  sub = _compact(_extract(sub))
  pred = _extract(pred)
  obj = _compact(_extract(obj))
  if obj == '':
    obj = 'UNK'
    obj_literal='UNK'
  
  return sub, pred, obj, obj_literal, status

def get_label_of_entity(uri):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery("""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label
        WHERE { <%s> rdfs:label ?label }
    """ % (uri))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    for result in results["results"]["bindings"]:
        try:
            if result["label"]["xml:lang"] == "en":
                return result["label"]["value"]
        except:
            return result["label"]["value"]
    
    return "None"

def get_label_of_entity_lmdb(uri):
    sparql = SPARQLWrapper("https://api.triplydb.com/datasets/Triply/linkedmdb/services/linkedmdb/sparql")
    sparql.setQuery("""
        PREFIX film_set_designer: <https://triplydb.com/Triply/linkedmdb/id/film_set_designer/>
        PREFIX film_format: <https://triplydb.com/Triply/linkedmdb/id/film_format/>
        PREFIX country: <https://triplydb.com/Triply/linkedmdb/id/country/>
        PREFIX film_subject: <https://triplydb.com/Triply/linkedmdb/id/film_subject/>
        PREFIX cinematographer: <https://triplydb.com/Triply/linkedmdb/id/cinematographer/>
        PREFIX production_company: <https://triplydb.com/Triply/linkedmdb/id/production_company/>
        PREFIX music_contributor: <https://triplydb.com/Triply/linkedmdb/id/music_contributor/>
        PREFIX editor: <https://triplydb.com/Triply/linkedmdb/id/editor/>
        PREFIX film_cut: <https://triplydb.com/Triply/linkedmdb/id/film_cut/>
        PREFIX director: <https://triplydb.com/Triply/linkedmdb/id/director/>
        PREFIX producer: <https://triplydb.com/Triply/linkedmdb/id/producer/>
        PREFIX writer: <https://triplydb.com/Triply/linkedmdb/id/writer/>
        PREFIX film_story_contributor: <https://triplydb.com/Triply/linkedmdb/id/film_story_contributor/> 
        PREFIX film_genre: <https://triplydb.com/Triply/linkedmdb/id/film_genre/>
        PREFIX performance: <https://triplydb.com/Triply/linkedmdb/id/performance/>
        PREFIX actor: <https://triplydb.com/Triply/linkedmdb/id/actor/>
        PREFIX film_art_director: <https://triplydb.com/Triply/linkedmdb/id/film_art_director/>
        PREFIX film: <https://triplydb.com/Triply/linkedmdb/id/film/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?s ?label
        WHERE { %s rdfs:label ?label }
    """ % uri)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    for result in results["results"]["bindings"]:
        try:
            if result["label"]["xml:lang"] == "en":
                return result["label"]["value"]
        except:
            return result["label"]["value"]
    
    return "None"

def split_upper(s):
    return re.split("([A-Z][^A-Z]*)", s)

def main():
    ds_name = "faces"
    data, entity2ix, pred2ix = process_data(ds_name)
    print(entity2ix)
if __name__ == "__main__":
    main()
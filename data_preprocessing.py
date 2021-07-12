from data_loader import process_data, split_data
import os
import os.path as path
import argparse

IN_DBPEDIA_DIR = os.path.join(path.dirname(os.getcwd()), 'v1.2/data/ESBM_benchmark_v1.2', 'dbpedia_data')
IN_LMDB_DIR = os.path.join(path.dirname(os.getcwd()), 'v1.2/data/ESBM_benchmark_v1.2', 'lmdb_data')
IN_FACES_DIR = os.path.join(path.dirname(os.getcwd()), 'GATES/data/FACES', 'faces_data')

# save entity2ix and pred2ix into file
def gen_data(ds_name, data, entity_to_ix, pred_to_ix,  top_n, file_n):
    # make dir
    if ds_name == "dbpedia":
        directory = path.join(path.join("data/ESBM_benchmark_v1.2"), "dbpedia_rotate")
        db_dir = IN_DBPEDIA_DIR
    elif ds_name == "lmdb":
        directory = path.join(path.join("data/ESBM_benchmark_v1.2"), "lmdb_rotate")
        db_dir = IN_LMDB_DIR
    elif ds_name == "faces":
        directory = path.join(path.join("data/FACES"), "faces_embeddings")
        db_dir = IN_FACES_DIR
    else:
        raise ValueError("The database's name must be dbpedia or lmdb")
    
    if not path.exists(directory):
        os.makedirs(directory)
    
    with open(path.join(directory, "entities.dict"), "w", encoding="utf-8") as f:
        dict_sorted =  sorted(entity_to_ix.items(), key = lambda x:x[1], reverse = False)
        for entity in dict_sorted:
            f.write("{}\t{}\n".format(entity[1], entity[0]))

    with open(path.join(directory, "relations.dict"), "w", encoding="utf-8") as f:
        dict_sorted =  sorted(pred_to_ix.items(), key = lambda x:x[1], reverse = False)
        for relation in dict_sorted:
            f.write("{}\t{}\n".format(relation[1], relation[0]))
    
    _, train_data_edescs, _, _, valid_data_edescs, _, _, test_data_edescs, _ = split_data(ds_name, db_dir, top_n, file_n, 'coo_matrix')

    # train2id
    train_data = list()
    for i in range(5):
        for train_data_edesc in train_data_edescs[i]:
            for _, sub, pred, obj, obj_literal in train_data_edesc:
                train_data_tuple = (sub, pred, obj)
                train_data.append(train_data_tuple)

    with open(path.join(directory, "train.txt"), "w", encoding="utf-8") as f:
        for [sub, pred, obj] in train_data:
            f.write("{}\t{}\t{}\n".format(sub, pred, obj))

    valid_data = list()
    for i in range(5):
        for valid_data_edesc in valid_data_edescs[i]:
            for _, sub, pred, obj, obj_literal in valid_data_edesc:
                valid_data_tuple = (sub, pred, obj)
                valid_data.append(valid_data_tuple)
    
    with open(path.join(directory, "valid.txt"), "w", encoding="utf-8") as f:
        for [sub, pred, obj] in valid_data:
            f.write("{}\t{}\t{}\n".format(sub, pred, obj))

    test_data = list()
    for i in range(5):
        for test_data_edesc in test_data_edescs[i]:
            for _, sub, pred, obj, obj_literal in test_data_edesc:
                test_data_tuple = (sub, pred, obj)
                test_data.append(test_data_tuple)

    with open(path.join(directory, "test.txt"), "w", encoding="utf-8") as f:
        for [sub, pred, obj] in test_data:
            f.write("{}\t{}\t{}\n".format(sub, pred, obj))

def main(ds_name, top_n, file_n):
    print('generate data and dictionaries')
    data, entity2ix, pred2ix= process_data(ds_name)
    print('writing data to file')
    gen_data(ds_name, data, entity2ix, pred2ix, top_n, file_n)

if __name__ == "__main__":
      parser = argparse.ArgumentParser(description='GATES: Preparing data for KGE')
      parser.add_argument("--ds_name", type=str, default="dbpedia", help="use dbpedia or lmdb")
      parser.add_argument("--top_n", type=int, default=10, help="use top 5 or 10 gold(label) files")
      parser.add_argument("--file_n", type=int, default=6, help="the number of gold(label) files in ESBM benchmark")
      args = parser.parse_args()
      main(args.ds_name, args.top_n, args.file_n)


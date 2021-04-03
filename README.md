# GATES
## Graph Attention Networks for Entity Summarization

The  entity  summarization  task  has  recently  gained  significant  attention  to  provide  concise  information  about  various  facts  con-tained  in  large  knowledge  graphs.  Presently,  the  best  performing  approaches rely on a supervised learning model using neural network methods with sequence to sequence learning. In contrast with existing methods, we introduce GATES as a new approach for entity summarization task using deep learning for graphs. It combines leveraging graph structure and textual semantics to encode triples and advantages deep learn-ing on graphs to generate a score for each candidate triple. We evaluated GATES on the ESBM benchmark, which comprises DBpedia and LinkedMDB datasets. Our results show that GATES outperforms state-of-the-art approaches on all datasets, in which F1 scores for the top-5 and top-10 of DBpedia are 0.478 and 0,629, respectively. Also, F1 scores for the top-5 and top-10 of LinkedMDB are 0.503 and 0.529, consecutively.

## Dataset

On this experiment, [ESBM benchmark v.1.2](https://github.com/nju-websoft/ESBM/tree/master/v1.2) is used as dataset to train and test the GATES model. It consists of 175 entities related to 150 entities from DBpedia and 25 entities from LinkedMDB.

## Pre-trained Knowledge Graph Embedding Models

GATES implements knowledge graph embedding and also provides pre-trained model for each model on ESBM benchmark dataset as follows:
* ComplEx
* ConEx
* DistMult

## Pre-trained Word Embedding Models 

GATES applies Glove and fastText as word embeddings.

### Glove
1. Download pre-trained model [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
2. Extract the zip file in data folder

### fastText
1. Download pre-trained model [wiki-news-300d-1M.vec.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
2. Extract the zip file and put the file on data folder

## Visualization Tools

We use a third party to visualize the training and validation loss, and accuracy. 
If you don't install visdom yet, please install visdom as follows:
```
pip install visdom
``` 

And run the visdom before you execute the train model. Just type visdom on terminal and enter.
```
visdom
```

## Installation
```
git clone https://github.com/dice-group/GATES.git  
```


## Usage
```
usage: main.py [-h] [--mode MODE] [--kge_model KGE_MODEL]
               [--loss_function LOSS_FUNCTION] [--ent_emb_dim ENT_EMB_DIM]
               [--pred_emb_dim PRED_EMB_DIM] [--hidden_layers HIDDEN_LAYERS]
               [--nheads NHEADS] [--lr LR] [--dropout DROPOUT]
               [--weight_decay WEIGHT_DECAY] [--regularization REGULARIZATION]
               [--save_every SAVE_EVERY] [--n_epoch N_EPOCH]
               [--word_emb_model WORD_EMB_MODEL]
               [--word_emb_calc WORD_EMB_CALC]
               [--use_epoch USE_EPOCH [USE_EPOCH ...]]
               [--concat_model CONCAT_MODEL]
               [--weighted_edges_method WEIGHTED_EDGES_METHOD]

GATES: Graph Attention Network for Entity Summarization

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           use which mode type: train/test/all
  --kge_model KGE_MODEL
                        use ComplEx/DistMult/ConEx
  --loss_function LOSS_FUNCTION
                        use which loss type: BCE/MSE
  --ent_emb_dim ENT_EMB_DIM
                        the embeddiing dimension of entity
  --pred_emb_dim PRED_EMB_DIM
                        the embeddiing dimension of predicate
  --hidden_layers HIDDEN_LAYERS
                        the number of hidden layers
  --nheads NHEADS       the number of heads attention
  --lr LR               use to define learning rate hyperparameter
  --dropout DROPOUT     use to define dropout hyperparameter
  --weight_decay WEIGHT_DECAY
                        use to define weight decay hyperparameter if the
                        regularization set to True
  --regularization REGULARIZATION
                        use to define regularization: True/False
  --save_every SAVE_EVERY
                        save model in every n epochs
  --n_epoch N_EPOCH     train model in total n epochs
  --word_emb_model WORD_EMB_MODEL
                        use which word embedding model: fasttext/Glove
  --word_emb_calc WORD_EMB_CALC
                        use which method to compute textual form: SUM/AVG
  --use_epoch USE_EPOCH [USE_EPOCH ...]
                        how many epochs to train the model
  --concat_model CONCAT_MODEL
                        use which concatenation model (1 or 2). In which, 1
                        refers to KGE + Word embedding, and 2 refers to KGE +
                        (KGE/Word embeddings) depends on the object value
  --weighted_edges_method WEIGHTED_EDGES_METHOD
                        use which apply the initialize weighted edges method:
                        tf-idf

```

### Training the model

```
python main.py --mode train --weighted_edges_method tf-idf
```

### Testing the model
```
python main.py --mode test --weighted_edges_method tf-idf
```

### Evaluation Result

Evaluation Method: F-Measure

| Model               | DBpedia                  || LMDB                   ||
| ------------------- | ------------| ------------|------------|------------|
|                     | K=5         | K=10        | K=5        | K=10       |
| DeepLENS            | 0,402       | 0,574       | 0,474      | 0,493      |
| ESA                 | 0,331       | 0,532       | 0,350      | 0,416      |
| GATES               | 0,478       | 0,629       | 0,503      | 0,529      |

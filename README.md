# GATES
## Graph Attention Networks for Entity Summarization

The sheer size of modern knowledge graphs has led to in-
creased attention being paid to the entity summarization task. Given a
knowledge graph T and an entity e found therein, solutions to entity
summarization select a subset of the triples from T which summarize
e’s concise bound description. Presently, the best performing approaches
rely on sequence-to-sequence models to generate entity summaries and
rely on little to none of the structure information of T during the sum-
marization process. We hypothesize that this structure information can
be exploited to compute better summaries. To verify our hypothesis,
we develop GATES, a new entity summarization approach that com-
bines topological information and knowledge graph embeddings to en-
code triples. The topological information is encoded by means of a Graph
Attention Network. We evaluate GATES on the ESBM benchmark. Our
results show that GATES outperforms the state-of-the-art approaches
DeepLENS and ESA and reaches up to 0.62 F-measure. 

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

## Environment and Dependency

### Environment

* Ubuntu 10.04.2 LTS
* python 3.6+
* pytorch 1.7.0

### Dependencies

Our dependencies from external library that are required to run the model, you need to install them as follow:

```
pip install numpy==1.19.2
pip install tqdm
pip install gensim==3.8.3
pip install scipy==1.5.4
pip install nltk==3.5
pip install psutil==5.8.0
```
or

```
pip install -r requirements.txt
```
## Visualization Tools

We use a third party to visualize the training and validation loss, and accuracy. 
If you haven't install visdom yet, please install visdom as follows:
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

| Model               | DBpedia                  || LMDB                   || FACES ||	
| ------------------- | ------------| ------------|------------|------------|-------|--------|
|                     | K=5         | K=10        | K=5        | K=10       | K=5   | K=10   |
| DeepLENS            | 0,402       | 0,574       | 0,474      | 0,493      |0,133  | 0,249  |
| ESA                 | 0,331       | 0,532       | 0,350      | 0,416      |-      |-       |
| NEST                | -	     |  -           |     -       |     -    |**0,272** |0,346 |
| GATES               | **0,447**   | **0,592**   | **0,491**  | **0,505**  |0,229     | **0.356** |


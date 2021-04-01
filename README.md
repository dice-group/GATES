# GATES
## Graph Attention Networks for Entity Summarization

The  entity  summarization  task  has  recently  gained  signifi-cant  attention  to  provide  concise  information  about  various  facts  con-tained  in  large  knowledge  graphs.  Presently,  the  best  performing  ap-proaches rely on a supervised learning model using neural network meth-ods with sequence to sequence learning. In contrast with existing meth-ods, we introduce GATES as a new approach for entity summarizationtask using deep learning for graphs. It combines leveraging graph struc-ture and textual semantics to encode triples and advantages deep learn-ing on graphs to generate a score for each candidate triple. We evalu-ated GATES on the ESBM benchmark, which comprises DBpedia andLinkedMDB datasets. Our results show that GATES outperforms state-of-the-art approaches on all datasets, in which F1 scores for the top-5 andtop-10 of DBpedia are 0.462 and 0,615, respectively. Also, F1 scores forthe top-5 and top-10 of LinkedMDB are 0.495 and 0.514, consecutively.

## Dataset

On this experiment, [ESBM benchmark v.1.2](https://github.com/nju-websoft/ESBM/tree/master/v1.2) is used as dataset to train and test the GATES model. It consists of 175 entities related to 150 entities from DBpedia and 25 entities from LinkedMDB.

## Pre-trained Knowledge Graph Embedding Model

GATES implements knowledge graph embedding and also provides pre-trained model for each model on ESBM benchmark dataset as follows:
* ComplEx
* ConEx
* DistMult

## Pre-trained Word Embedding Model 

GATES applies Glove and fastText as word embeddings.

### Glove
1. Download pre-trained model [glove.6B.zip] (http://nlp.stanford.edu/data/glove.6B.zip)
2. Extract the zip file in data folder

### fastText
1. Download pre-trained model [wiki-news-300d-1M.vec.zip] (https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
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

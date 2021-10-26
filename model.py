#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:04:58 2020
"""

# Define model
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from utils import normalize_adj, normalize_features

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, weighted_adjacency_matrix, alpha, concat=True, in_edge_features=1):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.in_edge_features = in_edge_features
        self.weighted_adjacency_matrix = weighted_adjacency_matrix

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        if weighted_adjacency_matrix==False:
            self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        else:
            self.a = nn.Parameter(torch.empty(size=(3*out_features, 1)))
        
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.We = nn.Parameter(torch.empty(size=(in_edge_features, out_features)))
        nn.init.xavier_uniform_(self.We.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        if self.weighted_adjacency_matrix==True:
            WeE = torch.mm(edge, self.We)
            a_input = self._prepare_attentional_mechanism_input_with_edge_features(Wh, WeE)
        else:
            a_input = self._prepare_attentional_mechanism_input(Wh)
            
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    #def _prepare_edge_attention(self, Eh):   
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1in_edge_features, 
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eNin_edge_features, 

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)
    
    def _prepare_attentional_mechanism_input_with_edge_features(self, Wh, WeE):
        N = Wh.size()[0] # number of nodes
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        WeE_repeated_in_chunks = WeE.repeat_interleave(N, dim=0)
        
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating, WeE_repeated_in_chunks], dim=1)
        
        return all_combinations_matrix.view(N, N, 3 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, weighted_adjacency_matrix):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout, weighted_adjacency_matrix, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, weighted_adjacency_matrix, alpha=alpha, concat=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, edge, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge, adj))
        #return F.log_softmax(x, dim=1)
        return self.softmax(x)

class GATES(nn.Module):
  def __init__(self, pred2ix_size, entity2ix_size, pred_embedding_dim, entity_embedding_dim, device, dropout, hidden_layer, nheads, weighted_adjacency_matrix):
    super(GATES, self).__init__()
    self.pred2ix_size = pred2ix_size
    self.entity2ix_size = entity2ix_size
    self.pred_embedding_dim = pred_embedding_dim
    self.entity_embedding_dim = entity_embedding_dim
    self.input_size = self.entity_embedding_dim + self.pred_embedding_dim
    self.hidden_layer = hidden_layer
    self.nheads = nheads
    self.dropout = dropout
    self.weighted_adjacency_matrix = weighted_adjacency_matrix
    self.device = device
    self.gat = GAT(nfeat=self.input_size, nhid=self.hidden_layer, nclass=1, dropout=self.dropout, alpha=0.2, nheads=self.nheads, weighted_adjacency_matrix=self.weighted_adjacency_matrix)
    self.device = device
        
  def forward(self, input_tensor, adj):
    
    pred_embedded = input_tensor[0]
    obj_embedded = input_tensor[1]
    embedded = torch.cat((pred_embedded, obj_embedded), 2)
    embedded = torch.flatten(embedded, start_dim=1)
    
    edge = adj.data
    
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))

    features = normalize_features(embedded.detach().numpy())
    features = torch.FloatTensor(np.array(features))
    
    edge = torch.FloatTensor(np.array(edge)).unsqueeze(1)
    
    logits = self.gat(features, edge, adj)
    
    return logits
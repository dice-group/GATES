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

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return self.softmax(x)

class GATES(nn.Module):
  def __init__(self, pred2ix_size, entity2ix_size, pred_embedding_dim, entity_embedding_dim, device, dropout, hidden_layer, nheads):
    super(GATES, self).__init__()
    self.pred2ix_size = pred2ix_size
    self.entity2ix_size = entity2ix_size
    self.pred_embedding_dim = pred_embedding_dim
    self.entity_embedding_dim = entity_embedding_dim
    self.input_size = self.entity_embedding_dim + self.pred_embedding_dim
    self.hidden_layer = hidden_layer
    self.nheads = nheads
    self.dropout = dropout
    self.device = device
    self.gat = GAT(nfeat=self.input_size, nhid=self.hidden_layer, nclass=1, dropout=self.dropout, nheads=self.nheads, alpha=0.2)
    #self.gat = models.GAT(self.input_size, self.hidden_layer, 1, self.nheads, self.dropout, self.device)
    self.device = device
        
  def forward(self, input_tensor, adj):
    
    pred_embedded = input_tensor[0]
    obj_embedded = input_tensor[1]
    embedded = torch.cat((pred_embedded, obj_embedded), 2)
    embedded = torch.flatten(embedded, start_dim=1)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))\

    features = normalize_features(embedded.detach().numpy())
    features = torch.FloatTensor(np.array(features))
    
    logits = self.gat(features, adj)
    
    return logits
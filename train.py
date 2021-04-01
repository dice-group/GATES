#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:05:52 2020

@author: Asep Fajar Firmansyah
"""
import os # to communicate with operation system 
import os.path as path
import torch
from torch import optim
import numpy as np

from model import GATES
from utils import tensor_from_data, tensor_from_weight, _eval_Fmeasure
from data_loader import get_data_gold

def train_iter(ds_name, train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, reg, n_epoch, save_every, device, entity_dict, \
                       pred_dict, loss_function, pred2ix_size, hidden_size, pred_emb_dim, ent_emb_dim, lr, dropout, entity2ix_size, hidden_layers, nheads, \
                       word_emb, db_dir, weight_decay, word_emb_calc, topk, file_n, viz, concat_model):
    if reg == True:
        print("use regularization in training")
    best_epoch_list=[]
    gates = GATES(pred2ix_size, entity2ix_size, pred_emb_dim, ent_emb_dim, device, dropout, hidden_layers, nheads)
    gates.to(device)
    for i in range(5):
        if reg:
            optimizer = optim.Adam(gates.parameters(), lr=lr, weight_decay=weight_decay)
        else:    
            optimizer = optim.Adam(gates.parameters(), lr=lr)
        directory = os.path.join(os.getcwd(), path.join("model", "gates_checkpoint-{}-{}-{}".format(ds_name, topk, i)))
        best_epoch = train(gates, ds_name, train_adjs[i], train_facts[i], train_labels[i], \
                           val_adjs[i], val_facts[i], val_labels[i], \
                           loss_function, optimizer, n_epoch, save_every, device, entity_dict, pred_dict, reg, directory, word_emb_calc, viz, i, word_emb, db_dir, topk, file_n, concat_model)
        best_epoch_list.append(best_epoch)
        
    print('Best epoch', best_epoch_list)

# Define training model
def train(kgsumm, ds_name, adj, edesc, label, val_adj, val_edesc, val_label, \
        loss_function, optimizer, n_epoch, save_every, device, entity_dict, pred_dict, reg, directory, word_emb_calc, viz, fold, word_emb, db_dir, topk, file_n, concat_model):
    if not path.exists(directory):
        os.makedirs(directory)
    print('n epoch', n_epoch)
    viz.line([[0.0], [0.0]], [0.], win='{}_loss_{}_top{}'.format(ds_name, fold, topk), opts=dict(title='train loss K-Fold {} top{}'.format(fold, topk), legend=['train loss', 'validation loss']))
    viz.line([[0.]], [0.], win='{}_accuracy_{}_top{}'.format(ds_name, fold, topk), opts=dict(title='acc top {} - Fold {}'.format(topk, fold),legend=['acc. top {}'.format(topk)]))
    best_epoch = 0
    best_acc = 0
    
    for epoch in range(n_epoch):
        
        total_loss = 0
        val_total_loss = 0
        kgsumm.train()
        for i in range(len(edesc)):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            pred_tensor, obj_tensor = tensor_from_data(concat_model, entity_dict, pred_dict, edesc[i], word_emb, word_emb_calc)
            input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
            target_tensor = tensor_from_weight(len(edesc[i]), edesc[i], label[i]).to(device)
            output_tensor = kgsumm(input_tensor, adj[i])
            loss = loss_function(output_tensor.view(-1), target_tensor.view(-1)).to(device)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        kgsumm.eval()
        favg_top_list = []
        with torch.no_grad():
            for i in range(len(val_edesc)):
                eid = val_edesc[i][0][0]
                val_pred_tensor, val_obj_tensor = tensor_from_data(concat_model, entity_dict, pred_dict, val_edesc[i], word_emb, word_emb_calc)
                val_input_tensor = [val_pred_tensor.to(device), val_obj_tensor.to(device)]
                val_target_tensor = tensor_from_weight(len(val_edesc[i]), val_edesc[i], val_label[i]).to(device)
                val_output_tensor = kgsumm(val_input_tensor, val_adj[i])
                val_loss = loss_function(val_output_tensor.view(-1), val_target_tensor.view(-1)).to(device)
                
                val_output_tensor = val_output_tensor.view(1, -1).cpu()
                (output_top_scores, output_top) = torch.topk(val_output_tensor, topk)
                gold_list_top = get_data_gold(db_dir, eid, topk, file_n)
                top_list_output_top = output_top.squeeze(0).numpy().tolist()
                favg_top = _eval_Fmeasure(top_list_output_top, gold_list_top)
                favg_top_list.append(favg_top)
                val_total_loss += val_loss.item()
            
        total_loss = total_loss/len(edesc)
        val_total_loss = val_total_loss/len(val_edesc)
        favg_top = np.mean(favg_top_list)
        if epoch % save_every == 0:
            if favg_top >= best_acc:
                best_acc = favg_top
                best_epoch = epoch
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": kgsumm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss,
                'best_epoch': best_epoch
                }, path.join(directory, "checkpoint_epoch_{}.pt".format(epoch)))
            viz.line([[total_loss, val_total_loss]], [epoch], win='{}_loss_{}_top{}'.format(ds_name, fold, topk), update='append')
            viz.line([[favg_top]], [epoch], win='{}_accuracy_{}_top{}'.format(ds_name, fold, topk), update='append')
            
        print("epoch: {}".format(epoch), "loss train", total_loss, "acc-top{}".format(topk), favg_top, "best epoch", best_epoch)
        
    return best_epoch
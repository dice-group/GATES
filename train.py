#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:05:52 2020
"""
import os # to communicate with operation system 
import os.path as path
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import GATES
from utils import tensor_from_data, tensor_from_weight, _eval_Fmeasure, accuracy
from data_loader import get_data_gold

import math
import time
import psutil


def asHours(s):
	m = math.floor(s / 60)
	h = math.floor(m / 60)
	s -= m * 60
	m -= h * 60
	return '%dh %dm %ds' % (h, m, s)

def mem():
	mem = psutil.cpu_percent()
	print('Current mem usage:')
	print(mem)
	return "Current mem usage: %s \n" % (mem)

def train_iter(ds_name, train_adjs, train_facts, train_labels, val_adjs, val_facts, val_labels, reg, n_epoch, save_every, device, entity_dict, \
                       pred_dict, loss_function, pred2ix_size, hidden_size, pred_emb_dim, ent_emb_dim, lr, dropout, entity2ix_size, hidden_layers, nheads, \
                       word_emb, db_dir, weight_decay, word_emb_calc, topk, file_n, concat_model, print_to):
    if reg == True:
        print("use regularization in training")
    best_epoch_list=[] 
    if not path.exists("models"):
        os.makedirs("models")
    times = []
    start = time.time()  
    print("Current memory", mem())
    valid_epoch_list = []
    arEpochs = []
    losses = {'Training set':[], 'Validation set': []}
    gates = GATES(pred2ix_size, entity2ix_size, pred_emb_dim, ent_emb_dim, device, dropout, hidden_layers, nheads)
    gates.to(device)
        
    for i in range(5):
        arEpochs.append(i)
        if reg:
            optimizer = optim.Adam(gates.parameters(), lr=lr, weight_decay=weight_decay)
        else:    
            optimizer = optim.Adam(gates.parameters(), lr=lr)
        directory = os.path.join(os.getcwd(), path.join("models", "gates_checkpoint-{}-{}-{}".format(ds_name, topk, i)))
        print("Training GATES model on Fold {} on top {} of {} dataset".format(i+1, topk, ds_name))
        best_epoch, total_loss, total_val_loss, total_accuracy, valid_epoch, _, _ = train(gates, ds_name, train_adjs[i], train_facts[i], train_labels[i], \
                           val_adjs[i], val_facts[i], val_labels[i], loss_function, optimizer, n_epoch, save_every, device, entity_dict, pred_dict, reg, \
                           directory, word_emb_calc, i, word_emb, db_dir, topk, file_n, concat_model, print_to)
        best_epoch_list.append(best_epoch)
        valid_epoch_list.append(valid_epoch)
        
        now = time.time()
        print("Iter: {} \n Time {} \n Time(second) {} \n Memeory usage: {} \n Average training loss: {} \n Average validation loss: {} \n Average accuracy {}".format(i, asHours(now-start), now-start, mem(), total_loss, total_val_loss, total_accuracy))
        times.append((time.time()-start)/60)
        losses['Training set'].append(total_loss)
        losses['Validation set'].append(total_val_loss)
        showPlot(arEpochs, losses, "gates_{}_{}".format(ds_name, topk), "Average training vs validation loss")
        with open(print_to, 'a') as f:
            f.write("Times: {}\n".format((time.time()-start)/60))
            f.write("Iteration: {}\n".format(i))
            f.write("Average training loss: {}\n".format(total_loss))
            f.write("Average validation loss: {}\n".format(total_val_loss))
            f.write("\n")
    return valid_epoch_list  
      
# Define training model
def train(gates, ds_name, adj, edesc, label, val_adj, val_edesc, val_label, \
        loss_function, optimizer, n_epoch, save_every, device, entity_dict, pred_dict, reg, directory, word_emb_calc, fold, word_emb, db_dir, topk, file_n, concat_model, print_to):
    if not path.exists(directory):
        os.makedirs(directory)
    best_epoch = 0
    best_acc = 0
    total_avg_loss=[]
    total_avg_val_loss = []
    total_accuracy = []
    stop_valid_epoch = None
    stop_valid_loss = None
    stop_train_loss = None
    arEpochs = []
    losses = {'Training set':[], 'Validation set': []}
    acc_graph = {'Training loss':[], 'Accuracy': []}
    
    for epoch in range(n_epoch):
        arEpochs.append(epoch)
        total_loss = 0
        val_total_loss = 0
        gates.train()
        for i in range(len(edesc)):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            pred_tensor, obj_tensor = tensor_from_data(concat_model, entity_dict, pred_dict, edesc[i], word_emb, word_emb_calc)
            input_tensor = [pred_tensor.to(device), obj_tensor.to(device)]
            target_tensor = tensor_from_weight(len(edesc[i]), edesc[i], label[i]).to(device)
            output_tensor = gates(input_tensor, adj[i])
            loss = loss_function(output_tensor.view(-1), target_tensor.view(-1)).to(device)
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(gates.parameters(), 0.25)
            optimizer.step()
            total_loss += loss.item()

        #gates.eval()
        favg_top_list = []
        acc_list = []
        with torch.no_grad():
            for i in range(len(val_edesc)):
                eid = val_edesc[i][0][0]
                val_pred_tensor, val_obj_tensor = tensor_from_data(concat_model, entity_dict, pred_dict, val_edesc[i], word_emb, word_emb_calc)
                val_input_tensor = [val_pred_tensor.to(device), val_obj_tensor.to(device)]
                val_target_tensor = tensor_from_weight(len(val_edesc[i]), val_edesc[i], val_label[i]).to(device)
                val_output_tensor = gates(val_input_tensor, val_adj[i])
                val_loss = loss_function(val_output_tensor.view(-1), val_target_tensor.view(-1)).to(device)
                
                val_output_tensor = val_output_tensor.view(1, -1).cpu()
                val_target_tensor = val_target_tensor.view(1, -1).cpu()
                (label_top_scores, label_top) = torch.topk(val_target_tensor, topk)
                (output_top_scores, output_top) = torch.topk(val_output_tensor, topk)
                gold_list_top = get_data_gold(db_dir, eid, topk, file_n)
                top_list_output_top = output_top.squeeze(0).numpy().tolist()
                favg_top = _eval_Fmeasure(top_list_output_top, gold_list_top)
                acc = accuracy(output_top.squeeze(0).numpy().tolist(), gold_list_top)
                favg_top_list.append(favg_top)
                acc_list.append(acc)
                val_total_loss += val_loss.item()
            
        total_loss = total_loss/len(edesc)
        
        total_avg_loss.append(total_loss)
        val_total_loss = val_total_loss/len(val_edesc)
        total_avg_val_loss.append(val_total_loss)
        favg_top = np.mean(favg_top_list)
        acc_avg = np.mean(acc_list)
        total_accuracy.append(acc_avg)
        if epoch % save_every == 0:
            if favg_top >= best_acc:
                best_acc = favg_top
                best_epoch = epoch
            if stop_valid_loss == None or val_total_loss<stop_valid_loss:
                stop_valid_loss = val_total_loss
                if stop_valid_epoch != None:
                    if os.path.exists(path.join(directory, "checkpoint_epoch_{}.pt".format(stop_valid_epoch))):
                        os.remove(path.join(directory, "checkpoint_epoch_{}.pt".format(stop_valid_epoch)))
                stop_valid_epoch = epoch
                stop_train_loss = total_loss
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": gates.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss,
                    'best_epoch': best_epoch
                    }, path.join(directory, "checkpoint_epoch_{}.pt".format(epoch)))
                
            #else:
            #    break
        losses['Training set'].append(total_loss)
        losses['Validation set'].append(val_total_loss)
        acc_graph['Training loss'].append(total_loss)
        acc_graph['Accuracy'].append(acc_avg)
        
        print("epoch: {}".format(epoch), "training loss", total_loss, "validation loss: {}".format(val_total_loss), "accuracy: {}".format(acc_avg))
        with open(print_to, 'a') as f:
            f.write("Epoch: {}\n".format(epoch))
            f.write("Training loss: {}\n".format(total_loss))
            f.write("Validation loss: {}\n".format(val_total_loss))
            f.write("\n")
        showPlot(arEpochs, losses, "gates_{}_{}_fold_{}".format(ds_name, topk, fold), "Training vs validation loss")
        showPlot(arEpochs, acc_graph, "acc_gates_{}_{}_fold_{}".format(ds_name, topk, fold), "Training loss vs accuracy")
        
    avg_total_loss = sum(total_avg_loss)/len(total_avg_loss)
    avg_total_val_loss = sum(total_avg_val_loss)/len(total_avg_val_loss) 
    avg_total_accuracy = sum(total_accuracy)/len(total_accuracy)
    return best_epoch, avg_total_loss, avg_total_val_loss, avg_total_accuracy, stop_valid_epoch, stop_train_loss, stop_valid_loss

'''Used to plot the progress of training. Plots the loss value vs. time'''
def showPlot(epochs, losses, fig_name, title):
    colors = ('red','blue')
    x_axis_label = 'Epochs'
    i = 0
    for key, losses in losses.items():
      if len(losses) > 0:
        plt.plot(epochs, losses, label=key, color=colors[i])
        i += 1
    plt.legend(loc='upper left')
    plt.xlabel(x_axis_label)
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(fig_name+'.png')
    plt.close('all')
    
'''prints the current memory consumption'''
import numpy as np
import pandas as pd
import gensim
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import random
import csv

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
        
def gen_minibatch(tokens, labels, mini_batch_size, shuffle= False):
    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
        token_x=[]
        for i in token:
            c = []
            for j in i:
                for k in j:
                    c.append(k)
            token_x.append(c)
        token1 = pad_batch_back(token_x)
        yield token1, Variable(torch.from_numpy(label), requires_grad= False).cuda() 
        

def gen_minibatch_HAN(tokens, labels, mini_batch_size, shuffle= False):
    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
        token1 = pad_batch_HCL_back(token)


        yield token1, Variable(torch.from_numpy(label), requires_grad= False).cuda() 
        
        
def gen_minibatch_HCL(tokens, labels, mini_batch_size, shuffle= False):
    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
        token1 = pad_batch_HCL_forward(token)
        token_x=[]
        for i in token:
            c = []
            for j in i:
                for k in j:
                    c.append(k)
            token_x.append(c)

        token2 = pad_batch_forward(token_x)

        yield token1, token2, Variable(torch.from_numpy(label), requires_grad= False).cuda() 
        

def gen_minibatch_only_HCL(tokens, labels, mini_batch_size, shuffle= False):
    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
        token1 = pad_batch_HCL_forward(token)

        yield token1, Variable(torch.from_numpy(label), requires_grad= False).cuda() 
        
        
def pad_batch_HCL_back(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(x) for x in mini_batch]))
    max_token_len = int(np.max([len(val) for sublist in mini_batch for val in sublist]))
    
        
    idx_sent=[]
    for x in mini_batch:
        if(len(x) >= max_sent_len):
            idx_sent.append(max_sent_len)
        else:
            idx_sent.append(len(x))
    idx_word=[]
    for sublist in mini_batch:
        bb=[]
        aba=0
        for val in sublist:
            aba+=1
            bb.append(len(val))
        while(1):
            bb.append(0)
            aba+=1
            if(aba>=max_sent_len):
            
                break
        idx_word.append(bb)
   
    
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
    for i in range(main_matrix.shape[0]):
        if(idx_sent[i] <= max_sent_len):
            b = max_sent_len-idx_sent[i]
        else:
            b=0
        for j in range(main_matrix.shape[1]):
            if(idx_word[i][j] <= max_token_len):
                c = max_token_len-idx_word[i][j]
            else:
                c=0
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i,b+j,c+k] = mini_batch[i][j][k]
                except IndexError:
                    pass
    return Variable(torch.from_numpy(main_matrix))


def pad_batch_HCL_forward(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(x) for x in mini_batch]))
    max_token_len = int(np.max([len(val) for sublist in mini_batch for val in sublist]))
    if(max_sent_len<2):
        max_sent_len =2
    if(max_sent_len>30):
        max_sent_len =30
    if(max_token_len>50):
        max_token_len =50
    
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i,j,k] = mini_batch[i][j][k]
                except IndexError:
                    pass
    return Variable(torch.from_numpy(main_matrix))



def pad_batch_back(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(mini_batch[x]) for x in range(0,len(mini_batch))]))
    idx_sent=[]
    for x in range(0,len(mini_batch)):
        idx_sent.append(len(mini_batch[x]))

    main_matrix = np.zeros((mini_batch_size, max_sent_len), dtype= np.int)

    for i in range(main_matrix.shape[0]):
        b = max_sent_len-idx_sent[i]
        for k in range(main_matrix.shape[1]):
            try:
                main_matrix[i,b+k] = mini_batch[i][k]
            except IndexError:
                pass
    return Variable(torch.from_numpy(main_matrix))
    
    
def pad_batch_forward(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(mini_batch[x]) for x in range(0,len(mini_batch))]))
    if(max_sent_len<=2):
        max_sent_len=3
    if(max_sent_len >=100):
        max_sent_len = 100
    idx_sent=[]
    for x in range(0,len(mini_batch)):
        idx_sent.append(len(mini_batch[x]))
    main_matrix = np.zeros((mini_batch_size, max_sent_len), dtype= np.int)
    for i in range(main_matrix.shape[0]):
         for k in range(main_matrix.shape[1]):
            try:
                main_matrix[i,k] = mini_batch[i][k]
            except IndexError:
                pass
    return Variable(torch.from_numpy(main_matrix))



def tokenize_all_reviews(reviews_split, embed_lookup):
    reviews_words = reviews_split.split(' ')
    tokenized_reviews = []
    for review in reviews_words:
        ints = []
        for word in review.split(' '):
            if(word==''):
                continue
            if(word=='.'):
                continue
            try:
                idx = embed_lookup.vocab[word].index
                #idx = token2idx[word]
            except: 
                idx = 0
            tokenized_reviews.append(idx)
    return tokenized_reviews




def mydataset_read(file_name, embed_lookup):
    fr = open(file_name, 'r', encoding='utf-8')
    rdrr = csv.reader(fr,  delimiter='\t')
    X_train=list()
    y_train=list()
    for line in rdrr:
        b = int(line[0])
        #print(b)
        if(line[1]==''):
            continue
        a_t=' '
        j=[]
        for e in line[1].split(' '):
            a_t +=e+' '

            c_t = list(tokenize_all_reviews( a_t, embed_lookup))

            if(len(c_t) <=3):
                continue

            if('.' in e):
                b_t = list(tokenize_all_reviews(a_t, embed_lookup))
                j.append(b_t)
                a_t=' '
            elif('?' in e):
                b_t = list(tokenize_all_reviews(a_t, embed_lookup))
                j.append(b_t)
                a_t=' '
            elif('!' in e):
                b_t = list(tokenize_all_reviews(a_t, embed_lookup))
                j.append(b_t)
                a_t=' '

        if(len(j) == 0 ):
            b_t = list(tokenize_all_reviews(line[1],embed_lookup))
            j.append(b_t)
        elif(len(j) > 0 and len(a_t)>=3):
            b_t = list(tokenize_all_reviews(a_t,embed_lookup))
            j.append(b_t)
        if(len(j)==0):
            continue 
        X_train.append(j)
        y_train.append(b)
    fr.close()
    
    return X_train, y_train
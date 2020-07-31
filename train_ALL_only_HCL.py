# @Author : juhyounglee
# @Datetime : 2020/08/01 
# @File : train_ALL_only_HCL.py
# @Last Modify Time : 2020/08/01
# @Contact : juhyounglee@{yonsei.ac.kr}

import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import shutil
import random
from DataLoader.mydatasets_loader import gen_minibatch
from DataLoader.mydatasets_loader import gen_minibatch_HCL
from DataLoader.mydatasets_loader import gen_minibatch_only_HCL
from DataUtils.Common import seed_num
from earlystopping import EarlyStopping
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
torch.manual_seed(seed_num)
random.seed(seed_num)


def train(X_train, y_train, X_dev, y_dev, model,embedding, args):
    if args.cuda:
        print("#####################model.cuda#()################")
        model.cuda()
    
    if args.Adam is True:
        print("Adam Training......")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
    elif args.SGD is True:
        print("SGD Training.......")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay,
                                    momentum=args.momentum_value)
    elif args.Adadelta is True:
        print("Adadelta Training.......")
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.init_weight_decay)
        
    criterion = nn.NLLLoss()

    
    steps = 0
    epoch_step = 0
    model_count = 0
    loss_full = []
    best_accuracy = Best_Result()
    model.train()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    for epoch in range(1, args.epochs+1):
        steps = 0
        
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        loss_epoch = []
        g = gen_minibatch_only_HCL(X_train, y_train, args.batch_size,shuffle= True)
        for tokens1, labels in g:
            tokens1 = embedding(tokens1.long())
            batch_size, max_sents, max_tokens, embed_size = tokens1.size()
            
            optimizer.zero_grad()

            logit = model(tokens1.cuda(), max_sents )
            loss = criterion(logit, labels)
            loss.backward()
            #if args.init_clip_max_norm is not None:
            #    utils.clip_grad_norm_(model.parameters(), max_norm=args.init_clip_max_norm)
            optimizer.step()


            loss_full.append(loss.item())
            loss_epoch.append(loss.item())


        torch.cuda.empty_cache()      
        print ('Average training loss at this epoch..minibatch ' ,  np.mean(loss_epoch))
        model.eval()
        val_loss = []
        g = gen_minibatch_only_HCL(X_dev, y_dev, 4,shuffle= False)
        for tokens1, labels in g:
            tokens1 = embedding(tokens1.long())
            batch_size, max_sents, max_tokens, embed_size = tokens1.size()

            optimizer.zero_grad()

            logit = model(tokens1.cuda(), max_sents )
            loss = criterion(logit, labels)
            #print(loss)
            val_loss.append(loss.data.cpu())
            
       # print("val_loss#:", val_loss)
        vlos =  np.mean(val_loss)
        
 
        print( 'dev Loss at ',epoch, ' is ',vlos)

        torch.cuda.empty_cache()
        early_stopping(vlos, model)
        if early_stopping.early_stop:
            print("Early stopping")
            model_count += 1
            break

        model.train()

      
                
    return model_count


def eval(X_train, y_train, model,embedding, args):
    
    corrects, avg_loss = 0, 0
    p = []
    p2 = []
    l = []
    g = gen_minibatch_only_HCL(X_train, y_train, 4 ,shuffle= False)
    model.load_state_dict(torch.load("./SaveModel/checkpoint_attn.pt"))
    model.eval()
    for tokens1, labels in g:
        tokens1 = embedding(tokens1.long())
        batch_size, max_sents, max_tokens, embed_size = tokens1.size()

        logit = model(tokens1.cuda(),max_sents )
        softmax = nn.Softmax()
        y_pred3 = softmax(logit)
        y_pred33, y_pred1 = torch.max(y_pred3, 1)
        p2.append(np.ndarray.flatten(y_pred3[:, 1].data.cpu().numpy()))
        
        p.append(np.ndarray.flatten(y_pred1.data.cpu().numpy()))
        l.append(np.ndarray.flatten(labels.data.cpu().numpy()))

    p2 = [item for sublist in p2 for item in sublist]
    p = [item for sublist in p for item in sublist]
    l = [item for sublist in l for item in sublist]
    p = np.array(p)
    l = np.array(l)

    recall=0
    precision=0
    ROC  = roc_auc_score(l, p2)
    print("test 갯수 :", len(p))
    precision = precision_score(l, p, average='micro')
    recall = recall_score(l, p, average='micro')
    F1score = f1_score(l,p,average='micro')

    print("F-1score : ", F1score)
    print("AUC : ",ROC )
    
    num_correct = sum(p == l)
 
    print("############")
    print("맞춘갯수 : ", num_correct)
    

            
    print(' Evaluation - acc: {:.4f}'.format(F1score))
    '''
    if test is False:
        if accuracy >= best_accuracy.best_dev_accuracy:
            best_accuracy.best_dev_accuracy = accuracy
            best_accuracy.best_epoch = epoch
            best_accuracy.best_test = True
    if test is True and best_accuracy.best_test is True:
        best_accuracy.accuracy = accuracy

    if test is True:
        print("The Current Best Dev Accuracy: {:.4f}, and Test Accuracy is :{:.4f}, locate on {} epoch.\n".format(
            best_accuracy.best_dev_accuracy, best_accuracy.accuracy, best_accuracy.best_epoch))
    if test is True:
        best_accuracy.best_test = False
    '''

class Best_Result:
    def __init__(self):
        self.best_dev_accuracy = -1
        self.best_accuracy = -1
        self.best_epoch = 1
        self.best_test = False
        self.accuracy = -1

# @Author : juhyounglee
# @Datetime : 2020/08/01 
# @File : train_ALL_HAN.py
# @Last Modify Time : 2020/08/01
# @Contact : juhyounglee@{yonsei.ac.kr}

import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
import shutil
import random
from DataLoader.mydatasets_loader import gen_minibatch
from DataLoader.mydatasets_loader import gen_minibatch_HCL
from DataUtils.Common import seed_num
from earlystopping import EarlyStopping
torch.manual_seed(seed_num)
random.seed(seed_num)


def train(X_train, y_train, X_dev, y_dev, X_test, y_test, model,weights, args):
    if args.cuda:
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
        if(model in 'HCL'):
            g = gen_minibatch_HAN(X_train, y_train, mini_batch_size,shuffle= True)
            for tokens, labels in g:
                embedding = nn.Embedding.from_pretrained(weights)
                tokens = embedding(tokens.long())
               
                optimizer.zero_grad()

                logit = model(tokens)
                loss = criterion(logit, labels)
                loss.backward()
                if args.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(model.parameters(), max_norm=args.init_clip_max_norm)
                optimizer.step()


                loss_full.append(loss.item())
                loss_epoch.append(loss.item())


            torch.cuda.empty_cache()      
            model.eval()
            print ('Average training loss at this epoch..minibatch ' ,  np.mean(loss_epoch))
            vlos = check_val_loss(X_test, y_test, mini_batch_size, sent_attn)
            print( 'Test Loss at ',i, ' is ',vlos)

            torch.cuda.empty_cache()
            early_stopping(vlos, sent_attn)
            if early_stopping.early_stop:
                print("Early stopping")
                model_count += 1
                break

            model.train()
           
    return model_count


def eval(data_iter, model, args, best_accuracy, epoch, test=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target)
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.item()/size
    accuracy = 100.0 * float(corrects)/size
    model.train()
    print(' Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, size))
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


class Best_Result:
    def __init__(self):
        self.best_dev_accuracy = -1
        self.best_accuracy = -1
        self.best_epoch = 1
        self.best_test = False
        self.accuracy = -1



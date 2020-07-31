# @Author : juhyounglee
# @Datetime : 2020/08/01 
# @File : train_ALL_LSTM.py
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
import numpy as np
from DataUtils.Common import seed_num
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

    steps = 0
    model_count = 0
    best_accuracy = Best_Result()
    model.train()
    for epoch in range(1, args.epochs+1):
        steps = 0
        print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        for batch in train_iter:
            feature, target = batch.text, batch.label.data.sub_(1)
            if args.cuda is True:
                feature, target = feature.cuda(), target.cuda()

            target = autograd.Variable(target)  # question 1
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            if args.init_clip_max_norm is not None:
                utils.clip_grad_norm_(model.parameters(), max_norm=args.init_clip_max_norm)
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                            train_size,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                print("\nDev  Accuracy: ", end="")
                eval(dev_iter, model, args, best_accuracy, epoch, test=False)
                print("Test Accuracy: ", end="")
                eval(test_iter, model, args, best_accuracy, epoch, test=True)
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model.state_dict(), save_path)
                if os.path.isfile(save_path) and args.rm_model is True:
                    os.remove(save_path)
                model_count += 1
    return model_count


def eval(data_iter, model, args, best_accuracy, epoch, test=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        target.data.sub_(1)
        if args.cuda is True:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.item()/size
    accuracy = float(corrects)/size * 100.0
    model.train()
    print(' Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss, accuracy, corrects, size))
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



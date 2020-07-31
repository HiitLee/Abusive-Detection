# @Author : juhyounglee
# @Datetime : 2020/08/01 
# @File : main.py
# @Last Modify Time : 2020/08/01
# @Contact : juhyounglee@{yonsei.ac.kr}

import os
import argparse
import logging
import datetime
import Config.config as configurable
import torch
import torchtext.data as data
from models.model_BiLSTM import BiLSTM
from models.model_BiLSTM_Attn import BiLSTM_Attn
from models.model_BiLSTM_Maxpool import BiLSTM_Maxpool
from models.model_CLSTM import CLSTM
from models.model_HAN import HAN
from models.model_HCL import HCL
from models.model_HCL_CLSTM import HCL_CLSTM
from models.model_HCL_CLSTM_CLSTM import HCL_CLSTM_CLSTM
from models.model_LSTM import LSTM
from models.model_LSTM_Atten import LSTM_Attn
from models.model_LSTM_maxpool import LSTM_Maxpool


import train_ALL_CNN_LSTM
import train_ALL_HAN
import train_ALL_HCL
import train_ALL_only_HCL
from DataLoader import mydatasets_self_five
from DataLoader import mydatasets_self_two
from DataLoader import mydatasets_loader

from DataUtils.Load_Pretrained_Embed import load_pretrained_emb_zeros, load_pretrained_emb_avg, load_pretrained_emb_Embedding, load_pretrained_emb_uniform
import torch.nn as nn
import multiprocessing as mu
import shutil
import numpy as np
import random
import gensim
# solve encoding
from imp import reload
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
from DataUtils.Common import seed_num, pad, unk
torch.manual_seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.cuda.manual_seed(seed_num)


def load_preEmbedding():
    # load word2vec
    static_pretrain_embed = None
    pretrain_embed = None
    if config.word_Embedding:
        print("word_Embedding_Path {} ".format(config.word_Embedding_Path))
        path = config.word_Embedding_Path
        print("loading pretrain embedding......")
        embed_lookup = gensim.models.KeyedVectors.load_word2vec_format(path, limit=10000)
        print("pretrain FastText embedding load finished!")
        
    weights = list()
    for i in range(0, len(embed_lookup.wv.vocab)):
        cc = embed_lookup.wv.index2word[i]
        if(i == 0):

            weights.append(np.ndarray.tolist(np.zeros(300,)))
            continue
        try:
            weights.append(np.ndarray.tolist(embed_lookup[cc]))
        except KeyError:
            weights.append(np.ndarray.tolist(np.zeros(300,)))

    weights = np.array(weights, dtype=np.float32)
    weights = torch.from_numpy(weights)

    weights = torch.FloatTensor(weights)

        
    return embed_lookup, weights


def Load_Data():
    """
    load five classification task data and two classification task data
    :return:
    """
    train_iter, dev_iter, test_iter = None, None, None
    if config.FIVE_CLASS_TASK:
        print("Executing 5 Classification Task......")
        if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
            train_iter, dev_iter, test_iter = mrs_five_mui(config.datafile_path, config.name_trainfile, config.name_devfile, config.name_testfile, config.char_data, text_field=config.text_field, label_field=config.label_field,
                                                           static_text_field=config.static_text_field, static_label_field=config.static_label_field, repeat=False, shuffle=config.epochs_shuffle, sort=False)
        else:
            train_iter, dev_iter, test_iter = mrs_five(config.datafile_path, config.name_trainfile, config.name_devfile, config.name_testfile, config.char_data,
                                                       config.text_field, config.label_field, repeat=False, shuffle=config.epochs_shuffle, sort=False)
    elif config.TWO_CLASS_TASK:
        print("Executing 2 Classification Task......")
        if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
            train_iter, dev_iter, test_iter = mrs_two_mui(config.datafile_path, config.name_trainfile, config.name_devfile, config.name_testfile, config.char_data, text_field=config.text_field, label_field=config.label_field,
                                                          static_text_field=config.static_text_field, static_label_field=config.static_label_field, repeat=False, shuffle=config.epochs_shuffle, sort=False)
        else:
            train_iter, dev_iter, test_iter = mrs_two(config.datafile_path, config.name_trainfile, config.name_devfile, config.name_testfile, config.char_data, config.text_field,
                                                      config.label_field, repeat=False, shuffle=config.epochs_shuffle, sort=False)

    return train_iter, dev_iter, test_iter




def define_dict():
    """
     use torchtext to define word and label dict
    """
    print("use torchtext to define word dict......")
    config.text_field = data.Field(lower=True)
    config.label_field = data.Field(sequential=False)
    config.static_text_field = data.Field(lower=True)
    config.static_label_field = data.Field(sequential=False)
    print("use torchtext to define word dict finished.")
    # return text_field


def save_arguments():
    shutil.copytree("./Config", "./snapshot/" + config.mulu + "/Config")


def update_arguments():
    config.lr = config.learning_rate
    config.init_weight_decay = config.weight_decay
    config.init_clip_max_norm = config.clip_max_norm
    config.embed_num = len(config.text_field.vocab)
    config.class_num = len(config.label_field.vocab) - 1
    config.paddingId = config.text_field.vocab.stoi[pad]
    config.unkId = config.text_field.vocab.stoi[unk]
    if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
        config.embed_num_mui = len(config.static_text_field.vocab)
        config.paddingId_mui = config.static_text_field.vocab.stoi[pad]
        config.unkId_mui = config.static_text_field.vocab.stoi[unk]
    # config.kernel_sizes = [int(k) for k in config.kernel_sizes.split(',')]
    print(config.kernel_sizes)
    mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.mulu = mulu
    config.save_dir = os.path.join(""+config.save_dir, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)


def load_model(model):
   
    if model == 'HCL':
        print("loading HCL model......")
        model = HCL(config)
        #shutil.copy("./models/model_HCL.py", "./snapshot/" + config.mulu)
    elif model == 'HCL_CLSTM':
        print("loading HCL_CLSTM model......")
        # model = model_BiLSTM_lexicon.BiLSTM_1(config)
        model = HCL_CLSTM(config)
        #shutil.copy("./models/model_HCL_CLSTM.py", "./snapshot/" + config.mulu)
    elif model == 'HCL_CLSTM_CLSTM':
        print("loading HCL_CLSTM_CLSTM model......")
        model = HCL_CLSTM_CLSTM(config)
        #shutil.copy("./models/model_HCL_CLSTM_CLSTM.py", "./snapshot/" + config.mulu)

    print("model", model)
    if config.cuda is True:
        model = model.cuda()
    return model






def start_train(modelName, model, train_data, train_y_data, dev_data, dev_y_data, weights):
    """
    :function：start train
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    if config.predict is not None:
        label = train_ALL_CNN.predict(config.predict, model, config.text_field, config.label_field)
        print('\n[Text]  {}[Label] {}\n'.format(config.predict, label))
    elif config.test:
        try:
            print(test_iter)
            train_ALL_CNN.test_eval(test_iter, model, config)
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print("\n cpu_count \n", mu.cpu_count())
        embedding = nn.Embedding.from_pretrained(weights)
        torch.set_num_threads(config.num_threads)
        if os.path.exists("./Test_Result.txt"):
            os.remove("./Test_Result.txt")
        if modelName == 'HCL':
            print("HCL training start......")
            model_count = train_ALL_only_HCL.train(train_data, train_y_data, dev_data, dev_y_data,  model,embedding, config)
        elif modelName == 'HCL_CLSTM':
            print("HCL_CLSTM training start......")
            model_count = train_ALL_HCL.train(train_data, train_y_data, dev_data, dev_y_data,  model,embedding, config)
        elif modelName == 'HCL_CLSTM_CLSTM':
            print("HCL_CLSTM_CLSTM training start......")
            model_count = train_ALL_HCL.train(train_data, train_y_data, dev_data, dev_y_data,  model,embedding, config)

        print("Model_count", model_count)
        resultlist = []
        if os.path.exists("./Test_Result.txt"):
            file = open("./Test_Result.txt")
            for line in file.readlines():
                if line[:10] == "Evaluation":
                    resultlist.append(float(line[34:41]))
            result = sorted(resultlist)
            file.close()
            file = open("./Test_Result.txt", "a")
            file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
            file.write("\n")
            file.close()
            shutil.copy("./Test_Result.txt", "./snapshot/" + config.mulu + "/Test_Result.txt")
            
            
def start_eval(modelName, model, test_data, test_y_data, weights):
    """
    :function：start train
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    embedding = nn.Embedding.from_pretrained(weights)
    torch.set_num_threads(config.num_threads)
    if modelName == 'HCL':
        print("HCL training start......")
        model_count = train_ALL_only_HCL.eval(test_data, test_y_data, model,embedding, config)
    elif modelName == 'HCL_CLSTM':
        print("HCL_CLSTM training start......")
        model_count = train_ALL_HCL.eval(test_data, test_y_data, model,embedding, config)
    elif modelName == 'HCL_CLSTM_CLSTM':
        print("HCL_CLSTM_CLSTM training start......")
        model_count = train_ALL_HCL.eval(test_data, test_y_data, model,embedding, config)


    print("Model_count", model_count)
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
        file.write("\n")
        file.close()
        shutil.copy("./Test_Result.txt", "./snapshot/" + config.mulu + "/Test_Result.txt")

def main(mode, modelName):
    """
        main function
    """
    print("mode : ", mode)
    print("model : ", modelName)
    embed_lookup, weights = load_preEmbedding()
    if(mode == "train"):
        for i in range(0, 5):
            file_dev = "./toxic2/toxic_dev"+str(i)+".tsv"
            file_train = "./toxic2/toxic_train"+str(i)+".tsv"
            train_data, train_y_data = mydatasets_loader.mydataset_read(file_train, embed_lookup)
            dev_data, dev_y_data = mydatasets_loader.mydataset_read(file_dev, embed_lookup)
            

            train_data,train_y_data  = np.array(train_data), np.array(train_y_data)
            dev_data,dev_y_data  = np.array(dev_data), np.array(dev_y_data)
            

            print("train_data#:", train_data.shape)
            print("**************training data loading **************")

            model = load_model(modelName)
            start_train(modelName, model.cuda(), train_data, train_y_data, dev_data, dev_y_data, weights)
            file_test = "./toxic2/test_toxic.tsv"
            test_data, test_y_data = mydatasets_loader.mydataset_read(file_test, embed_lookup)
            test_data,test_y_data  = np.array(test_data), np.array(test_y_data)
            print("test_data#:", test_data.shape)
            print("**************evaluating load finish**************")
            start_eval(modelName, model , test_data, test_y_data, weights)
            
    elif(mode == "eval"):
        file_test = "./toxic2/test_toxic.tsv"
        test_data, test_y_data = mydatasets_loader.mydataset_read(file_test, embed_lookup)
        test_data,test_y_data  = np.array(test_data), np.array(test_y_data)
        print("test_data#:", test_data.shape)
        print("**************evaluating load finish**************")
        model = load_model(model)
        start_eval(model , test_data, test_y_data, weights)



if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Neural Networks")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    
    parser.add_argument('--mode', default="train", choices=['train','eval'],help='What mode?')
    
    parser.add_argument('--model', default="HCL_CLSTM", choices=['HCL', 'HCL_CLSTM','HCL_CLSTM_CLSTM'],help='What model?')
    
    args = parser.parse_args()

    config = configurable.Configurable(config_file=args.config_file)
    if config.cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())
    main(args.mode, args.model)






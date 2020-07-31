# @Author : juhyounglee
# @Datetime : 22020/08/01 
# @File : model_HCL.py
# @Last Modify Time : 2020/08/01
# @Contact : juhyounglee@{yonsei.ac.kr}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from DataUtils.Common import seed_num
torch.manual_seed(seed_num)
random.seed(seed_num)


"""
Neural Networks model : Hierarchical C-SLTM
"""

class wordCLSTM(nn.Module):
    def __init__(self, args):
        super(wordCLSTM, self).__init__()

        D = args.embed_dim
        C = args.class_num
        Ci = 1
        F = args.clstm_filter_size
        H = args.clstm_hidden_size
        self.num_layers = args.lstm_num_layers
        Wk = args.word_kernel
        KK=[]
        for K in Wk:
            KK.append( K + 1 if K % 2 == 0 else K)
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(Ci, F, (k, D), padding=(k//2,0)) 
            for k in KK])
        self.bilstm = nn.LSTM(F, H, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)
        
    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)
        return cnn_x

    def forward(self, input):
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
       
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)
        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        return bilstm_out.unsqueeze(0)

    
    
    
class sentCLSTM(nn.Module):
    
    def __init__(self, args):
        super(sentCLSTM, self).__init__()
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        F = args.clstm_filter_size
        H = args.clstm_hidden_size
        Sk = args.sent_kernel
        self.num_layers = args.lstm_num_layers
        KK=[]
        for K in Sk:
            KK.append( K + 1 if K % 2 == 0 else K)
            
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(Ci, F, (k, H*2), padding=(k//2,0)) 
            for k in KK])
        self.bilstm = nn.LSTM(F, H, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)
        self.hidden2label1 = nn.Linear(H*2, C)

    def conv_and_pool(self, input, conv):
        cnn_x = conv(input)
        cnn_x = F.relu(cnn_x)
        cnn_x = cnn_x.squeeze(3)
        return cnn_x
    
    def forward(self, input):
        input = torch.transpose(input, 1,0)
        embeds = input.unsqueeze(1)
        embeds = self.dropout(embeds)
        cnn_x = [self.conv_and_pool(embeds, conv) for conv in self.convs_1d]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = torch.transpose(cnn_x, 1,2)
        bilstm_out,(final_hidden_state, final_cell_state) =self.bilstm(cnn_x)
        bilstm_out = torch.transpose(bilstm_out, 2,1)
        
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = self.hidden2label1(bilstm_out)

        return bilstm_out
    
    
class HCL(nn.Module):
    
    def __init__(self, args):
        super(HCL, self).__init__()
        self.wordCLSTM = wordCLSTM(args)
        self.senCLSTM = sentCLSTM(args)

      

    def forward(self, embed, max_sents):
        s = None
        for i in range(max_sents):
            _s = self.wordCLSTM(embed[:,i,:])
            if(s is None):
                s = _s
            else:
                s = torch.cat((s,_s),0)    

        logits = self.senCLSTM(s)
     
        return F.log_softmax(logits, dim=1)
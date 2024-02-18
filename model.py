import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
from model_erc import emoplex
import torch.fft


class TextCNN(nn.Module):
    def __init__(self, input_dim, emb_size=128, in_channels=1, out_channels=128, kernel_heights=[3,4,5], dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, emb_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size() #dia_len, utt_len, batch_size, feat_dim
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        return embd

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits,-1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss



def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def simple_batch_graphify(features, lengths, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)  

    if not no_cuda:
        node_features = node_features.cuda()
    return node_features, None, None, None, None



class MMGatedAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, att_type='general'):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.dropouta = nn.Dropout(0.5)
        self.dropoutv = nn.Dropout(0.5)
        self.dropoutl = nn.Dropout(0.5)
        if att_type=='av_bg_fusion':
            self.transform_al = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            self.transform_vl = nn.Linear(mem_dim*2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type=='general':
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_av = nn.Linear(mem_dim*3,1)
            self.transform_al = nn.Linear(mem_dim*3,1)
            self.transform_vl = nn.Linear(mem_dim*3,1)

    def forward(self, a, v, l, modals=None):
        a = self.dropouta(a) if len(a) !=0 else a
        v = self.dropoutv(v) if len(v) !=0 else v
        l = self.dropoutl(l) if len(l) !=0 else l
        if self.att_type == 'av_bg_fusion':
            if 'a' in modals:
                fal = torch.cat([a, l],dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa*(self.scalar_al(a))
            if 'v' in modals:
                fvl = torch.cat([v, l],dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv*(self.scalar_vl(v))
            if len(modals) == 3:
                hmf = torch.cat([l,hma,hmv], dim=-1)
            elif 'a' in modals:
                hmf = torch.cat([l,hma], dim=-1)
            elif 'v' in modals:
                hmf = torch.cat([l,hmv], dim=-1)
            return hmf
        elif self.att_type == 'general':
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 'l' in modals else l

            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a,v,a*v],dim=-1)))
                h_av = z_av*ha + (1-z_av)*hv
                if 'l' not in modals:
                    return h_av
            if 'a' in modals and 'l' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a,l,a*l],dim=-1)))
                h_al = z_al*ha + (1-z_al)*hl
                if 'v' not in modals:
                    return h_al
            if 'v' in modals and 'l' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v,l,v*l],dim=-1)))
                h_vl = z_vl*hv + (1-z_vl)*hl
                if 'a' not in modals:
                    return h_vl
            return torch.cat([h_av, h_al, h_vl],dim=-1)

class Model(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, model_hidden_size, n_speakers, max_seq_len,  
                 n_classes=7, dropout_rec=0.5, dropout=0.5, no_cuda=False, model_type='emoplex',
                 D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='IEMOCAP',
                 norm='LN2'):
        
        super(Model, self).__init__()

        self.base_model = base_model
        self.no_cuda = no_cuda
        self.model_type=model_type
        self.dropout = dropout
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.att_type = att_type
        self.normBNa = nn.BatchNorm1d(1024, affine=True)
        self.normBNb = nn.BatchNorm1d(1024, affine=True)
        self.normBNc = nn.BatchNorm1d(1024, affine=True)
        self.normBNd = nn.BatchNorm1d(1024, affine=True)
        self.normLNa = nn.LayerNorm(1024, elementwise_affine=True)
        self.normLNb = nn.LayerNorm(1024, elementwise_affine=True)
        self.normLNc = nn.LayerNorm(1024, elementwise_affine=True)
        self.normLNd = nn.LayerNorm(1024, elementwise_affine=True)
        self.norm_strategy = norm
        if self.att_type == 'gated' or self.att_type == 'concat_subsequently' or self.att_type == 'concat_DHT' or self.att_type == 'concat_ori':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.use_bert_seq = False
        self.dataset = dataset

        #self.complex_weight = torch.nn.Parameter(torch.randn(1, 110//2 + 1, D_g, 2, dtype=torch.float32) * 0.02)
        #self.layer_norm = nn.LayerNorm(D_g, elementwise_affine=True)

        if self.base_model == 'LSTM':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'al':
                    hidden_ = 150
                elif ''.join(self.modals) == 'vl':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 'a' in self.modals:
                    hidden_a = D_g
                    self.linear_a = nn.Linear(D_m_a, hidden_a)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = D_g
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                if 'l' in self.modals:
                    hidden_l = D_g
                    if self.use_bert_seq:
                        self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                    else:
                        self.linear_l = nn.Linear(D_m, hidden_l)
                    self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
                    #self.lstm_l = CLSTM(in_channels = hidden_l, hidden_size = D_g//2, num_layers=2)
                    #self.mu_l = nn.Parameter(torch.ones(110, 1, 1))

        elif self.base_model == 'GRU':
            #self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                if self.av_using_lstm:
                    self.gru_a = nn.GRU(input_size=hidden_a, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                if self.av_using_lstm:
                    self.gru_v = nn.GRU(input_size=hidden_v, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                hidden_l = D_g
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.gru_l = nn.GRU(input_size=hidden_l, hidden_size=D_g//2, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'Transformer':
            if 'a' in self.modals:
                hidden_a = D_g
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                self.trans_a = nn.TransformerEncoderLayer(d_model=hidden_a, nhead=4)
            if 'v' in self.modals:
                hidden_v = D_g
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                self.trans_v = nn.TransformerEncoderLayer(d_model=hidden_v, nhead=4)
            if 'l' in self.modals:
                hidden_l = D_g
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.trans_l = nn.TransformerEncoderLayer(d_model=hidden_l, nhead=4)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError 



        if self.model_type=='emoplex':
            self.emo_model = emoplex(a_dim=D_g, v_dim=D_g, l_dim=D_g, n_dim=D_g, nlayers=64, nhidden=model_hidden_size, nclass=n_classes, 
                                        dropout=self.dropout, return_feature=self.return_feature, n_speakers=n_speakers, modals=self.modals)
            print("construct "+self.model_type)
        elif self.model_type=='None':
            if not self.multi_modal:
                self.graph_net = nn.Linear(2*D_e, n_classes)
            else:
                if 'a' in self.modals:
                    self.graph_net_a = nn.Linear(2*D_e, model_hidden_size)
                if 'v' in self.modals:
                    self.graph_net_v = nn.Linear(2*D_e, model_hidden_size)
                if 'l' in self.modals:
                    self.graph_net_l = nn.Linear(2*D_e, model_hidden_size)
            print("construct Bi-LSTM")
        else:
            print("There are no such kind of model")
        if self.multi_modal:
            #self.gatedatt = MMGatedAttention(D_g + model_hidden_size, model_hidden_size, att_type='general')
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(model_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                self.smax_fc = nn.Linear((model_hidden_size)*1, n_classes)
            elif self.att_type == 'concat_DHT':
                self.smax_fc = nn.Linear((model_hidden_size*2)*len(self.modals), n_classes)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), model_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, model_hidden_size)
            elif self.att_type == 'concat_ori':
                self.smax_fc = nn.Linear(342+1582+512, n_classes)
            else:
                self.smax_fc = nn.Linear(model_hidden_size, n_classes)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):

        #=============roberta features
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()
        if self.norm_strategy == 'LN':
            r1 = self.normLNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normLNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normLNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normLNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        elif self.norm_strategy == 'BN':
            r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        elif self.norm_strategy == 'LN2':
            norm2 = nn.LayerNorm((seq_len, feature_dim), elementwise_affine=False)
            r1 = norm2(r1.transpose(0, 1)).transpose(0, 1)
            r2 = norm2(r2.transpose(0, 1)).transpose(0, 1)
            r3 = norm2(r3.transpose(0, 1)).transpose(0, 1)
            r4 = norm2(r4.transpose(0, 1)).transpose(0, 1)
        else:
            pass

        U = (r1 + r2 + r3 + r4)/4
        #U = torch.cat((textf,acouf),dim=-1)
        #=============roberta features


        if self.base_model == 'LSTM':
            if not self.multi_modal:
                U = self.linear_(U)
                emotions, hidden = self.lstm(U)
            else:
                if 'a' in self.modals:
                    U_a = self.linear_a(U_a)
                    if self.av_using_lstm:  #worse than v
                        emotions_a, hidden_a = self.lstm_a(U_a)
                    else:
                        emotions_a = U_a
                if 'v' in self.modals:
                    U_v = self.linear_v(U_v)
                    if self.av_using_lstm:
                        emotions_v, hidden_v = self.lstm_v(U_v)
                    else:
                        emotions_v = U_v
                if 'l' in self.modals:
                    if self.use_bert_seq:
                        U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                        U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
                    else:
                        U = self.linear_l(U)
                    emotions_l, hidden_l = self.lstm_l(U)
                    #emotions_l, hidden_l = self.lstm_l(torch.complex(self.mu_l[:U.size(0)]*torch.cos(U),self.mu_l[:U.size(0)]*torch.sin(U)))

        elif self.base_model == 'GRU':
            #emotions, hidden = self.gru(U)
            if 'a' in self.modals:
                U_a = self.linear_a(U_a)
                if self.av_using_lstm:
                    emotions_a, hidden_a = self.gru_a(U_a)
                else:
                    emotions_a = U_a
            if 'v' in self.modals:
                U_v = self.linear_v(U_v)
                if self.av_using_lstm:
                    emotions_v, hidden_v = self.gru_v(U_v)
                else:
                    emotions_v = U_v
            if 'l' in self.modals:
                if self.use_bert_seq:
                    U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                    U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
                else:
                    U = self.linear_l(U)
                #self.gru_l.flatten_parameters()
                emotions_l, hidden_l = self.gru_l(U)

        elif self.base_model == 'Transformer':
            if 'a' in self.modals:
                U_a = self.linear_a(U_a)
                if self.av_using_lstm:
                    emotions_a = self.trans_a(U_a)
                else:
                    emotions_a = U_a
            if 'v' in self.modals:
                U_v = self.linear_v(U_v)
                if self.av_using_lstm:
                    emotions_v = self.trans_v(U_v)
                else:
                    emotions_v = U_v
            if 'l' in self.modals:
                if self.use_bert_seq:
                    U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                    U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
                else:
                    U = self.linear_l(U)
                emotions_l = self.trans_l(U)
        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        if not self.multi_modal:
            features, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                #features_a, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
                features_a = emotions_a
            else:
                features_a = []
            if 'v' in self.modals:
                #features_v, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
                features_v = emotions_v
            else:
                features_v = []
            if 'l' in self.modals:
                #features_l, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
                features_l = emotions_l
            else:
                features_l = []

        if self.model_type=='emoplex':
            emotions_feat = self.emo_model(features_a, features_v, features_l, seq_lengths, qmask, umask)
            emotions_feat, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_feat, seq_lengths, self.no_cuda)####
            emotions_feat = self.dropout_(emotions_feat)
            emotions_feat = nn.ReLU()(emotions_feat)
            #print('emotions_feat:',emotions_feat.shape)
            #print('self.smax_fc:',self.smax_fc)  #3072->6
            log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)

        else:
            print("There are no such kind of graph")        
        return log_prob


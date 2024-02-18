import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Parameter, init
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
import ipdb
from itertools import permutations
import torch.fft

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)

class emoplex(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, return_feature,
                n_speakers=2, modals=['a','v','l']):
        super(emoplex, self).__init__()
        self.return_feature = return_feature  #True
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.modals = modals
        self.use_position = False
        self.nhidden = nhidden
        self.n_dim = n_dim
        #------------------------------------    

        self.fc1 = nn.Linear(n_dim, nhidden)      
        self.fc2 = nn.Linear(n_dim, nhidden)      
        self.act_fn = nn.ReLU()
        
        #self.emb_r = nn.Linear(n_dim*3, nhidden*3)
        #self.emb_i = nn.Linear(n_dim*3, nhidden*3)
        
        self.shifta = nn.Parameter(torch.randn(1, 1, n_dim))
        self.shiftv = nn.Parameter(torch.randn(1, 1, n_dim))
        #self.shiftl = nn.Parameter(torch.randn(1, 1, n_dim))
        self.mu_a = nn.Parameter(torch.ones(110, 1, 1))
        self.mu_v = nn.Parameter(torch.ones(110, 1, 1))
        self.mu_l = nn.Parameter(torch.ones(110, 1, 1))
        self.turna = nn.Parameter(torch.rand(110, 1, n_dim))
        self.turnb = nn.Parameter(torch.rand(110, 1, n_dim))
        #self.turn = nn.Parameter(torch.rand(110, 1, n_dim))

    def forward(self, a, v, l, dia_len, qmask, umask):
        ca=torch.complex(self.mu_a[:a.size(0)]*torch.cos(a),self.mu_a[:a.size(0)]*torch.sin(a))
        cv=torch.complex(self.mu_v[:v.size(0)]*torch.cos(v),self.mu_v[:v.size(0)]*torch.sin(v))
        cl=torch.complex(self.mu_l[:l.size(0)]*torch.cos(l),self.mu_l[:l.size(0)]*torch.sin(l))
        self.shiftar, self.shiftai = torch.cos(self.shifta), torch.sin(self.shifta)
        self.shiftvr, self.shiftvi = torch.cos(self.shiftv), torch.sin(self.shiftv)
        #self.shiftlr, self.shiftli = torch.cos(self.shiftl), torch.sin(self.shiftl)

        tmp_ar = ca.real*self.shiftar - ca.imag*self.shiftai
        tmp_ai = ca.real*self.shiftai + ca.imag*self.shiftar
        #tmp_a = ca*torch.complex(self.shiftar, self.shiftai)

        tmp_vr = cv.real*self.shiftvr - cv.imag*self.shiftvi
        tmp_vi = cv.real*self.shiftvi + cv.imag*self.shiftvr
        #tmp_v = cv*torch.complex(self.shiftvr, self.shiftvi)

        #tmp_lr = cl.real*self.shiftlr - cl.imag*self.shiftli
        #tmp_li = cl.real*self.shiftli + cl.imag*self.shiftlr

        la = cl + torch.complex(tmp_ar, tmp_ai)
        lav = la + torch.complex(tmp_vr, tmp_vi)
        features = lav 

        #-------------------
        TMPr, TMPi = [], []
        pA, pB = torch.complex(torch.ones_like(features[0].real),torch.ones_like(features[0].imag)), \
                torch.complex(torch.ones_like(features[0].real),torch.ones_like(features[0].imag))
        for idx, i in enumerate(features):
            if idx == 0:
                temp = i
                TMPr.append(temp.real)
                TMPi.append(temp.imag)
                pA = (1-qmask[idx][:,0].unsqueeze(-1))*pA + qmask[idx][:,0].unsqueeze(-1)*i
                pB = (1-qmask[idx][:,1].unsqueeze(-1))*pB + qmask[idx][:,1].unsqueeze(-1)*i
            else:
                turna = pA*torch.complex(torch.cos(self.turna[idx]),torch.sin(self.turna[idx]))#pA*self.turna[idx]
                turnb = pB*torch.complex(torch.cos(self.turnb[idx]),torch.sin(self.turnb[idx]))#pB*self.turnb[idx]
                turn = qmask[idx][:,0].unsqueeze(-1)*turna + qmask[idx][:,1].unsqueeze(-1)*turnb
                turn = turn.where(turn==0, turn/(turn.abs()+1e-9))
                temp = i + turn
                pA = (1-qmask[idx][:,0].unsqueeze(-1))*pA + qmask[idx][:,0].unsqueeze(-1)*temp
                pB = (1-qmask[idx][:,1].unsqueeze(-1))*pB + qmask[idx][:,1].unsqueeze(-1)*temp
                TMPr.append(temp.real)
                TMPi.append(temp.imag)

        features = torch.complex(torch.stack(TMPr),torch.stack(TMPi))
        #-------------------
        
        
        xr, xi = self.fc1(features.real)-self.fc2(features.imag), self.fc1(features.imag)+self.fc2(features.real)
        features = xr + xi 
        return features


    
class MAG(nn.Module):
    def __init__(self, n_dim, nhidden, dropout_prob):
        super(MAG, self).__init__()

        self.W_hv = nn.Linear(n_dim + n_dim, nhidden)
        self.W_ha = nn.Linear(n_dim + n_dim, nhidden)

        self.W_v = nn.Linear(n_dim, nhidden)
        self.W_a = nn.Linear(n_dim, nhidden)
        self.beta_shift = 5e-4

        self.LayerNorm = nn.LayerNorm(nhidden)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).cuda()
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).cuda()

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = acoustic_vis_embedding + text_embedding

        return embedding_output

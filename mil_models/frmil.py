import torch
import torch.nn as nn
import torch.nn.functional as F
import math, numpy as np

import sys, argparse, os, copy, itertools, glob, datetime
os.environ['CUDA_VISIBLE_DEVICES']='6'

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V     = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_Q, dim_V)
        self.fc_v = nn.Linear(dim_Q, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, inst_mode=False):
        Q    = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        
        if inst_mode: #[N,K,D] --> [N,K,D]
            return O
        else:# bag mode [N,1,D] --> [N,D]
            return O.squeeze(1)


class FRMIL(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dataset   = args.dataset #cm16/msi
        self.num_outputs = args.num_classes #2,
        dim_hidden       = args.feats_size #512,
        num_heads        = args.n_heads   #1,
        self.k           = 1
        
        
        self.enc = nn.Sequential(
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )
        
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, dim_hidden))
        nn.init.xavier_uniform_(self.cls_token)
        
        self.conv_head = torch.nn.Conv2d(dim_hidden, dim_hidden, 3, 1, 3//2,  groups=dim_hidden)  
        torch.nn.init.xavier_uniform_(self.conv_head.weight)
        
        self.selt_att  = MAB(dim_hidden, dim_hidden, num_heads)
        self.fc   = nn.Sequential(
            nn.Linear(dim_hidden, self.num_outputs),
        )
        
        self.mode = 0
        
    def recalib(self, inputs, option='max'):
        A1  = []
        Q   = []
        bs  = inputs.shape[0]
        if option == 'mean':
            Q  = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):
                a1  = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                #print(a1.shape)
                _, m_indices = torch.sort(a1, 0, descending=True)
                #print(m_indices.shape)
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(self.k):
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
                    else:
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[len_i - i_q, :])
                    feat_q.append(feats)
                    
                feats = torch.stack(feat_q)
        
                A1.append(a1.squeeze(1))
                Q.append(feats.mean(0))
                
            A1 = torch.stack(A1)
            Q  = torch.stack(Q)
            return A1, Q
            
    def forward(self, inputs):
        
        inputs = inputs.unsqueeze(0) 
        if self.mode == 1:
            # used in feature magnitude analysis
            return self.selt_att(inputs,inputs,True)
        
        A1, Q = self.recalib(inputs, 'max')
        
        ##################################################################
        # shift features
        if self.dataset == 'msi':
            i_shift = inputs
        else:
            inputs  = F.relu(inputs - Q)
            i_shift = inputs
        ##################################################################
        
        ##################################################################
        #---->pad inputs 
        H          = inputs.shape[1] # Number of Instances in Bag    
        _H, _W     = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        inputs     = torch.cat([inputs, inputs[:,:add_length,:]],dim = 1) #[B, N+29, D//2 ]
        
    
        #---->cls_token
        B          = inputs.shape[0] # Batch Size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        inputs     = torch.cat((cls_tokens, inputs), dim=1)
        
        # CNN Position Learning
        B, _, C  = inputs.shape
        cls_token, feat_token = inputs[:, 0], inputs[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, _H, _W)
        cnn_feat = self.conv_head(cnn_feat) + cnn_feat
        x        = cnn_feat.flatten(2).transpose(1, 2)
        x        = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        ##################################################################
        
        ##################################################################
        # Bag pooling with critical feature
        bag = self.selt_att(Q, x) 
        out = self.fc(bag)
        ##################################################################
        
        
        # if self.training:
        return out, i_shift, A1 
        # else:
        #     return out

class FeatMag(nn.Module):
    
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        
    def forward(self, feat_pos, feat_neg, w_scale=1.0):
        
        loss_act = self.margin - torch.norm(torch.mean(feat_pos, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_neg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um/w_scale
               
        
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train F-MIL')
    
#     parser.add_argument('--dataset', default='her2', type=str, help='Dataset folder name')
#     parser.add_argument('--model', default='frmil', type=str, help='MIL model [dsmil]')
#     parser.add_argument('--num_classes', default=2, type=int, help='MIL model [dsmil]')
#     args = parser.parse_args()
    
#     args.num_out    = 1
#     args.n_heads    = 8
#     args.feats_size = 512

#     # ''' define model '''
#     model = FRMIL(args).cuda()
#     model.eval()
    
#     inp = torch.randn((4,30000,512)).cuda()
#     out = model(inp)
    
#     print('-*-'*20)
#     print(f'in  : {inp.shape}')
#     print(f'out : {out.shape}')
#     print('-*-'*20)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_Q, dim_V)
        self.fc_v = nn.Linear(dim_Q, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()

    def forward(self, Q, K, inst_mode=False):
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=2), dim=0)
        K_ = torch.cat(K.split(dim_split, dim=2), dim=0)
        V_ = torch.cat(V.split(dim_split, dim=2), dim=0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), dim=0), dim=2)
        O = self.ln1(O + F.relu(self.fc_o(self.ln0(O))))

        return O if inst_mode else O.squeeze(1)


class FRMIL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dataset = args.dataset
        self.num_outputs = args.num_classes
        dim_hidden = args.feats_size
        num_heads = args.n_heads
        self.k = 1
        self.mode = 0

        self.enc = nn.Sequential(
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )

        self.cls_token = nn.Parameter(torch.Tensor(1, 1, dim_hidden))
        nn.init.xavier_uniform_(self.cls_token)

        self.conv_head = nn.Conv2d(dim_hidden, dim_hidden, 3, 1, 1, groups=dim_hidden)
        nn.init.xavier_uniform_(self.conv_head.weight)

        self.selt_att = MAB(dim_hidden, dim_hidden, num_heads)
        self.fc = nn.Linear(dim_hidden, self.num_outputs)

    def recalib(self, inputs, option='max'):
        B, N, D = inputs.shape
        if option == 'mean':
            Q = inputs.mean(dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q

        A1_list, Q_list = [], []
        for b in range(B):
            a1 = self.enc(inputs[b])  # (N, 1)
            _, indices = torch.sort(a1.squeeze(), descending=True)
            selected_feats = torch.stack([inputs[b][indices[i]] for i in range(self.k)])
            Q_list.append(selected_feats.mean(0))
            A1_list.append(a1.squeeze())

        return torch.stack(A1_list), torch.stack(Q_list)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(0)  # make batch size = 1
        if self.mode == 1:
            return self.selt_att(inputs, inputs, inst_mode=True)

        A1, Q = self.recalib(inputs, option='max')

        if self.dataset != 'msi':
            inputs = F.relu(inputs - Q)

        B, N, D = inputs.shape
        H = W = int(np.ceil(np.sqrt(N)))
        pad_len = H * W - N

        if pad_len > 0:
            padding = inputs[:, :pad_len, :]
            inputs = torch.cat([inputs, padding], dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        inputs = torch.cat([cls_tokens, inputs], dim=1)

        cls_token, feat_token = inputs[:, 0], inputs[:, 1:]
        feat_map = feat_token.transpose(1, 2).reshape(B, D, H, W)
        feat_map = self.conv_head(feat_map) + feat_map
        flat_feat = feat_map.flatten(2).transpose(1, 2)

        x = torch.cat([cls_token.unsqueeze(1), flat_feat], dim=1)
        bag = self.selt_att(Q, x)
        out = self.fc(bag)

        return out, inputs, A1


class FeatMag(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, feat_pos, feat_neg, w_scale=1.0):
        pos_norm = torch.norm(feat_pos.mean(dim=1), p=2, dim=1)
        neg_norm = torch.norm(feat_neg.mean(dim=1), p=2, dim=1)

        loss_act = torch.clamp(self.margin - pos_norm, min=0)
        loss = ((loss_act + neg_norm) ** 2).mean()
        return loss / w_scale

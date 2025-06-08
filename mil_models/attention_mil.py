import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys


def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 512, D = 256, dropout = True, n_classes = 1,batch_norm=False):
        super(Attn_Net_Gated, self).__init__()
       
        self.attention_a = [
            nn.Linear(L, D),
            
            nn.Tanh()]
        
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
            ]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

        
        
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class MIL_Attention_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = True, n_classes = 2, feats_size=512, batch_norm=False):
        super(MIL_Attention_fc, self).__init__()
        self.size_dict = {"small": [feats_size, 256, 128], "big": [feats_size, 384, 256]}
        size = self.size_dict[size_arg]
        if batch_norm == False:
            fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        else:
            fc = [nn.Linear(size[0], size[1]), nn.BatchNorm1d(size[1]), nn.ReLU()]


        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1,batch_norm=batch_norm)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)
                
    def relocate(self, device_id=None):
        if device_id is not None:
            device = 'cuda:{}'.format(device_id)
            self.attention_net = self.attention_net.to(device)
            self.classifier = self.classifier.to(device)
            self.device = device
        
        else:
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.attention_net = self.attention_net.to(device)
            self.classifier = self.classifier.to(device)
            self.device = None


    def forward(self, h, return_features=False, attention_only=False):

        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        logits  = self.classifier(M) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return Y_prob
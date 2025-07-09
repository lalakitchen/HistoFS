import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
''' Own library '''
own_dir = os.path.join('../')
sys.path.append(own_dir)

from utils.style_augment import FeatureStatistics


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
    def __init__(self, gate=True, size_arg="small", dropout=True, n_classes=2, feats_size=512, batch_norm=False):
        super(MIL_Attention_fc, self).__init__()
        self.size_dict = {"small": [feats_size, 256, 128], "big": [feats_size, 384, 256]}
        size = self.size_dict[size_arg]

        # Define the early layer with optional Batch Normalization
        fc = []
        if batch_norm:
            fc.append(nn.BatchNorm1d(size[0]))
        fc.append(nn.Linear(size[0], size[1]))
        fc.append(nn.ReLU())

        if dropout:
            fc.append(nn.Dropout(0.25))

        # Define the attention network
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1, batch_norm=batch_norm)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        # Define the classifier
        self.classifier = nn.Linear(size[1], n_classes)

        # Feature statistics for style extraction
        self.feature_stats = FeatureStatistics()

        # Initialize weights
        initialize_weights(self)

    def relocate(self, device_id=None):
        if device_id is not None:
            device = f'cuda:{device_id}'
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attention_net = self.attention_net.to(device)
        self.classifier = self.classifier.to(device)
        self.device = device

    def forward_recalibrate(self, h, new_h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)
        new_A, new_h = self.attention_net(new_h)

        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, new_h)
        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return Y_prob, Y_hat, A

    def forward(self, h, return_features=False, attention_only=False):
        random_index = torch.randint(0, h.shape[0], (1,)).item()
        random_style = h[random_index]

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return Y_prob,Y_hat,logits,  M

    def auth_module(self, A_old, A_novel):
        A_old_flat = A_old.view(1, -1)  
        A_novel_flat = A_novel.view(1, -1)
        auth_score = 1- (0.4 * F.cosine_similarity(A_old_flat, A_novel_flat, dim=1))
        A_align = auth_score * A_novel
        return A_align

        

    def forward_auth(self, h, h_old, return_features=False, attention_only=False):
        # Select a random style and compute its statistics
        random_index = torch.randint(0, h.shape[0], (1,)).item()
        random_style = h[random_index]
        mu, var = self.feature_stats.get_statistics(random_style)

        # Apply attention network to current and old inputs
        A, h = self.attention_net(h)
        A = A.transpose(1, 0)  # Transpose A for consistency

        A_old, h_old = self.attention_net(h_old)
        A_old = A_old.transpose(1, 0)

        # Reshape attention weights for cosine similarity comparison
        A_old_flat = A_old.view(1, -1)  
        A_flat = A.view(1, -1)

        # Compute authentication score and modify A
        auth_score = 1 - (0.4 * F.cosine_similarity(A_old_flat, A_flat, dim=1))
        A = auth_score.unsqueeze(1) * A_flat  # Ensure broadcasting


        # Early return if only attention weights are needed
        if attention_only:
            return A

        # Process attention scores and compute context vectors
        A_raw = A
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)  # Context vector

        # Classify using the context vector
        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        # Return results with optional features
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return Y_prob, mu, var, A



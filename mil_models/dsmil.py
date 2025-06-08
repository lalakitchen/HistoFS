import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1, instance_norm=True):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
        self.instance_norm=instance_norm
        self.instanceNorm = nn.LayerNorm(normalized_shape=in_size, elementwise_affine=True)
    def forward(self, feats):
        if self.instance_norm == True:
            feats = self.instanceNorm(feats)
        x = self.fc(feats)
        return feats, x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class, instance_norm=True):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        self.instanceNorm = nn.LayerNorm(normalized_shape=512, elementwise_affine=True)
        self.instance_norm=instance_norm
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        if self.instance_norm == True:
            feats = self.instanceNorm(feats)
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False, instance_norm=True): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        self.instance_norm=instance_norm
        self.instanceNorm = nn.LayerNorm(normalized_shape=input_size, elementwise_affine=True)
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        if self.instance_norm == True:
            feats = self.instanceNorm(feats)
      
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 

        random_index = torch.randint(0, feats.shape[0], (1,)).item()
        # Get the value from feats at the random index
        random_style = feats[random_index]
        

        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)

        return C
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag = self.b_classifier(feats, classes)
        
        return classes, prediction_bag
        
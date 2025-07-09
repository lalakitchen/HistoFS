import random
import torch
import torch.nn as nn

import torch.nn.functional as F


class StyleStatistics:
    def __init__(self):
        pass

    def get_statistics(self, m_feats):
        mu = m_feats.mean(dim=1)            # Shape: [N]
        var = m_feats.var(dim=1, unbiased=False)  # Shape: [N]
        return mu, var
import torch
import torch.nn as nn
import random

class StyleTransfer(nn.Module):
    def __init__(self, args, eps=1e-6, transfer_ratio=0.5):
        super().__init__()
        self.eps = eps
        self.args = args
        self.transfer_ratio = transfer_ratio  # e.g., 0.5 means 50% of patches will be styled

    def forward(self, feats, statistics):
        if not statistics['bag_mu'] or not statistics['bag_var']:
            return feats

        N, D = feats.size()
        new_feats = []

        # Randomly choose which indices to apply style to
        indices = set(random.sample(range(N), int(N * self.transfer_ratio)))

        for i in range(N):
            f = feats[i].unsqueeze(0)  # [1, D]

            if i in indices:
                mu = f.mean(dim=1, keepdim=True)  # [1, 1]
                var = f.var(dim=1, unbiased=False, keepdim=True)  # [1, 1]
                sig = (var + self.eps).sqrt()  # [1, 1]

                # Sample target style
                mu2 = random.choice(statistics['bag_mu']).to(f.device).view(1, -1)  # [1, D]
                sig2 = random.choice(statistics['bag_var']).to(f.device).view(1, -1)  # [1, D]

                # Apply style transfer
                f_normed = (f - mu) / sig  # [1, D]
                f_styled = f_normed * sig2 + mu2  # [1, D]
                new_feats.append(f_styled)
            else:
                new_feats.append(f)

        return torch.cat(new_feats, dim=0)  # [N, D]

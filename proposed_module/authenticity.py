import torch
import torch.nn as nn
import torch.nn.functional as F

class AuthenticityModule(nn.Module):
    def __init__(self, lambda_val=0.4, eps=1e-8):
        """
        Authenticity Module to align RoIs between original and augmented WSIs.

        Args:
            lambda_val (float): weighting factor for cosine similarity penalty.
            eps (float): small constant to avoid division by zero.
        """
        super(AuthenticityModule, self).__init__()
        self.lambda_val = lambda_val
        self.eps = eps

    def compute_score(self, attn_orig, attn_aug):
        """
        Compute the authenticity score using cosine similarity.

        Args:
            attn_orig (Tensor): Original attention weights. Shape: (P,)
            attn_aug  (Tensor): Augmented attention weights. Shape: (P,)

        Returns:
            auth_score (float): Scalar value reflecting attention similarity.
        """
        norm_orig = attn_orig / (attn_orig.norm(p=2) + self.eps)
        norm_aug  = attn_aug  / (attn_aug.norm(p=2) + self.eps)
        cos_sim = torch.sum(norm_orig * norm_aug)
        auth_score = 1.0 - self.lambda_val * cos_sim.item()
        return auth_score

    def align(self, attn_aug, auth_score):
        """
        Align augmented attention weights based on the authenticity score.

        Args:
            attn_aug (Tensor): Augmented attention weights. Shape: (P,)
            auth_score (float): Authenticity score

        Returns:
            aligned_attn (Tensor): Aligned attention weights. Shape: (P,)
        """
        return auth_score * attn_aug

    def forward(self, attn_orig, attn_aug):
        """
        Complete forward pass: compute authenticity score and aligned weights.

        Args:
            attn_orig (Tensor): Original attention weights. Shape: (P,)
            attn_aug (Tensor): Augmented attention weights. Shape: (P,)

        Returns:
            aligned_attn (Tensor): Aligned augmented attention weights.
        """
        auth_score = self.compute_score(attn_orig, attn_aug)
        return self.align(attn_aug, auth_score)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScalarAdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        """
        Scalar Adaptive Instance Normalization module.
        Applies scalar style transfer using pseudo-style mean and std.
        """
        super(ScalarAdaIN, self).__init__()
        self.eps = eps

    def forward(self, patch_features, style_mean, style_std):
        """
        Apply AdaIN to patch features using scalar style.

        Args:
            patch_features (Tensor): shape [N, D], where N is #patches, D is feature dim.
            style_mean (float or Tensor): scalar mean (e.g., μ_k from pseudo-style)
            style_std (float or Tensor): scalar std  (e.g., σ_k from pseudo-style)

        Returns:
            Tensor: Stylized patch features, shape [N, D]
        """
        patch_mean = patch_features.mean(dim=1, keepdim=True)  # [N, 1]
        patch_std  = patch_features.std(dim=1, keepdim=True) + self.eps  # [N, 1]

        normalized = (patch_features - patch_mean) / patch_std
        stylized = normalized * style_std + style_mean  # scalar broadcast

        return stylized


class PseudoBagStyleTransfer(nn.Module):
    def __init__(self):
        """
        Wrapper class for applying scalar AdaIN using pseudo bag style.
        """
        super(PseudoBagStyleTransfer, self).__init__()
        self.adain = ScalarAdaIN()

    def forward(self, patch_features, pseudo_styles):
        """
        Randomly apply one of the pseudo styles to patch features.

        Args:
            patch_features (Tensor): shape [N, D]
            pseudo_styles (Tensor): shape [K, 2], each row is (mean, std)

        Returns:
            Tensor: Stylized patch features, shape [N, D]
        """
        # Select one pseudo style randomly
        K = pseudo_styles.shape[0]
        k = torch.randint(0, K, (1,)).item()

        style_mean = pseudo_styles[k, 0]
        style_std  = pseudo_styles[k, 1]

        return self.adain(patch_features, style_mean, style_std)

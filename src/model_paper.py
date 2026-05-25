"""
WS-UDA Model Architecture — Exact Paper Replication
Dai et al. (2020) "Adversarial Training Based Multi-Source UDA for Sentiment Analysis"

Key differences from model.py (sequential version):
- Discriminator is MULTI-CLASS (K+1 classes: one per domain including target)
- Discriminator output dim = num_domains (source + target)
- predict() uses normalized discriminator output as weights (Eq. 6 in paper)

Author: Syarif Sanad - 5025221257
Reference: Algorithm 1, Eq. 2-6 in Dai et al. (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ============================================================================
# Gradient Reversal Layer
# ============================================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


# ============================================================================
# Network Modules
# ============================================================================
class SharedExtractor(nn.Module):
    """Es — Shared feature extractor. Input: 5001-dim BoW. Output: 256-dim."""
    def __init__(self, input_dim: int = 5001, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.net(x)


class PrivateExtractor(nn.Module):
    """Ep_k — Domain-specific extractor for domain k. Same arch as Es."""
    def __init__(self, input_dim: int = 5001, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.net(x)


class SentimentClassifier(nn.Module):
    """C — Sentiment classifier. Input: 512-dim (shared+private). Output: 2."""
    def __init__(self, input_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class DomainDiscriminator(nn.Module):
    """
    D — Multi-class Domain Discriminator (PAPER EXACT).

    KEY DIFFERENCE from sequential version:
    - Output = num_all_domains classes (source domains + target domain)
    - e.g. 4 classes for Amazon: Books(0), DVD(1), Electronics(2), Kitchen(3)
    - Used both for adversarial training AND as probability estimator for
      instance-to-domain weights (Eq. 6 in paper)

    Input : [Batch, 256]
    Output: [Batch, num_all_domains]
    """
    def __init__(self, input_dim: int = 256, num_all_domains: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_all_domains)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# WS-UDA Paper Model
# ============================================================================
class WSUDAPaper(nn.Module):
    """
    WS-UDA — Exact replication of Dai et al. (2020).

    Args:
        num_source_domains: K (number of source domains)
        target_domain_id  : domain id assigned to target domain
        num_all_domains   : K + 1 (source + target)
        input_dim         : BoW feature dimension (5001)
        hidden_dim        : Hidden layer size (256)
    """
    def __init__(
        self,
        num_source_domains: int,
        target_domain_id  : int,
        num_all_domains   : int = None,
        input_dim : int = 5001,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.num_source_domains = num_source_domains
        self.target_domain_id   = target_domain_id
        self.num_all_domains    = num_all_domains or (num_source_domains + 1)

        self.Es = SharedExtractor(input_dim, hidden_dim)

        # One private extractor per SOURCE domain only
        self.Ep = nn.ModuleList([
            PrivateExtractor(input_dim, hidden_dim)
            for _ in range(num_source_domains)
        ])

        self.C = SentimentClassifier(hidden_dim * 2, num_classes=2)

        # Multi-class discriminator: K+1 output classes
        self.D = DomainDiscriminator(hidden_dim, self.num_all_domains)

    def forward(
        self,
        x        : torch.Tensor,
        domain_id: torch.Tensor,
        alpha    : float = 1.0,
        is_target: bool  = False
    ):
        """
        Forward pass following Algorithm 1 in paper.

        For SOURCE domains:
          - Compute shared features via Es
          - Compute private features via Ep[k] (routed by domain_id)
          - Classify sentiment with C(shared, private)
          - Discriminate domain on BOTH shared (via GRL) and private (direct)

        For TARGET domain:
          - Only shared features computed (no private extractor for target)
          - Discriminate domain on shared (via GRL) only
          - No sentiment supervision (unlabeled)

        Args:
            x        : Input features [B, input_dim]
            domain_id: Domain labels  [B] — global domain id
            alpha    : GRL scale
            is_target: True if this batch is from target domain

        Returns:
            sentiment_out     : [B, 2] or None if target
            domain_out_shared : [B, num_all_domains]
            domain_out_private: [B, num_all_domains] or None if target
            domain_id         : pass-through for loss computation
        """
        shared     = self.Es(x)
        shared_grl = grad_reverse(shared, alpha)
        domain_out_shared = self.D(shared_grl)

        if is_target:
            # Target: no private extractor, no sentiment label
            return None, domain_out_shared, None, domain_id

        # Source: route to correct Ep[k]
        private = torch.zeros_like(shared)
        for k in range(self.num_source_domains):
            mask = (domain_id == k)
            if mask.sum() == 0:
                continue
            private[mask] = self.Ep[k](x[mask])

        # Private features: direct (no GRL) — domain-informative
        domain_out_private = self.D(private)

        # Sentiment classification
        combined      = torch.cat([shared, private], dim=1)
        sentiment_out = self.C(combined)

        return sentiment_out, domain_out_shared, domain_out_private, domain_id

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        WS-UDA weighted prediction — Eq. 6 in paper.

        w^j_i = Normalize(D(Es(x_i)))[j]  ← confidence that x_i is from domain j
        ŷ_i   = Σ_j  w^j_i · f^j(x_i)

        Each source classifier f^j uses Es + Ep[j].
        Weights come from discriminator's source-domain probabilities.

        Args:
            x: Target features [B, input_dim]

        Returns:
            final_logits: [B, 2]
        """
        self.eval()
        shared = self.Es(x)

        # Discriminator output → softmax → use source-domain probs as weights
        domain_logits = self.D(shared)                        # [B, K+1]
        domain_probs  = F.softmax(domain_logits, dim=1)       # [B, K+1]

        # Use only source domain columns (exclude target column)
        source_cols = [i for i in range(self.num_all_domains)
                       if i != self.target_domain_id]
        weights = domain_probs[:, source_cols]                # [B, K]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # normalize

        # Sentiment from each source classifier
        all_logits = []
        for k in range(self.num_source_domains):
            private_k  = self.Ep[k](x)
            combined_k = torch.cat([shared, private_k], dim=1)
            logits_k   = self.C(combined_k)                   # [B, 2]
            all_logits.append(logits_k)

        # Stack: [K, B, 2] → [B, K, 2]
        stacked = torch.stack(all_logits, dim=0).permute(1, 0, 2)

        # Weighted sum: [B, K, 1] * [B, K, 2] → sum → [B, 2]
        final_logits = (stacked * weights.unsqueeze(2)).sum(dim=1)

        self.train()
        return final_logits

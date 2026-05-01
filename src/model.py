"""
WS-UDA Model Architecture — Sequential Setting
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

Key change from original WS-UDA:
- Discriminator is now BINARY (source=0 vs target=1)
- More stable for sequential setting where domains arrive one at a time
- Private extractors still exist per domain (only active ones are used)

Components:
    - GradientReversalLayer (GRL) : Adversarial training core
    - SharedExtractor (Es)        : Extracts domain-invariant features
    - PrivateExtractor (Ep_k)     : Extracts domain-specific features per source
    - SentimentClassifier (C)     : Predicts sentiment polarity
    - DomainDiscriminator (D)     : Binary — source vs target

Reference: Dai et al. (2020) - Adversarial Training Based MS-UDA
           Ganin et al. (2016) - Domain-Adversarial Training of Neural Networks
           Eq. 2.7 in Proposal: dR(x)/dx = -eta * I

Author: Syarif Sanad - 5025221257
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ============================================================================
# Gradient Reversal Layer (GRL)
# ============================================================================
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer — core of adversarial training.

    Forward  : identity (passes input unchanged)
    Backward : reverses gradient sign, scaled by alpha

    Reference: Ganin et al. (2016), Eq. 2.7 in Proposal
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to tensor x with scale alpha."""
    return GradientReversalFunction.apply(x, alpha)


# ============================================================================
# Network Modules
# ============================================================================
class SharedExtractor(nn.Module):
    """
    Shared Feature Extractor (Es).
    Extracts domain-invariant features shared across all domains.

    Input : [Batch, 5001]
    Output: [Batch, 256]
    """

    def __init__(self, input_dim: int = 5001, hidden_dim: int = 256):
        super(SharedExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrivateExtractor(nn.Module):
    """
    Private Feature Extractor (Ep_k) for domain k.
    Extracts domain-specific features for one source domain.

    Input : [Batch, 5001]
    Output: [Batch, 256]
    """

    def __init__(self, input_dim: int = 5001, hidden_dim: int = 256):
        super(PrivateExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SentimentClassifier(nn.Module):
    """
    Sentiment Classifier (C).
    Predicts sentiment from concatenated shared + private features.

    Input : [Batch, 512]  (256 shared + 256 private)
    Output: [Batch, 2]    (Negative / Positive logits)
    """

    def __init__(self, input_dim: int = 512, num_classes: int = 2):
        super(SentimentClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainDiscriminator(nn.Module):
    """
    Binary Domain Discriminator (D).

    KEY CHANGE: Now binary (source=0 vs target=1) instead of multi-class.

    Why binary for sequential setting:
    - Multi-class discriminator needs to know ALL domains upfront
    - Binary discriminator works for any number of domains
    - More stable when new domains arrive one at a time
    - Still measures source-target alignment (core WS-UDA requirement)

    Input : [Batch, 256]  (shared or private features)
    Output: [Batch, 2]    (source / target logits)
    """

    def __init__(self, input_dim: int = 256):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)   # Binary: source(0) vs target(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# WS-UDA Main Model
# ============================================================================
class WSUDA(nn.Module):
    """
    Weighting Scheme based Unsupervised Domain Adaptation (WS-UDA).
    Sequential Setting with Binary Discriminator.

    Args:
        num_source_domains: Number of source domains (K)
        input_dim         : Feature vector dimension (default 5001)
        hidden_dim        : Hidden layer dimension (default 256)
    """

    def __init__(
        self,
        num_source_domains: int,
        input_dim : int = 5001,
        hidden_dim: int = 256
    ):
        super(WSUDA, self).__init__()

        self.num_source_domains = num_source_domains

        # Shared extractor (one for all domains)
        self.Es = SharedExtractor(input_dim, hidden_dim)

        # Private extractors (one per source domain)
        self.Ep = nn.ModuleList([
            PrivateExtractor(input_dim, hidden_dim)
            for _ in range(num_source_domains)
        ])

        # Sentiment classifier (shared + private features)
        self.C = SentimentClassifier(hidden_dim * 2, num_classes=2)

        # Binary domain discriminator (source vs target)
        self.D = DomainDiscriminator(hidden_dim)

    def forward(
        self,
        x        : torch.Tensor,
        domain_id: torch.Tensor,
        alpha    : float = 1.0,
        is_source: bool  = True
    ):
        """
        Forward pass for training.

        Args:
            x        : Input features [Batch, input_dim]
            domain_id: Domain labels  [Batch] (0-indexed per source domains)
            alpha    : GRL scale factor
            is_source: True if batch is from source domain, False if target

        Returns:
            sentiment_out    : [Batch, 2]  sentiment logits
            domain_out_shared: [Batch, 2]  binary discriminator on shared (via GRL)
            domain_out_private:[Batch, 2]  binary discriminator on private (direct)
        """
        batch_size = x.shape[0]

        # Binary domain label: source=0, target=1
        binary_label = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        if not is_source:
            binary_label = torch.ones(batch_size, dtype=torch.long, device=x.device)

        # --- Shared features ---
        shared = self.Es(x)

        # Discriminator on shared (via GRL — adversarial)
        shared_grl       = grad_reverse(shared, alpha)
        domain_out_shared = self.D(shared_grl)

        # --- Private features ---
        private = torch.zeros_like(shared)

        for k in range(self.num_source_domains):
            mask = (domain_id == k)
            if mask.sum() == 0:
                continue
            private[mask] = self.Ep[k](x[mask])

        # Discriminator on private (direct — domain-informative)
        domain_out_private = self.D(private)

        # --- Sentiment classification ---
        combined      = torch.cat([shared, private], dim=1)
        sentiment_out = self.C(combined)

        return sentiment_out, domain_out_shared, domain_out_private, binary_label

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        WS-UDA weighted prediction at inference (Eq. 6, Dai et al. 2020).

        Uses discriminator confidence as weights for each source classifier.
        Higher confidence = source domain more similar to target = higher weight.

        Args:
            x: Target input features [Batch, input_dim]

        Returns:
            final_logits: Weighted sentiment logits [Batch, 2]
        """
        self.eval()

        shared = self.Es(x)

        # Get source-confidence from discriminator as weights
        # P(source) = softmax(D(shared))[:, 0]
        # Higher P(source) means this domain is more aligned with target
        domain_logits = self.D(shared)                          # [Batch, 2]
        source_probs  = F.softmax(domain_logits, dim=1)[:, 0]  # [Batch]

        # Get sentiment prediction from each source classifier
        all_logits = []
        for k in range(self.num_source_domains):
            private_k  = self.Ep[k](x)
            combined_k = torch.cat([shared, private_k], dim=1)
            logits_k   = self.C(combined_k)                    # [Batch, 2]
            all_logits.append(logits_k)

        # Stack: [K, Batch, 2] -> permute -> [Batch, K, 2]
        stacked = torch.stack(all_logits, dim=0).permute(1, 0, 2)

        # Normalize weights across K domains
        # Use uniform weights — binary discriminator gives same score per domain
        weights = torch.ones(
            x.shape[0], self.num_source_domains,
            device=x.device
        ) / self.num_source_domains                            # [Batch, K]

        # Weighted sum -> [Batch, 2]
        final_logits = (stacked * weights.unsqueeze(2)).sum(dim=1)

        self.train()
        return final_logits

"""
Experience Replay Buffer with Per-Domain Reservoir Sampling
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

Uses separate buffer per domain to ensure fair representation.
Each domain gets equal slots (capacity // num_domains) in the buffer.

FIX: Buffer now uses actual domain_id passed at store time,
not the domain label in the data (which resets to 0 per loader).

Reference: Rebuffi et al. (2017) iCaRL
           Eq. 2.11 in Proposal: P(x in M) = K/t

Author: Syarif Sanad - 5025221257
"""

import random
import torch
from typing import Optional, Tuple, Dict, List


class ReplayBuffer:
    """
    Per-Domain Memory Buffer with Reservoir Sampling.

    Maintains a SEPARATE sub-buffer for each source domain.
    Each domain gets equal capacity = total_capacity // num_domains.

    Args:
        capacity    : Total buffer capacity across all domains (K)
        num_domains : Number of source domains
    """

    def __init__(self, capacity: int = 500, num_domains: int = 3):
        self.capacity      = capacity
        self.num_domains   = num_domains
        self.slots_per_dom = capacity // num_domains

        # Separate buffer and counter per domain
        self.domain_buffers: Dict[int, List] = {
            k: [] for k in range(num_domains)
        }
        self.domain_seen: Dict[int, int] = {
            k: 0 for k in range(num_domains)
        }

        print(f"  [Buffer] Initialized: {capacity} total | "
              f"{self.slots_per_dom} slots/domain | "
              f"{num_domains} domains")

    # =========================================================================
    # Add samples
    # =========================================================================
    def _add_one(self, x: torch.Tensor, y: torch.Tensor, domain_id: int):
        """
        Add ONE sample to the sub-buffer of domain_id using Reservoir Sampling.

        Args:
            x        : Feature tensor [feature_dim]
            y        : Label tensor (scalar)
            domain_id: The REAL domain id (0-indexed across all source domains)
        """
        if domain_id not in self.domain_buffers:
            return

        # Create domain tensor with the REAL domain_id
        d_tensor = torch.tensor(domain_id, dtype=torch.long)
        sample   = (x, y, d_tensor)

        self.domain_seen[domain_id] += 1
        seen = self.domain_seen[domain_id]
        buf  = self.domain_buffers[domain_id]

        if len(buf) < self.slots_per_dom:
            buf.append(sample)
        else:
            # Reservoir Sampling: replace with probability slots/seen
            idx = random.randint(0, seen - 1)
            if idx < self.slots_per_dom:
                buf[idx] = sample

    def add_domain_data(
        self,
        data_loader,
        domain_id: int,
        device: torch.device,
        max_samples: Optional[int] = None
    ):
        """
        Add all samples from a DataLoader to domain_id's sub-buffer.

        NOTE: We ignore the domain label in the data (d) and use
        the provided domain_id instead, because sequential loader
        resets domain IDs to 0 when loading a single domain.

        Args:
            data_loader: DataLoader for the completed domain
            domain_id  : REAL domain id (timestep - 1, 0-indexed)
            device     : Current training device
            max_samples: Optional limit
        """
        added = 0
        for x_batch, y_batch, _ in data_loader:  # ignore d from loader
            x_batch = x_batch.detach().cpu()
            y_batch = y_batch.detach().cpu()

            for i in range(x_batch.shape[0]):
                self._add_one(x_batch[i], y_batch[i], domain_id)
                added += 1
                if max_samples is not None and added >= max_samples:
                    break
            if max_samples is not None and added >= max_samples:
                break

        buf_size = len(self.domain_buffers[domain_id])
        print(f"  [Buffer] Domain {domain_id}: "
              f"stored {buf_size}/{self.slots_per_dom} samples")

    # =========================================================================
    # Sample from buffer
    # =========================================================================
    def sample(
        self,
        batch_size: int,
        device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample a balanced batch from ALL non-empty domain sub-buffers.

        Args:
            batch_size: Total samples to retrieve
            device    : Device to move tensors to

        Returns:
            (x, y, d) tensors, or None if buffer is empty
        """
        active = [k for k, buf in self.domain_buffers.items() if len(buf) > 0]

        if not active:
            return None

        per_domain  = max(1, batch_size // len(active))
        all_samples = []

        for k in active:
            buf = self.domain_buffers[k]
            n   = min(per_domain, len(buf))
            all_samples.extend(random.sample(buf, n))

        if not all_samples:
            return None

        x = torch.stack([s[0] for s in all_samples]).to(device)
        y = torch.stack([s[1] for s in all_samples]).to(device)
        d = torch.stack([s[2] for s in all_samples]).to(device)

        return x, y, d

    # =========================================================================
    # Utilities
    # =========================================================================
    def is_empty(self) -> bool:
        return all(len(buf) == 0 for buf in self.domain_buffers.values())

    def __len__(self) -> int:
        return sum(len(buf) for buf in self.domain_buffers.values())

    def get_stats(self) -> Dict:
        domain_sizes = {k: len(buf) for k, buf in self.domain_buffers.items()}
        total = sum(domain_sizes.values())
        return {
            "size"            : total,
            "capacity"        : self.capacity,
            "slots_per_domain": self.slots_per_dom,
            "fill_rate"       : f"{100 * total / max(self.capacity, 1):.1f}%",
            "domain_sizes"    : domain_sizes
        }

"""
Sequential Domain Loader
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

FIX: Each domain now gets its CORRECT global domain_id via domain_id_offset.

t=1: books      → domain_id = 0
t=2: dvd        → domain_id = 1
t=3: electronics → domain_id = 2

This ensures consistency between:
- Training batches (correct domain_id for Ep[k] routing)
- Replay buffer (stores correct domain_id)
- Model forward pass (routes to correct Ep[k])

Author: Syarif Sanad - 5025221257
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List, Tuple

from data_loader import AmazonDataset, build_vocabulary


class SequentialDomainLoader:
    """
    Loader for sequential domain adaptation experiments.

    At each timestep t:
    - train_loader: ONLY the new domain's data with CORRECT global domain_id
    - test_loaders: ALL seen domains + target

    Args:
        data_dir     : Path to processed_acl directory
        all_domains  : All domain names including target
        target_domain: Name of target domain
        vocabulary   : word_to_id mapping
    """

    def __init__(
        self,
        data_dir     : str,
        all_domains  : List[str],
        target_domain: str,
        vocabulary   : Dict[str, int]
    ):
        self.data_dir      = data_dir
        self.all_domains   = all_domains
        self.target_domain = target_domain
        self.vocabulary    = vocabulary

        self.source_domains = [d for d in all_domains if d != target_domain]

        print(f"\n{'='*60}")
        print(f"Sequential Domain Loader Ready")
        print(f"  Source domains : {self.source_domains}")
        print(f"  Target domain  : {target_domain}")
        print(f"  Total timesteps: {len(self.source_domains)}")
        print(f"{'='*60}\n")

    def get_loader_at_timestep(
        self,
        t         : int,
        batch_size: int = 8,
        shuffle   : bool = True
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """
        Get loaders for timestep t.

        FIX: train_loader now uses domain_id_offset=t-1 so that:
        - t=1 (books)       → domain_id = 0
        - t=2 (dvd)         → domain_id = 1
        - t=3 (electronics) → domain_id = 2

        Args:
            t         : Current timestep (1-indexed)
            batch_size: Batch size
            shuffle   : Whether to shuffle train loader

        Returns:
            train_loader : DataLoader for the new domain with correct domain_id
            test_loaders : Dict {domain_name: DataLoader} for evaluation
        """
        if t < 1 or t > len(self.source_domains):
            raise ValueError(f"Timestep t={t} out of range [1, {len(self.source_domains)}]")

        current_domain = self.source_domains[t - 1]
        seen_domains   = self.source_domains[:t]
        domain_id      = t - 1  # CORRECT global domain id

        print(f"\n[Timestep {t}] New domain: '{current_domain}'")
        print(f"  Seen so far: {seen_domains}")

        # ── Train loader: NEW domain only, with CORRECT domain_id ────────────
        train_dataset = AmazonDataset(
            data_dir         = self.data_dir,
            domains          = [current_domain],
            file_types       = ["positive.review", "negative.review"],
            vocabulary       = self.vocabulary,
            domain_id_offset = domain_id   # KEY FIX: dvd gets 1, electronics gets 2
        )

        train_loader = DataLoader(
            dataset    = train_dataset,
            batch_size = batch_size,
            shuffle    = shuffle,
            drop_last  = True
        )

        print(f"  Train samples : {len(train_dataset)} | "
              f"Batches: {len(train_loader)} | domain_id: {domain_id}")

        # ── Test loaders: all seen domains + target ───────────────────────────
        test_loaders = {}

        for i, domain in enumerate(seen_domains):
            # Each test domain also needs correct global domain_id
            global_id = self.source_domains.index(domain)
            ds = AmazonDataset(
                data_dir         = self.data_dir,
                domains          = [domain],
                file_types       = ["positive.review", "negative.review"],
                vocabulary       = self.vocabulary,
                domain_id_offset = global_id
            )
            test_loaders[domain] = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # Target domain test loader
        target_ds = AmazonDataset(
            data_dir         = self.data_dir,
            domains          = [self.target_domain],
            file_types       = ["positive.review", "negative.review"],
            vocabulary       = self.vocabulary,
            domain_id_offset = 0  # target domain id doesn't matter for eval
        )
        test_loaders[self.target_domain] = DataLoader(
            target_ds, batch_size=batch_size, shuffle=False
        )

        print(f"  Test loaders  : {list(test_loaders.keys())}")

        return train_loader, test_loaders

    def get_oracle_loader(
        self,
        batch_size: int = 8
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """
        Get loader for Oracle baseline (joint training on ALL source domains).

        All domains loaded simultaneously with correct global domain_ids:
        books=0, dvd=1, electronics=2
        """
        print("\n[Oracle] Loading ALL source domains simultaneously...")

        # Load all source domains with correct global IDs
        all_datasets = []
        for i, domain in enumerate(self.source_domains):
            ds = AmazonDataset(
                data_dir         = self.data_dir,
                domains          = [domain],
                file_types       = ["positive.review", "negative.review"],
                vocabulary       = self.vocabulary,
                domain_id_offset = i   # books=0, dvd=1, electronics=2
            )
            all_datasets.append(ds)

        from torch.utils.data import ConcatDataset
        combined = ConcatDataset(all_datasets)

        train_loader = DataLoader(
            dataset    = combined,
            batch_size = batch_size,
            shuffle    = True,
            drop_last  = True
        )

        # Test loaders for all domains
        test_loaders = {}
        for i, domain in enumerate(self.source_domains):
            ds = AmazonDataset(
                data_dir         = self.data_dir,
                domains          = [domain],
                file_types       = ["positive.review", "negative.review"],
                vocabulary       = self.vocabulary,
                domain_id_offset = i
            )
            test_loaders[domain] = DataLoader(ds, batch_size=batch_size, shuffle=False)

        target_ds = AmazonDataset(
            data_dir         = self.data_dir,
            domains          = [self.target_domain],
            file_types       = ["positive.review", "negative.review"],
            vocabulary       = self.vocabulary,
            domain_id_offset = 0
        )
        test_loaders[self.target_domain] = DataLoader(
            target_ds, batch_size=batch_size, shuffle=False
        )

        print(f"  Total train samples: {len(combined)}")
        print(f"  Test loaders: {list(test_loaders.keys())}\n")

        return train_loader, test_loaders

"""
Debug Script — Diagnose Domain ID Issue in Replay Buffer
Run this BEFORE train_replay.py to understand what's happening.

Author: Syarif Sanad - 5025221257
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import build_vocabulary, AmazonDataset
from sequential_loader import SequentialDomainLoader
from replay_buffer import ReplayBuffer
from torch.utils.data import DataLoader

DATA_DIR      = r"C:\ITS\SEMESTER 8\TA\dataset\processed_acl"
ALL_DOMAINS   = ["books", "dvd", "electronics", "kitchen"]
TARGET_DOMAIN = "kitchen"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("\n" + "="*60)
    print("DEBUG — Domain ID Diagnostic")
    print("="*60 + "\n")

    vocab = build_vocabulary(DATA_DIR, ALL_DOMAINS)

    seq_loader = SequentialDomainLoader(
        data_dir=DATA_DIR, all_domains=ALL_DOMAINS,
        target_domain=TARGET_DOMAIN, vocabulary=vocab
    )

    replay_buffer = ReplayBuffer(capacity=500, num_domains=3)

    # =========================================================================
    # TEST 1: What domain_id does the loader assign?
    # =========================================================================
    print("=" * 60)
    print("TEST 1: Domain IDs assigned by sequential loader")
    print("=" * 60)

    for t in range(1, 4):
        domain_name = seq_loader.source_domains[t - 1]
        train_loader, _ = seq_loader.get_loader_at_timestep(t, batch_size=8)

        # Check first batch
        x, y, d = next(iter(train_loader))
        unique_d = torch.unique(d).tolist()

        print(f"\nTimestep {t} — domain '{domain_name}':")
        print(f"  domain_id in batch (d): {unique_d}")
        print(f"  Expected real id      : {t-1}")
        print(f"  Match?                : {'✓' if unique_d == [t-1] else '✗ MISMATCH!'}")

    # =========================================================================
    # TEST 2: What domain_id does replay buffer store & return?
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Domain IDs stored in replay buffer")
    print("=" * 60)

    for t in range(1, 4):
        domain_name = seq_loader.source_domains[t - 1]
        domain_id   = t - 1
        train_loader, _ = seq_loader.get_loader_at_timestep(t, batch_size=8)

        # Simulate storing to buffer
        replay_buffer.add_domain_data(train_loader, domain_id=domain_id, device=DEVICE)

        # Check what's stored
        buf = replay_buffer.domain_buffers[domain_id]
        if buf:
            stored_d = buf[0][2].item()  # d tensor of first sample
            print(f"\nTimestep {t} — domain '{domain_name}' (real id={domain_id}):")
            print(f"  Stored domain_id in buffer: {stored_d}")
            print(f"  Match with real id?        : {'✓' if stored_d == domain_id else '✗ MISMATCH!'}")

    # =========================================================================
    # TEST 3: What does sample() return?
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Domain IDs returned by sample()")
    print("=" * 60)

    xr, yr, dr = replay_buffer.sample(batch_size=16, device=DEVICE)
    unique_dr = torch.unique(dr).tolist()

    print(f"\nSampled batch domain IDs: {unique_dr}")
    print(f"Expected               : [0, 1, 2] (all domains)")
    print(f"Match?                 : {'✓' if set(unique_dr) == {0,1,2} else '✗ MISSING DOMAINS!'}")
    print(f"\nDistribution in sample:")
    for k in range(3):
        count = (dr == k).sum().item()
        print(f"  Domain {k}: {count} samples")

    # =========================================================================
    # TEST 4: Does model.Ep[k] get called with correct k?
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Model private extractor indexing")
    print("=" * 60)

    from model import WSUDA
    model = WSUDA(num_source_domains=3).to(DEVICE)

    print("\nSimulating forward pass with replay batch:")
    print(f"  dr (domain ids in replay): {unique_dr}")
    print(f"  model.num_source_domains : {model.num_source_domains}")

    # Check which Ep[k] gets called for each domain_id
    for k in range(model.num_source_domains):
        mask = (dr == k)
        count = mask.sum().item()
        print(f"  Ep[{k}] will process {count} samples (domain_id={k})")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    print("\nLook for any '✗ MISMATCH!' above — that's the bug location.")


if __name__ == "__main__":
    main()

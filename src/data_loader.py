"""
Data Loader for Amazon Review Dataset
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

FIX: Added domain_id_offset parameter to AmazonDataset so that
sequential loader can assign the CORRECT global domain_id
instead of always starting from 0.

Author : Syarif Sanad - 5025221257
Dataset: Amazon Review Benchmark (Blitzer et al., 2007)
"""

import os
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ============================================================================
# Constants
# ============================================================================
TOP_K_FEATURES: int = 5000
FEATURE_DIM   : int = TOP_K_FEATURES + 1  # +1 for 1-indexed vocabulary


# ============================================================================
# Vocabulary
# ============================================================================
def build_vocabulary(data_dir: str, domains: List[str]) -> Dict[str, int]:
    """
    Build vocabulary from all domains by selecting top-K frequent words.

    Args:
        data_dir : Root directory containing domain subdirectories
        domains  : List of domain names

    Returns:
        word_to_id: Dict mapping word -> index (1-indexed, 0 = padding)
    """
    print("Building vocabulary...")
    word_counts = Counter()

    for domain in domains:
        domain_path = os.path.join(data_dir, domain)
        for filename in os.listdir(domain_path):
            if not filename.endswith(".review"):
                continue
            filepath = os.path.join(domain_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"  Reading {domain}/{filename}", leave=False):
                    try:
                        words = [item.split(':')[0] for item in line.strip().split()]
                        word_counts.update(words)
                    except (ValueError, IndexError):
                        continue

    top_k      = [word for word, _ in word_counts.most_common(TOP_K_FEATURES)]
    word_to_id = {word: idx + 1 for idx, word in enumerate(top_k)}

    print(f"Vocabulary built: {len(word_to_id)} features\n")
    return word_to_id


# ============================================================================
# Feature Extraction
# ============================================================================
def file_to_vectors(
    filepath  : str,
    word_to_id: Dict[str, int]
) -> Tuple[List[np.ndarray], List[int]]:
    """Convert a .review file into bag-of-words feature vectors."""
    if "positive" in filepath:
        label = 1
    elif "negative" in filepath:
        label = 0
    else:
        label = -1

    vectors, labels = [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            vector = np.zeros(FEATURE_DIM, dtype=np.float32)
            items  = line.strip().split()
            if not items:
                continue
            for item in items:
                try:
                    parts = item.split(':')
                    if len(parts) == 2:
                        word, count = parts[0], float(parts[1])
                        if word in word_to_id:
                            vector[word_to_id[word]] = count
                except (ValueError, IndexError):
                    continue
            vectors.append(vector)
            labels.append(label)

    return vectors, labels


def load_domain_data(
    data_dir       : str,
    domains        : List[str],
    file_types     : List[str],
    vocabulary     : Dict[str, int],
    domain_id_offset: int = 0
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    Load data from multiple domains.

    FIX: domain_id_offset allows correct global domain ID assignment.

    Example:
        domains=["dvd"], domain_id_offset=1
        → dvd samples get domain_id = 1 (not 0)

    Args:
        data_dir        : Root data directory
        domains         : List of domain names to load
        file_types      : File types to load
        vocabulary      : Word-to-id mapping
        domain_id_offset: Starting domain ID (default 0)
                          Set to timestep-1 for correct sequential IDs
    """
    all_vectors, all_labels, all_domain_ids = [], [], []

    for i, domain in enumerate(domains):
        # FIXED: domain_id = offset + position in current list
        domain_id   = domain_id_offset + i
        domain_path = os.path.join(data_dir, domain)

        for file_type in file_types:
            filepath = os.path.join(domain_path, file_type)
            if not os.path.exists(filepath):
                print(f"  [Warning] File not found: {filepath}")
                continue

            vectors, labels = file_to_vectors(filepath, vocabulary)
            all_vectors.extend(vectors)
            all_labels.extend(labels)
            all_domain_ids.extend([domain_id] * len(vectors))

    return all_vectors, all_labels, all_domain_ids


# ============================================================================
# PyTorch Dataset
# ============================================================================
class AmazonDataset(Dataset):
    """
    PyTorch Dataset for Amazon Review data.

    FIX: Added domain_id_offset parameter for correct sequential domain IDs.

    Each sample returns: (feature_vector, sentiment_label, domain_id)

    Args:
        data_dir        : Root data directory
        domains         : List of domain names to include
        file_types      : List of file types to load
        vocabulary      : Word-to-id mapping
        domain_id_offset: Starting domain ID (default 0)
                          Use timestep-1 for sequential setting
    """

    def __init__(
        self,
        data_dir        : str,
        domains         : List[str],
        file_types      : List[str],
        vocabulary      : Dict[str, int],
        domain_id_offset: int = 0
    ):
        vectors, labels, domain_ids = load_domain_data(
            data_dir, domains, file_types, vocabulary,
            domain_id_offset=domain_id_offset
        )

        self.vectors    = torch.tensor(np.array(vectors),    dtype=torch.float32)
        self.labels     = torch.tensor(np.array(labels),     dtype=torch.long)
        self.domain_ids = torch.tensor(np.array(domain_ids), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.vectors[idx], self.labels[idx], self.domain_ids[idx]

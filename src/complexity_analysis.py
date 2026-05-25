"""
Complexity Analysis Script — Empirical Measurement
Mengukur computational cost aktual dari ketiga metode:
- Oracle (Joint Training)
- Naive Sequential
- Experience Replay

Metrics yang diukur:
1. Wall-clock time (waktu training aktual)
2. Peak GPU memory usage (VRAM)
3. Estimasi FLOPS per forward pass

Output: results/complexity_results.json

Author: Syarif Sanad - 5025221257
"""

import torch
import torch.nn as nn
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import WSUDA
from data_loader import build_vocabulary, AmazonDataset
from sequential_loader import SequentialDomainLoader
from replay_buffer import ReplayBuffer
from torch.utils.data import DataLoader

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR      = r"C:\ITS\SEMESTER 8\TA\dataset\processed_acl"
ALL_DOMAINS   = ["books", "dvd", "electronics", "kitchen"]
TARGET_DOMAIN = "kitchen"
RESULTS_DIR   = r"C:\ITS\SEMESTER 8\TA\results"

BATCH_SIZE   = 8
LR           = 0.0001
EPOCHS_ORACLE = 30
EPOCHS_SEQ    = 10
N_CRITIC      = 5
LAMBDA_ADV    = 0.1
BETA_REPLAY   = 1.0
BUFFER_CAP    = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Helper: Memory Measurement
# ============================================================================
def get_gpu_memory_mb():
    """Return current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def get_peak_gpu_memory_mb():
    """Return peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0

def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# ============================================================================
# Helper: FLOPS Estimation
# ============================================================================
def estimate_flops_per_forward(model, input_dim=5001, batch_size=8):
    """
    Estimate FLOPS for one forward pass through the model.
    Menggunakan formula: FLOPS per Linear layer = 2 × input_dim × output_dim
    (factor 2 untuk multiply + add)
    """
    flops = 0

    # SharedExtractor: Linear(5001→512) + Linear(512→256)
    flops += 2 * input_dim * 512        # Es layer 1
    flops += 2 * 512 * 256              # Es layer 2

    # PrivateExtractor × K: Linear(5001→512) + Linear(512→256)
    k = model.num_source_domains
    flops += k * (2 * input_dim * 512)  # Ep layer 1 × K
    flops += k * (2 * 512 * 256)        # Ep layer 2 × K

    # Classifier: Linear(512→128) + Linear(128→2)
    flops += 2 * 512 * 128              # C layer 1
    flops += 2 * 128 * 2                # C layer 2

    # Discriminator: Linear(256→128) + Linear(128→2)
    flops += 2 * 256 * 128              # D layer 1
    flops += 2 * 128 * 2                # D layer 2

    # Multiply by batch size
    flops *= batch_size

    return flops


# ============================================================================
# Measure: Oracle
# ============================================================================
def measure_oracle(vocab, seq_loader):
    print("\n" + "="*60)
    print("MEASURING: Oracle (Joint Training)")
    print("="*60)

    reset_memory_stats()

    model    = WSUDA(num_source_domains=3).to(DEVICE)
    loss_cls = nn.CrossEntropyLoss()
    loss_dom = nn.CrossEntropyLoss()

    import torch.optim as optim
    opt_D  = optim.Adam(model.D.parameters(), lr=LR)
    opt_EC = optim.Adam(
        list(model.Es.parameters()) +
        list(model.Ep.parameters()) +
        list(model.C.parameters()), lr=LR
    )

    train_loader, _ = seq_loader.get_oracle_loader(batch_size=BATCH_SIZE)

    # Estimate FLOPS
    flops_per_forward = estimate_flops_per_forward(model, batch_size=BATCH_SIZE)
    total_iterations  = len(train_loader) * EPOCHS_ORACLE
    total_flops       = total_iterations * (N_CRITIC + 1) * flops_per_forward

    # Measure time and memory
    start_time = time.time()
    memory_before = get_gpu_memory_mb()

    iteration_count = 0

    for epoch in range(1, EPOCHS_ORACLE + 1):
        for x, y, d in train_loader:
            x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
            src_label = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)

            for _ in range(N_CRITIC):
                opt_D.zero_grad()
                with torch.no_grad():
                    shared = model.Es(x)
                loss_dom(model.D(shared), src_label).backward()
                opt_D.step()

            opt_EC.zero_grad()
            sent, dom_s, dom_p, bin_label = model(x, d, alpha=1.0)
            lc = loss_cls(sent, y)
            la = loss_dom(dom_s, bin_label)
            (lc + LAMBDA_ADV * la).backward()
            opt_EC.step()

            iteration_count += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{EPOCHS_ORACLE} | "
                  f"Memory: {get_gpu_memory_mb():.1f} MB")

    elapsed = time.time() - start_time
    peak_mem = get_peak_gpu_memory_mb()

    results = {
        "method"              : "Oracle (Joint Training)",
        "training_time_sec"   : round(elapsed, 2),
        "training_time_min"   : round(elapsed / 60, 2),
        "peak_gpu_memory_mb"  : round(peak_mem, 2),
        "total_iterations"    : iteration_count,
        "estimated_flops"     : total_flops,
        "estimated_gflops"    : round(total_flops / 1e9, 2),
        "epochs"              : EPOCHS_ORACLE,
        "samples_per_epoch"   : len(train_loader.dataset),
        "big_o_time"          : "O(N × K × E × nc)",
        "big_o_space"         : "O(N × K × D)",
    }

    print(f"\n  Training time : {results['training_time_min']:.2f} min")
    print(f"  Peak VRAM     : {results['peak_gpu_memory_mb']:.2f} MB")
    print(f"  Total iters   : {results['total_iterations']:,}")
    print(f"  Est. GFLOPS   : {results['estimated_gflops']:.2f}")

    return results


# ============================================================================
# Measure: Naive Sequential
# ============================================================================
def measure_naive(vocab, seq_loader):
    print("\n" + "="*60)
    print("MEASURING: Naive Sequential")
    print("="*60)

    reset_memory_stats()

    import torch.optim as optim
    model    = WSUDA(num_source_domains=3).to(DEVICE)
    loss_cls = nn.CrossEntropyLoss()
    loss_dom = nn.CrossEntropyLoss()

    flops_per_forward = estimate_flops_per_forward(model, batch_size=BATCH_SIZE)
    num_sources = len(seq_loader.source_domains)

    start_time = time.time()
    total_iteration_count = 0

    for t in range(1, num_sources + 1):
        train_loader, _ = seq_loader.get_loader_at_timestep(t, batch_size=BATCH_SIZE)

        opt_D  = optim.Adam(model.D.parameters(), lr=LR)
        opt_EC = optim.Adam(
            list(model.Es.parameters()) +
            list(model.Ep.parameters()) +
            list(model.C.parameters()), lr=LR
        )

        for epoch in range(1, EPOCHS_SEQ + 1):
            for x, y, d in train_loader:
                x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
                src_label = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)

                for _ in range(N_CRITIC):
                    opt_D.zero_grad()
                    with torch.no_grad():
                        shared = model.Es(x)
                    loss_dom(model.D(shared), src_label).backward()
                    opt_D.step()

                opt_EC.zero_grad()
                sent, dom_s, dom_p, bin_label = model(x, d, alpha=1.0)
                lc = loss_cls(sent, y)
                la = loss_dom(dom_s, bin_label)
                (lc + LAMBDA_ADV * la).backward()
                opt_EC.step()

                total_iteration_count += 1

        print(f"  Timestep {t}/{num_sources} done | "
              f"Memory: {get_gpu_memory_mb():.1f} MB")

    elapsed  = time.time() - start_time
    peak_mem = get_peak_gpu_memory_mb()

    total_flops = total_iteration_count * (N_CRITIC + 1) * flops_per_forward

    results = {
        "method"              : "Naive Sequential",
        "training_time_sec"   : round(elapsed, 2),
        "training_time_min"   : round(elapsed / 60, 2),
        "peak_gpu_memory_mb"  : round(peak_mem, 2),
        "total_iterations"    : total_iteration_count,
        "estimated_flops"     : total_flops,
        "estimated_gflops"    : round(total_flops / 1e9, 2),
        "epochs_per_timestep" : EPOCHS_SEQ,
        "num_timesteps"       : num_sources,
        "big_o_time"          : "O(N × K × E_s × nc)",
        "big_o_space"         : "O(N × D)",
    }

    print(f"\n  Training time : {results['training_time_min']:.2f} min")
    print(f"  Peak VRAM     : {results['peak_gpu_memory_mb']:.2f} MB")
    print(f"  Total iters   : {results['total_iterations']:,}")
    print(f"  Est. GFLOPS   : {results['estimated_gflops']:.2f}")

    return results


# ============================================================================
# Measure: Experience Replay
# ============================================================================
def measure_replay(vocab, seq_loader):
    print("\n" + "="*60)
    print("MEASURING: Experience Replay")
    print("="*60)

    reset_memory_stats()

    import torch.optim as optim
    model         = WSUDA(num_source_domains=3).to(DEVICE)
    loss_cls      = nn.CrossEntropyLoss()
    loss_dom      = nn.CrossEntropyLoss()
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAP, num_domains=3)

    flops_per_forward = estimate_flops_per_forward(model, batch_size=BATCH_SIZE)
    num_sources = len(seq_loader.source_domains)

    start_time = time.time()
    total_iteration_count = 0
    replay_iteration_count = 0

    for t in range(1, num_sources + 1):
        domain_id    = t - 1
        train_loader, _ = seq_loader.get_loader_at_timestep(t, batch_size=BATCH_SIZE)

        opt_D  = optim.Adam(model.D.parameters(), lr=LR)
        opt_EC = optim.Adam(
            list(model.Es.parameters()) +
            list(model.Ep.parameters()) +
            list(model.C.parameters()), lr=LR
        )

        for epoch in range(1, EPOCHS_SEQ + 1):
            for x, y, d in train_loader:
                x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
                src_label = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)

                for _ in range(N_CRITIC):
                    opt_D.zero_grad()
                    with torch.no_grad():
                        shared = model.Es(x)
                    ld = loss_dom(model.D(shared), src_label)
                    if not replay_buffer.is_empty():
                        xr, yr, dr = replay_buffer.sample(BATCH_SIZE, DEVICE)
                        src_r = torch.zeros(xr.shape[0], dtype=torch.long, device=DEVICE)
                        with torch.no_grad():
                            shared_r = model.Es(xr)
                        ld = ld + loss_dom(model.D(shared_r), src_r)
                    ld.backward()
                    opt_D.step()

                opt_EC.zero_grad()
                sent, dom_s, dom_p, bin_label = model(x, d, alpha=1.0)
                lc     = loss_cls(sent, y)
                la     = loss_dom(dom_s, bin_label)
                l_curr = lc + LAMBDA_ADV * la

                l_replay = torch.tensor(0.0, device=DEVICE)
                if not replay_buffer.is_empty():
                    xr, yr, dr = replay_buffer.sample(BATCH_SIZE, DEVICE)
                    sent_r, dom_s_r, _, bin_r = model(xr, dr, alpha=1.0)
                    l_replay = (loss_cls(sent_r, yr) +
                                LAMBDA_ADV * loss_dom(dom_s_r, bin_r))
                    replay_iteration_count += 1

                (l_curr + BETA_REPLAY * l_replay).backward()
                opt_EC.step()

                total_iteration_count += 1

        replay_buffer.add_domain_data(train_loader, domain_id=domain_id, device=DEVICE)
        print(f"  Timestep {t}/{num_sources} done | "
              f"Memory: {get_gpu_memory_mb():.1f} MB | "
              f"Buffer: {len(replay_buffer)}")

    elapsed  = time.time() - start_time
    peak_mem = get_peak_gpu_memory_mb()

    # Replay iterations tambah overhead forward pass
    total_flops = ((total_iteration_count + replay_iteration_count) *
                   (N_CRITIC + 1) * flops_per_forward)

    results = {
        "method"                 : "Experience Replay",
        "training_time_sec"      : round(elapsed, 2),
        "training_time_min"      : round(elapsed / 60, 2),
        "peak_gpu_memory_mb"     : round(peak_mem, 2),
        "total_iterations"       : total_iteration_count,
        "replay_iterations"      : replay_iteration_count,
        "estimated_flops"        : total_flops,
        "estimated_gflops"       : round(total_flops / 1e9, 2),
        "epochs_per_timestep"    : EPOCHS_SEQ,
        "num_timesteps"          : num_sources,
        "buffer_capacity"        : BUFFER_CAP,
        "buffer_ratio"           : round(BUFFER_CAP / (2000 * 3) * 100, 2),
        "big_o_time"             : "O((N+B) × K × E_s × nc)",
        "big_o_space"            : "O((N+B) × D)",
    }

    print(f"\n  Training time   : {results['training_time_min']:.2f} min")
    print(f"  Peak VRAM       : {results['peak_gpu_memory_mb']:.2f} MB")
    print(f"  Total iters     : {results['total_iterations']:,}")
    print(f"  Replay iters    : {results['replay_iterations']:,}")
    print(f"  Est. GFLOPS     : {results['estimated_gflops']:.2f}")
    print(f"  Buffer ratio    : {results['buffer_ratio']}% of total data")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS — Empirical Measurement")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Measuring: Oracle, Naive Sequential, Experience Replay")
    print("="*60)

    # Setup
    vocab = build_vocabulary(DATA_DIR, ALL_DOMAINS)
    seq_loader = SequentialDomainLoader(
        data_dir=DATA_DIR, all_domains=ALL_DOMAINS,
        target_domain=TARGET_DOMAIN, vocabulary=vocab
    )

    # Run measurements
    oracle_results = measure_oracle(vocab, seq_loader)
    naive_results  = measure_naive(vocab, seq_loader)
    replay_results = measure_replay(vocab, seq_loader)

    # Compute relative costs
    base_time = oracle_results["training_time_sec"]
    base_mem  = oracle_results["peak_gpu_memory_mb"]

    oracle_results["relative_time"] = 1.00
    naive_results["relative_time"]  = round(naive_results["training_time_sec"] / base_time, 3)
    replay_results["relative_time"] = round(replay_results["training_time_sec"] / base_time, 3)

    oracle_results["relative_memory"] = 1.00
    naive_results["relative_memory"]  = round(naive_results["peak_gpu_memory_mb"] / base_mem, 3)
    replay_results["relative_memory"] = round(replay_results["peak_gpu_memory_mb"] / base_mem, 3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — Complexity Comparison")
    print("="*60)
    print(f"\n{'Metric':<25} {'Oracle':>10} {'Naive':>10} {'Replay':>10}")
    print("-"*60)
    print(f"{'Training Time (min)':<25} "
          f"{oracle_results['training_time_min']:>10.2f} "
          f"{naive_results['training_time_min']:>10.2f} "
          f"{replay_results['training_time_min']:>10.2f}")
    print(f"{'Peak VRAM (MB)':<25} "
          f"{oracle_results['peak_gpu_memory_mb']:>10.2f} "
          f"{naive_results['peak_gpu_memory_mb']:>10.2f} "
          f"{replay_results['peak_gpu_memory_mb']:>10.2f}")
    print(f"{'Total Iterations':<25} "
          f"{oracle_results['total_iterations']:>10,} "
          f"{naive_results['total_iterations']:>10,} "
          f"{replay_results['total_iterations']:>10,}")
    print(f"{'Est. GFLOPS':<25} "
          f"{oracle_results['estimated_gflops']:>10.2f} "
          f"{naive_results['estimated_gflops']:>10.2f} "
          f"{replay_results['estimated_gflops']:>10.2f}")
    print(f"{'Relative Time':<25} "
          f"{oracle_results['relative_time']:>10.2f}x "
          f"{naive_results['relative_time']:>10.2f}x "
          f"{replay_results['relative_time']:>10.2f}x")
    print(f"{'Relative Memory':<25} "
          f"{oracle_results['relative_memory']:>10.2f}x "
          f"{naive_results['relative_memory']:>10.2f}x "
          f"{replay_results['relative_memory']:>10.2f}x")
    print("="*60)

    # Save results
    output = {
        "oracle" : oracle_results,
        "naive"  : naive_results,
        "replay" : replay_results,
        "comparison": {
            "replay_vs_oracle_time_reduction"  : f"{(1 - replay_results['relative_time']) * 100:.1f}%",
            "replay_vs_oracle_memory_reduction": f"{(1 - replay_results['relative_memory']) * 100:.1f}%",
            "replay_overhead_vs_naive_time"    : f"{(replay_results['relative_time'] / naive_results['relative_time'] - 1) * 100:.1f}%",
            "buffer_ratio_of_total_data"       : f"{replay_results['buffer_ratio']}%",
        }
    }

    out_path = os.path.join(RESULTS_DIR, "complexity_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {out_path}")
    print(f"\nKey findings:")
    print(f"  Replay vs Oracle time reduction  : {output['comparison']['replay_vs_oracle_time_reduction']}")
    print(f"  Replay vs Oracle memory reduction: {output['comparison']['replay_vs_oracle_memory_reduction']}")
    print(f"  Replay overhead vs Naive         : {output['comparison']['replay_overhead_vs_naive_time']}")
    print(f"  Buffer uses only                 : {output['comparison']['buffer_ratio_of_total_data']} of total data\n")


if __name__ == "__main__":
    main()

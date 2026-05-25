"""
Complexity Analysis — WS-UDA Paper Replication
Mengukur training time, peak VRAM, dan estimated FLOPS
untuk setting leave-one-out seperti paper Dai et al. (2020)

Hasil akan disimpan di results/complexity_results_paper.json
dan digabung dengan complexity_results.json yang sudah ada.

Author: Syarif Sanad - 5025221257
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import build_vocabulary, AmazonDataset
from model_paper import WSUDAPaper

# ============================================================================
# Configuration — same as train_wsuda_paper.py
# ============================================================================
DATA_DIR    = r"C:\ITS\SEMESTER 8\TA\dataset\processed_acl"
ALL_DOMAINS = ["books", "dvd", "electronics", "kitchen"]
RESULTS_DIR = r"C:\ITS\SEMESTER 8\TA\results"

BATCH_SIZE = 8
LR         = 0.0001
N_CRITIC   = 5
LAMBDA_ADV = 1.0
MEASURE_EPOCHS = 5   # cukup 5 epoch untuk ukur steady-state complexity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# FLOPS Estimator (same formula as complexity_analysis.py)
# ============================================================================
def estimate_flops_per_iter(model, batch_size, input_dim=5001, hidden_dim=256):
    """
    Estimate FLOPS per training iteration for WS-UDA paper model.

    Components:
    - Es  : 2 linear layers (input→512, 512→256)
    - Ep[k]: same as Es, for K source domains
    - C   : 2 linear layers (512→128, 128→2)
    - D   : 2 linear layers (256→128, 128→K+1)  ← multi-class, K+1 outputs

    Each linear(a→b): 2*a*b FLOPs (multiply-add)
    """
    K = model.num_source_domains

    # Es: Linear(5001→512) + Linear(512→256)
    flops_Es = 2 * (input_dim * 512 + 512 * hidden_dim)

    # Ep[k] × K (only active domain per batch in practice, but K total)
    flops_Ep = K * flops_Es

    # C: Linear(512→128) + Linear(128→2)
    flops_C = 2 * (hidden_dim * 2 * 128 + 128 * 2)

    # D: Linear(256→128) + Linear(128→K+1)  ← K+1 classes
    flops_D = 2 * (hidden_dim * 128 + 128 * (K + 1))

    # Per sample flops
    flops_per_sample = flops_Es + flops_Ep + flops_C + flops_D

    # Per iteration: batch_size samples × (N_CRITIC D updates + 1 EC update)
    flops_per_iter = batch_size * flops_per_sample * (N_CRITIC + 1)

    return flops_per_iter


# ============================================================================
# Measure one leave-one-out run
# ============================================================================
def measure_one_run(target_domain, vocab):
    """Measure complexity for one LOO run (target = target_domain)."""
    source_domains = [d for d in ALL_DOMAINS if d != target_domain]
    target_id      = len(source_domains)

    # ── Build loaders ────────────────────────────────────────────────────────
    src_datasets = []
    for i, domain in enumerate(source_domains):
        ds = AmazonDataset(
            data_dir=DATA_DIR, domains=[domain],
            file_types=["positive.review", "negative.review"],
            vocabulary=vocab, domain_id_offset=i
        )
        src_datasets.append(ds)

    source_combined  = ConcatDataset(src_datasets)
    source_loader    = DataLoader(source_combined, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True)

    target_full = AmazonDataset(
        data_dir=DATA_DIR, domains=[target_domain],
        file_types=["positive.review", "negative.review"],
        vocabulary=vocab, domain_id_offset=target_id
    )
    target_loader = DataLoader(target_full, batch_size=BATCH_SIZE,
                               shuffle=True, drop_last=True)

    # ── Model & optimizer ────────────────────────────────────────────────────
    model = WSUDAPaper(
        num_source_domains=len(source_domains),
        target_domain_id=target_id,
        num_all_domains=len(ALL_DOMAINS)
    ).to(DEVICE)

    loss_cls = nn.CrossEntropyLoss()
    loss_dom = nn.CrossEntropyLoss()
    opt_D    = optim.Adam(model.D.parameters(), lr=LR)
    opt_EC   = optim.Adam(
        list(model.Es.parameters()) +
        list(model.Ep.parameters()) +
        list(model.C.parameters()), lr=LR
    )

    # ── Reset VRAM stats ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.empty_cache()

    target_iter   = iter(target_loader)
    total_iters   = 0
    total_flops   = 0
    flops_per_iter = estimate_flops_per_iter(model, BATCH_SIZE)

    start_time = time.time()

    for epoch in range(1, MEASURE_EPOCHS + 1):
        model.train()
        for x_src, y_src, d_src in source_loader:
            x_src = x_src.to(DEVICE)
            y_src = y_src.to(DEVICE)
            d_src = d_src.to(DEVICE)

            try:
                x_tgt, _, d_tgt = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_tgt, _, d_tgt = next(target_iter)
            x_tgt = x_tgt.to(DEVICE)
            d_tgt = d_tgt.to(DEVICE)

            # D update
            for _ in range(N_CRITIC):
                opt_D.zero_grad()
                with torch.no_grad():
                    shared_src  = model.Es(x_src)
                    private_src = torch.zeros_like(shared_src)
                    for k in range(model.num_source_domains):
                        mask = (d_src == k)
                        if mask.sum() > 0:
                            private_src[mask] = model.Ep[k](x_src[mask])
                    shared_tgt = model.Es(x_tgt)
                ld = (loss_dom(model.D(shared_src), d_src) +
                      loss_dom(model.D(private_src), d_src) +
                      loss_dom(model.D(shared_tgt), d_tgt))
                ld.backward()
                opt_D.step()

            # EC update
            opt_EC.zero_grad()
            sent, dom_s, _, _ = model(x_src, d_src, alpha=1.0, is_target=False)
            _, dom_tgt_s, _, _ = model(x_tgt, d_tgt, alpha=1.0, is_target=True)
            lc = loss_cls(sent, y_src)
            la = loss_dom(dom_s, d_src) + loss_dom(dom_tgt_s, d_tgt)
            (lc + LAMBDA_ADV * la).backward()
            opt_EC.step()

            total_iters += 1
            total_flops += flops_per_iter

    elapsed_sec = time.time() - start_time

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)

    # Extrapolate to full training (from results_wsuda_paper.json timing)
    # We ran 5 epochs, actual training ran ~13-35 epochs per run
    # Use actual training time from results_wsuda_paper.json
    actual_times = {
        "books": 5.30, "dvd": 5.04,
        "electronics": 4.82, "kitchen": 12.76
    }
    actual_time_sec = actual_times[target_domain] * 60

    # Scale FLOPS to actual epoch count
    actual_epochs = {
        "books": 15, "dvd": 14, "electronics": 13, "kitchen": 35
    }
    scale = actual_epochs[target_domain] / MEASURE_EPOCHS
    actual_flops = total_flops * scale
    actual_iters = total_iters * scale

    return {
        "target_domain"       : target_domain,
        "source_domains"      : source_domains,
        "measured_epochs"     : MEASURE_EPOCHS,
        "actual_epochs"       : actual_epochs[target_domain],
        "training_time_sec"   : actual_time_sec,
        "training_time_min"   : actual_times[target_domain],
        "peak_gpu_memory_mb"  : round(peak_vram_mb, 2),
        "total_iterations"    : int(actual_iters),
        "estimated_flops"     : int(actual_flops),
        "estimated_gflops"    : round(actual_flops / 1e9, 2),
        "flops_per_iter"      : flops_per_iter,
        "num_source_domains"  : len(source_domains),
        "discriminator_classes": len(ALL_DOMAINS),
        "big_o_time"          : "O(N × K × E × nc) per run × (K+1) runs",
        "big_o_space"         : "O((N×K + N_target) × D)"
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print("\n" + "="*65)
    print("COMPLEXITY ANALYSIS — WS-UDA Paper Replication")
    print("Leave-One-Out Setting (4 runs)")
    print("="*65)
    print(f"Device: {DEVICE} | Measure epochs: {MEASURE_EPOCHS}/run")
    print("="*65 + "\n")

    vocab = build_vocabulary(DATA_DIR, ALL_DOMAINS)

    run_metrics = {}
    wall_start  = time.time()

    for target_domain in ALL_DOMAINS:
        print(f"\n[Measuring] Target = '{target_domain}'...")
        metrics = measure_one_run(target_domain, vocab)
        run_metrics[target_domain] = metrics
        print(f"  Time     : {metrics['training_time_min']:.2f} min")
        print(f"  VRAM     : {metrics['peak_gpu_memory_mb']:.2f} MB")
        print(f"  GFLOPS   : {metrics['estimated_gflops']:.2f}")
        print(f"  Iters    : {metrics['total_iterations']:,}")

    # ── Aggregate across all 4 runs ──────────────────────────────────────────
    total_time_sec  = sum(m["training_time_sec"]  for m in run_metrics.values())
    total_time_min  = sum(m["training_time_min"]  for m in run_metrics.values())
    peak_vram_mb    = max(m["peak_gpu_memory_mb"] for m in run_metrics.values())
    total_iters     = sum(m["total_iterations"]   for m in run_metrics.values())
    total_flops     = sum(m["estimated_flops"]    for m in run_metrics.values())
    total_gflops    = total_flops / 1e9

    # ── Load existing complexity results for comparison ───────────────────────
    existing_path = os.path.join(RESULTS_DIR, "complexity_results.json")
    existing = {}
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)

    oracle_time = existing.get("oracle", {}).get("training_time_min", 29.53)
    oracle_flop = existing.get("oracle", {}).get("estimated_gflops", 23468.73)

    rel_time   = round(total_time_min / oracle_time, 3)
    rel_memory = round(peak_vram_mb /
                       existing.get("oracle", {}).get("peak_gpu_memory_mb", 228.26), 3)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("AGGREGATE — All 4 LOO Runs Combined")
    print("="*65)
    print(f"  Total training time : {total_time_min:.2f} min")
    print(f"  Peak VRAM (max run) : {peak_vram_mb:.2f} MB")
    print(f"  Total GFLOPS        : {total_gflops:.2f}")
    print(f"  Total iterations    : {total_iters:,}")
    print(f"  Relative time vs Oracle: {rel_time:.2f}x")

    print("\n" + "="*65)
    print("COMPARISON — All Methods")
    print("="*65)
    print(f"{'Method':<28} {'Time (min)':>12} {'VRAM (MB)':>12} {'GFLOPS':>12}")
    print("-"*65)

    methods = {
        "Oracle (Joint Training)": (
            existing.get("oracle",{}).get("training_time_min", 29.53),
            existing.get("oracle",{}).get("peak_gpu_memory_mb", 228.26),
            existing.get("oracle",{}).get("estimated_gflops", 23468.73)
        ),
        "Naive Sequential": (
            existing.get("naive",{}).get("training_time_min", 1.47),
            existing.get("naive",{}).get("peak_gpu_memory_mb", 144.23),
            existing.get("naive",{}).get("estimated_gflops", 7822.91)
        ),
        "Experience Replay": (
            existing.get("replay",{}).get("training_time_min", 2.25),
            existing.get("replay",{}).get("peak_gpu_memory_mb", 228.44),
            existing.get("replay",{}).get("estimated_gflops", 13038.18)
        ),
        "WS-UDA Paper (LOO ×4)": (
            total_time_min, peak_vram_mb, total_gflops
        ),
    }
    for method, (t, v, f) in methods.items():
        print(f"  {method:<26} {t:>11.2f} {v:>11.2f} {f:>11.2f}")
    print("="*65)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "method"         : "WS-UDA Paper Replication (Leave-One-Out)",
        "reference"      : "Dai et al. (2020) AAAI",
        "setting"        : "Leave-one-out: 4 runs, each domain as target once",
        "configuration"  : {
            "batch_size" : BATCH_SIZE, "lr": LR,
            "n_critic"   : N_CRITIC,   "lambda_adv": LAMBDA_ADV,
            "discriminator": "multi-class (4 classes)",
            "unlabeled_target_in_training": True
        },
        "per_run_metrics": run_metrics,
        "aggregate"      : {
            "total_training_time_sec" : round(total_time_sec, 2),
            "total_training_time_min" : round(total_time_min, 2),
            "peak_gpu_memory_mb"      : round(peak_vram_mb, 2),
            "total_iterations"        : total_iters,
            "estimated_flops"         : total_flops,
            "estimated_gflops"        : round(total_gflops, 2),
            "big_o_time"              : "O(N × K × E × nc) × (K+1) LOO runs",
            "big_o_space"             : "O((N×K + N_target) × D)",
            "relative_time_vs_oracle" : rel_time,
            "relative_memory_vs_oracle": rel_memory
        },
        "comparison_note": (
            "WS-UDA paper runs 4 separate experiments (LOO) "
            "vs sequential methods that run once. "
            "Total time reflects full evaluation protocol."
        )
    }

    out_path = os.path.join(RESULTS_DIR, "complexity_results_paper.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved: {out_path}")
    print(f"✓ Wall time (measurement only): "
          f"{(time.time()-wall_start)/60:.1f} min\n")


if __name__ == "__main__":
    main()
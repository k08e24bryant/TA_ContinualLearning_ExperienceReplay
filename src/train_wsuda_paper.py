"""
WS-UDA Paper Replication — Exact Setting (Dai et al., 2020)
Amazon Review Dataset, Leave-One-Out Evaluation

Differences from train_oracle.py (sequential version):
1. Discriminator: multi-class (K+1) instead of binary
2. Evaluation:    leave-one-out (each domain takes turn as target)
3. Target data:   unlabeled target samples join adversarial training
4. Training:      joint (all source domains at once per run)
5. Early stopping on validation set (per paper implementation details)

Expected results (Table 2, Dai et al. 2020):
    Books:       79.39%
    DVD:         80.14%
    Electronics: 83.81%
    Kitchen:     87.66%
    Average:     82.75%

Author: Syarif Sanad - 5025221257
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import build_vocabulary, AmazonDataset
from model_paper import WSUDAPaper

# ============================================================================
# Configuration — matches paper exactly
# ============================================================================
DATA_DIR    = r"C:\ITS\SEMESTER 8\TA\dataset\processed_acl"
ALL_DOMAINS = ["books", "dvd", "electronics", "kitchen"]
RESULTS_DIR = r"C:\ITS\SEMESTER 8\TA\results"
CKPT_DIR    = r"C:\ITS\SEMESTER 8\TA\checkpoints"

BATCH_SIZE   = 8        # paper: batch size 8
LR           = 0.0001   # paper: learning rate 0.0001
N_CRITIC     = 5        # paper: ncritic
LAMBDA_ADV   = 1.0      # paper: λ (adversarial loss weight)
MAX_EPOCHS   = 50       # max epochs with early stopping
PATIENCE     = 10       # early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Data Loading — per leave-one-out run
# ============================================================================
def build_loaders_for_run(data_dir, all_domains, target_domain, vocabulary):
    """
    Build dataloaders for one leave-one-out run.

    Source domains: all except target (labeled)
    Target domain : unlabeled — joined adversarial training
    Val/Test      : 2000 labeled samples from target domain split 50/50

    Domain ID assignment:
        Source domains: 0, 1, ..., K-1  (alphabetical order of source list)
        Target domain : K
    """
    source_domains = [d for d in all_domains if d != target_domain]
    target_id      = len(source_domains)  # K

    print(f"\n  Source domains : {source_domains}")
    print(f"  Target domain  : {target_domain} (id={target_id})")

    # ── Source datasets (labeled) ────────────────────────────────────────────
    src_datasets = []
    for i, domain in enumerate(source_domains):
        ds = AmazonDataset(
            data_dir         = data_dir,
            domains          = [domain],
            file_types       = ["positive.review", "negative.review"],
            vocabulary       = vocabulary,
            domain_id_offset = i
        )
        src_datasets.append(ds)
        print(f"  Source [{i}] {domain}: {len(ds)} samples")

    source_combined = ConcatDataset(src_datasets)
    source_loader   = DataLoader(source_combined, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)

    # ── Target dataset (unlabeled — only used for adversarial D training) ────
    # Paper uses ALL unlabeled target data for adversarial training
    # We use labeled target data but IGNORE labels during adversarial step
    target_full = AmazonDataset(
        data_dir         = data_dir,
        domains          = [target_domain],
        file_types       = ["positive.review", "negative.review"],
        vocabulary       = vocabulary,
        domain_id_offset = target_id
    )

    # Split target: 1000 val, 1000 test (paper: validation set for early stopping)
    n_total   = len(target_full)
    n_val     = n_total // 2
    n_test    = n_total - n_val
    val_ds, test_ds = torch.utils.data.random_split(
        target_full, [n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Full target (unlabeled) for adversarial training
    target_adv_loader = DataLoader(target_full, batch_size=BATCH_SIZE,
                                   shuffle=True, drop_last=True)
    val_loader        = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)
    test_loader       = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Target [{target_id}] {target_domain}: "
          f"{len(target_full)} total | {n_val} val | {n_test} test")

    return (source_loader, target_adv_loader, val_loader, test_loader,
            source_domains, target_id)


# ============================================================================
# Evaluation
# ============================================================================
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y   = x.to(device), y.to(device)
            logits  = model.predict(x)
            preds   = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    model.train()
    return round(100.0 * correct / total, 4) if total > 0 else 0.0


# ============================================================================
# One Training Run (one leave-one-out fold)
# ============================================================================
def train_one_run(model, source_loader, target_adv_loader,
                  val_loader, test_loader,
                  source_domains, target_id, device):
    """
    Train WS-UDA for one leave-one-out fold.
    Follows Algorithm 1 in paper exactly.
    """
    loss_cls = nn.CrossEntropyLoss()
    loss_dom = nn.CrossEntropyLoss()

    opt_D  = optim.Adam(model.D.parameters(), lr=LR)
    opt_EC = optim.Adam(
        list(model.Es.parameters()) +
        list(model.Ep.parameters()) +
        list(model.C.parameters()), lr=LR
    )

    best_val_acc  = 0.0
    best_test_acc = 0.0
    patience_cnt  = 0
    history       = []

    # Make target loader iterable (cycle)
    target_iter = iter(target_adv_loader)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_lc, total_ld, n = 0.0, 0.0, 0

        for x_src, y_src, d_src in tqdm(source_loader,
                                         desc=f"    Epoch {epoch}", leave=False):
            x_src = x_src.to(device)
            y_src = y_src.to(device)
            d_src = d_src.to(device)

            # Get a target batch (for adversarial training)
            try:
                x_tgt, _, d_tgt = next(target_iter)
            except StopIteration:
                target_iter = iter(target_adv_loader)
                x_tgt, _, d_tgt = next(target_iter)

            x_tgt = x_tgt.to(device)
            d_tgt = d_tgt.to(device)

            # ── STAGE 1: Update Discriminator D (Algorithm 1, lines 3-13) ──
            # Paper: update D on ALL domains (source + target)
            for _ in range(N_CRITIC):
                opt_D.zero_grad()

                # Source — shared features (via GRL frozen for D update)
                with torch.no_grad():
                    shared_src  = model.Es(x_src)
                    private_src = torch.zeros_like(shared_src)
                    for k in range(model.num_source_domains):
                        mask = (d_src == k)
                        if mask.sum() > 0:
                            private_src[mask] = model.Ep[k](x_src[mask])
                    shared_tgt = model.Es(x_tgt)

                # D on source shared features → predict correct source domain
                ld_s = loss_dom(model.D(shared_src), d_src)

                # D on source private features → predict correct source domain
                # (only for source, not target — line 10-11 in Algorithm 1)
                ld_p = loss_dom(model.D(private_src), d_src)

                # D on target shared features → predict target domain id
                ld_t = loss_dom(model.D(shared_tgt), d_tgt)

                ld = ld_s + ld_p + ld_t
                ld.backward()
                opt_D.step()

            # ── STAGE 2: Update Es, Ep, C (Algorithm 1, lines 14-22) ──
            opt_EC.zero_grad()

            # Classification loss on source (line 18)
            sent, dom_s, dom_p, _ = model(x_src, d_src, alpha=1.0, is_target=False)
            lc = loss_cls(sent, y_src)

            # Adversarial loss on source shared — confuse D (line 21)
            # GRL already negates gradient in forward pass
            la_src = loss_dom(dom_s, d_src)

            # Adversarial loss on target shared — confuse D
            _, dom_tgt_s, _, _ = model(x_tgt, d_tgt, alpha=1.0, is_target=True)
            la_tgt = loss_dom(dom_tgt_s, d_tgt)

            loss = lc + LAMBDA_ADV * (la_src + la_tgt)
            loss.backward()
            opt_EC.step()

            total_lc += lc.item()
            total_ld += ld.item()
            n += 1

        # ── Evaluation & Early Stopping ──────────────────────────────────────
        val_acc  = evaluate(model, val_loader,  device)
        test_acc = evaluate(model, test_loader, device)

        print(f"    Epoch {epoch:3d} | "
              f"L_cls={total_lc/n:.4f} | L_dom={total_ld/n:.4f} | "
              f"Val={val_acc:.2f}% | Test={test_acc:.2f}%")

        history.append({
            "epoch": epoch, "val_acc": val_acc, "test_acc": test_acc,
            "loss_cls": total_lc/n, "loss_dom": total_ld/n
        })

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = test_acc
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stopping at epoch {epoch} "
                      f"(best val={best_val_acc:.2f}%, test={best_test_acc:.2f}%)")
                break

    return best_val_acc, best_test_acc, history


# ============================================================================
# Main — Leave-One-Out Loop
# ============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    print("\n" + "="*65)
    print("WS-UDA PAPER REPLICATION — Leave-One-Out Evaluation")
    print("Dai et al. (2020) — Amazon Review Dataset")
    print("="*65)
    print(f"Device: {DEVICE} | LR: {LR} | Batch: {BATCH_SIZE} | "
          f"N_critic: {N_CRITIC} | Patience: {PATIENCE}")
    print("="*65)

    start_total = time.time()

    print("\nBuilding vocabulary...")
    vocab = build_vocabulary(DATA_DIR, ALL_DOMAINS)

    run_results = {}

    # ── Leave-One-Out: each domain takes turn as target ───────────────────────
    for target_domain in ALL_DOMAINS:
        print(f"\n{'='*65}")
        print(f"RUN: Target = '{target_domain}'")
        print(f"{'='*65}")

        start_run = time.time()

        (source_loader, target_adv_loader, val_loader, test_loader,
         source_domains, target_id) = build_loaders_for_run(
            DATA_DIR, ALL_DOMAINS, target_domain, vocab
        )

        # Model: K source domains + 1 target = K+1 discriminator classes
        model = WSUDAPaper(
            num_source_domains = len(source_domains),
            target_domain_id   = target_id,
            num_all_domains    = len(ALL_DOMAINS),
            input_dim  = 5001,
            hidden_dim = 256
        ).to(DEVICE)

        print(f"\n  Model: {len(source_domains)} source + 1 target = "
              f"{len(ALL_DOMAINS)}-class discriminator")

        best_val, best_test, history = train_one_run(
            model, source_loader, target_adv_loader,
            val_loader, test_loader,
            source_domains, target_id, DEVICE
        )

        elapsed = round((time.time() - start_run) / 60, 2)
        print(f"\n  ✓ Run done: Val={best_val:.2f}% | "
              f"Test={best_test:.2f}% | Time={elapsed} min")

        run_results[target_domain] = {
            "target_domain" : target_domain,
            "source_domains": source_domains,
            "best_val_acc"  : best_val,
            "best_test_acc" : best_test,
            "training_time_min": elapsed,
            "history"       : history
        }

        # Save checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"wsuda_paper_{target_domain}.pth")
        torch.save(model.state_dict(), ckpt_path)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("FINAL RESULTS — WS-UDA Paper Replication")
    print("="*65)
    print(f"\n{'Domain':<15} {'Our Replic.':>14} {'Paper (Dai 2020)':>18} {'Gap':>8}")
    print("-"*55)

    paper_results = {
        "books"      : 79.39,
        "dvd"        : 80.14,
        "electronics": 83.81,
        "kitchen"    : 87.66
    }

    our_accs  = []
    gaps      = []
    for domain in ALL_DOMAINS:
        our_acc  = run_results[domain]["best_test_acc"]
        paper_ac = paper_results[domain]
        gap      = our_acc - paper_ac
        our_accs.append(our_acc)
        gaps.append(gap)
        marker = "✓" if abs(gap) <= 2.0 else "⚠"
        print(f"  {marker} {domain:<13} {our_acc:>13.2f}% "
              f"{paper_ac:>17.2f}% {gap:>+7.2f}%")

    our_avg   = sum(our_accs) / len(our_accs)
    paper_avg = 82.75
    avg_gap   = our_avg - paper_avg
    print("-"*55)
    print(f"  {'Average':<14} {our_avg:>13.2f}% "
          f"{paper_avg:>17.2f}% {avg_gap:>+7.2f}%")
    print(f"\n  Total time: {(time.time()-start_total)/60:.1f} min")
    print("="*65)

    if abs(avg_gap) <= 2.0:
        print("\n  ✓ Replication SUCCESSFUL — within 2% of paper results")
    elif abs(avg_gap) <= 5.0:
        print("\n  ⚠ Close replication — within 5% of paper results")
        print("    Possible causes: random seed, unlabeled data split, epochs")
    else:
        print("\n  ✗ Gap > 5% — check configuration")

    # ── Save results ──────────────────────────────────────────────────────────
    final_results = {
        "method"          : "WS-UDA Paper Replication",
        "reference"       : "Dai et al. (2020) AAAI",
        "evaluation"      : "Leave-One-Out",
        "configuration"   : {
            "batch_size" : BATCH_SIZE, "lr": LR,
            "n_critic"   : N_CRITIC,   "lambda_adv": LAMBDA_ADV,
            "max_epochs" : MAX_EPOCHS, "patience": PATIENCE,
            "discriminator": f"multi-class ({len(ALL_DOMAINS)} classes)"
        },
        "run_results"     : run_results,
        "summary"         : {
            domain: {
                "our_acc"  : run_results[domain]["best_test_acc"],
                "paper_acc": paper_results[domain],
                "gap"      : round(run_results[domain]["best_test_acc"]
                                   - paper_results[domain], 4)
            }
            for domain in ALL_DOMAINS
        },
        "our_avg"         : round(our_avg, 4),
        "paper_avg"       : paper_avg,
        "avg_gap"         : round(avg_gap, 4),
        "total_time_min"  : round((time.time() - start_total) / 60, 2)
    }

    out_path = os.path.join(RESULTS_DIR, "results_wsuda_paper.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Results saved: {out_path}")
    print(f"✓ Checkpoints  : {CKPT_DIR}/wsuda_paper_*.pth\n")


if __name__ == "__main__":
    main()

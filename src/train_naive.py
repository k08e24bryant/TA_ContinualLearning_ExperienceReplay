"""
Naive Sequential Baseline — Demonstrating Catastrophic Forgetting
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

Trains on domains one at a time WITHOUT any continual learning.
Expected result: significant catastrophic forgetting.
This is the LOWER BOUND.

Author: Syarif Sanad - 5025221257
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import build_vocabulary
from sequential_loader import SequentialDomainLoader
from model import WSUDA

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR      = r"C:\ITS\SEMESTER 8\TA\dataset\processed_acl"
ALL_DOMAINS   = ["books", "dvd", "electronics", "kitchen"]
TARGET_DOMAIN = "kitchen"
RESULTS_DIR   = r"C:\ITS\SEMESTER 8\TA\results"
CKPT_DIR      = r"C:\ITS\SEMESTER 8\TA\checkpoints"

BATCH_SIZE   = 8
LR           = 0.0001
EPOCHS_PER_T = 10
N_CRITIC     = 5
LAMBDA_ADV   = 0.1
LAMBDA_PRIV  = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Helpers
# ============================================================================
def evaluate(model, test_loaders, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for name, loader in test_loaders.items():
            correct, total = 0, 0
            for x, y, _ in loader:
                x, y   = x.to(device), y.to(device)
                logits  = model.predict(x)
                preds   = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
            results[name] = round(100.0 * correct / total, 4)
    model.train()
    return results


def train_one_timestep(model, loader, opt_D, opt_EC,
                       loss_cls, loss_dom, epochs, device):
    model.train()

    for epoch in range(1, epochs + 1):
        total_lc, total_ld, n = 0.0, 0.0, 0

        for x, y, d in tqdm(loader, desc=f"    Epoch {epoch}/{epochs}", leave=False):
            x, y, d = x.to(device), y.to(device), d.to(device)

            # Binary source label
            src_label = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            # ── A. Update Discriminator ────────────────────────────────────
            for _ in range(N_CRITIC):
                opt_D.zero_grad()

                with torch.no_grad():
                    shared  = model.Es(x)
                    private = torch.zeros_like(shared)
                    for k in range(model.num_source_domains):
                        mask = (d == k)
                        if mask.sum() > 0:
                            private[mask] = model.Ep[k](x[mask])

                ld = (loss_dom(model.D(shared), src_label) +
                      loss_dom(model.D(private), src_label))
                ld.backward()
                opt_D.step()

            # ── B. Update Extractor + Classifier ──────────────────────────
            opt_EC.zero_grad()

            sent, dom_s, dom_p, bin_label = model(x, d, alpha=1.0, is_source=True)
            lc   = loss_cls(sent, y)
            la   = loss_dom(dom_s, bin_label)
            lp   = loss_dom(dom_p, bin_label)
            loss = lc + LAMBDA_ADV * la + LAMBDA_PRIV * lp
            loss.backward()
            opt_EC.step()

            total_lc += lc.item()
            total_ld += ld.item()
            n += 1

        print(f"    Epoch {epoch}/{epochs} | "
              f"L_cls={total_lc/n:.4f} | L_dom={total_ld/n:.4f}")


def compute_forgetting(results_over_time, source_domains):
    forgetting = {}
    for domain in source_domains[:-1]:
        best  = max(r["accuracies"][domain] for r in results_over_time
                    if domain in r["accuracies"])
        final = results_over_time[-1]["accuracies"][domain]
        forgetting[domain] = round(best - final, 4)
    if forgetting:
        forgetting["average"] = round(sum(forgetting.values()) / len(forgetting), 4)
    else:
        forgetting["average"] = 0.0
    return forgetting


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    print("\n" + "="*60)
    print("NAIVE SEQUENTIAL — Catastrophic Forgetting Baseline")
    print("="*60)
    print(f"Device: {DEVICE} | Epochs/T: {EPOCHS_PER_T} | LR: {LR}")
    print("="*60 + "\n")

    start = time.time()

    vocab = build_vocabulary(DATA_DIR, ALL_DOMAINS)
    seq_loader = SequentialDomainLoader(
        data_dir=DATA_DIR, all_domains=ALL_DOMAINS,
        target_domain=TARGET_DOMAIN, vocabulary=vocab
    )

    num_sources = len(seq_loader.source_domains)
    model = WSUDA(num_source_domains=num_sources).to(DEVICE)

    loss_cls = nn.CrossEntropyLoss()
    loss_dom = nn.CrossEntropyLoss()

    results_over_time = []

    for t in range(1, num_sources + 1):
        domain_name = seq_loader.source_domains[t - 1]

        print(f"\n{'='*60}")
        print(f"TIMESTEP {t}/{num_sources} — New domain: '{domain_name}'")
        print(f"{'='*60}")

        train_loader, test_loaders = seq_loader.get_loader_at_timestep(
            t, batch_size=BATCH_SIZE
        )

        opt_D  = optim.Adam(model.D.parameters(), lr=LR)
        opt_EC = optim.Adam(
            list(model.Es.parameters()) +
            list(model.Ep.parameters()) +
            list(model.C.parameters()), lr=LR
        )

        train_one_timestep(model, train_loader, opt_D, opt_EC,
                           loss_cls, loss_dom, EPOCHS_PER_T, DEVICE)

        accs = evaluate(model, test_loaders, DEVICE)

        print(f"\n  Results at Timestep {t}:")
        for d, acc in accs.items():
            tag = " (TARGET)" if d == TARGET_DOMAIN else ""
            print(f"    {d:20s}: {acc:.2f}%{tag}")

        results_over_time.append({
            "timestep"  : t,
            "new_domain": domain_name,
            "accuracies": accs
        })

        if t > 1:
            print(f"\n  Forgetting Analysis:")
            for prev_d in seq_loader.source_domains[:t-1]:
                best = max(r["accuracies"][prev_d] for r in results_over_time
                           if prev_d in r["accuracies"])
                curr = accs[prev_d]
                print(f"    {prev_d:20s}: "
                      f"best={best:.2f}% -> now={curr:.2f}% "
                      f"(delta={curr-best:+.2f}%)")

    forgetting = compute_forgetting(results_over_time, seq_loader.source_domains)
    final_accs = results_over_time[-1]["accuracies"]
    src_accs   = [final_accs[d] for d in seq_loader.source_domains]
    avg_src    = round(sum(src_accs) / len(src_accs), 4)

    print("\n" + "="*60)
    print("FINAL RESULTS — Naive Sequential")
    print("="*60)
    for d, acc in final_accs.items():
        tag = " (TARGET)" if d == TARGET_DOMAIN else ""
        print(f"  {d:20s}: {acc:.2f}%{tag}")
    print(f"\n  Avg Source Accuracy : {avg_src:.2f}%")
    print(f"  Avg Forgetting Rate : {forgetting['average']:+.2f}%")

    if forgetting["average"] >= 4.0:
        print("\n  Hypothesis confirmed! Catastrophic forgetting exists.")
    elif forgetting["average"] >= 2.0:
        print("\n  Moderate forgetting observed.")
    else:
        print("\n  Low forgetting — check if model is underfitting.")
    print("="*60)

    results = {
        "method"            : "Naive Sequential (No Replay)",
        "configuration"     : {
            "epochs_per_t": EPOCHS_PER_T, "batch_size": BATCH_SIZE, "lr": LR,
            "lambda_adv": LAMBDA_ADV, "lambda_priv": LAMBDA_PRIV
        },
        "source_domains"    : seq_loader.source_domains,
        "target_domain"     : TARGET_DOMAIN,
        "results_over_time" : results_over_time,
        "forgetting_metrics": forgetting,
        "final_accuracies"  : final_accs,
        "avg_source_acc"    : avg_src,
        "training_time_min" : round((time.time() - start) / 60, 2)
    }

    out_path = os.path.join(RESULTS_DIR, "results_naive.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "naive_final.pth"))
    print(f"\n✓ Results saved to: {out_path}")
    print(f"✓ Model saved to  : {CKPT_DIR}/naive_final.pth\n")


if __name__ == "__main__":
    main()

"""
Oracle Baseline — Joint Training (Upper Bound)
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

Trains on ALL source domains simultaneously.
No forgetting possible — this is the UPPER BOUND.

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

BATCH_SIZE = 8
LR         = 0.0001
EPOCHS     = 30
N_CRITIC   = 5
LAMBDA_ADV = 0.1
LAMBDA_PRIV= 0.1

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


def train_epoch(model, loader, opt_D, opt_EC, loss_cls, loss_dom, device):
    model.train()
    total_lc, total_ld, n = 0.0, 0.0, 0

    for x, y, d in tqdm(loader, desc="  Training", leave=False):
        x, y, d = x.to(device), y.to(device), d.to(device)

        # Binary source label (all batches here are source)
        src_label = torch.zeros(x.shape[0], dtype=torch.long, device=device)

        # ── A. Update Discriminator ────────────────────────────────────────
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

        # ── B. Update Extractor + Classifier ──────────────────────────────
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

    return total_lc / n, total_ld / n


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    print("\n" + "="*60)
    print("ORACLE — Joint Training (Upper Bound)")
    print("="*60)
    print(f"Device : {DEVICE} | Epochs: {EPOCHS} | LR: {LR}")
    print("="*60 + "\n")

    start = time.time()

    vocab = build_vocabulary(DATA_DIR, ALL_DOMAINS)
    seq_loader = SequentialDomainLoader(
        data_dir=DATA_DIR, all_domains=ALL_DOMAINS,
        target_domain=TARGET_DOMAIN, vocabulary=vocab
    )

    train_loader, test_loaders = seq_loader.get_oracle_loader(batch_size=BATCH_SIZE)

    num_sources = len(seq_loader.source_domains)
    model = WSUDA(num_source_domains=num_sources).to(DEVICE)

    opt_D  = optim.Adam(model.D.parameters(), lr=LR)
    opt_EC = optim.Adam(
        list(model.Es.parameters()) +
        list(model.Ep.parameters()) +
        list(model.C.parameters()), lr=LR
    )

    loss_cls = nn.CrossEntropyLoss()
    loss_dom = nn.CrossEntropyLoss()

    history = []
    best_target_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        lc, ld = train_epoch(model, train_loader, opt_D, opt_EC,
                             loss_cls, loss_dom, DEVICE)
        accs = evaluate(model, test_loaders, DEVICE)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"L_cls={lc:.4f} | L_dom={ld:.4f} | "
              f"Target={accs[TARGET_DOMAIN]:.2f}%")

        history.append({"epoch": epoch, "loss_cls": lc, "loss_dom": ld, "accs": accs})

        if accs[TARGET_DOMAIN] > best_target_acc:
            best_target_acc = accs[TARGET_DOMAIN]
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "oracle_best.pth"))

    final_accs = history[-1]["accs"]
    src_accs   = [final_accs[d] for d in seq_loader.source_domains]
    avg_src    = sum(src_accs) / len(src_accs)

    print("\n" + "="*60)
    print("FINAL RESULTS — Oracle")
    print("="*60)
    for d, acc in final_accs.items():
        tag = " (TARGET)" if d == TARGET_DOMAIN else ""
        print(f"  {d:20s}: {acc:.2f}%{tag}")
    print(f"\n  Avg Source Acc : {avg_src:.2f}%")
    print(f"  Best Target Acc: {best_target_acc:.2f}%")
    print(f"  Time           : {(time.time()-start)/60:.1f} min")
    print("="*60)

    results = {
        "method"         : "Oracle (Joint Training)",
        "configuration"  : {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "source_domains" : seq_loader.source_domains,
        "target_domain"  : TARGET_DOMAIN,
        "history"        : history,
        "final_accuracies": final_accs,
        "avg_source_acc" : avg_src,
        "best_target_acc": best_target_acc,
        "training_time_min": (time.time() - start) / 60
    }

    out_path = os.path.join(RESULTS_DIR, "results_oracle.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {out_path}")
    print(f"✓ Model saved to  : {CKPT_DIR}/oracle_best.pth\n")


if __name__ == "__main__":
    main()

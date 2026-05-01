"""
Sequential Training with Experience Replay — Proposed Method
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

WS-UDA + Per-Domain Memory Buffer (Reservoir Sampling).

Loss: L_total = L_curr(D_t; theta) + beta * L_replay(M; theta)
Reference: Eq. 2.10 in Proposal, Rebuffi et al. (2017) iCaRL

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
from replay_buffer import ReplayBuffer

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

BETA_REPLAY     = 1.0
BUFFER_CAPACITY = 500

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


def train_one_timestep_with_replay(
    model, loader, replay_buffer,
    opt_D, opt_EC, loss_cls, loss_dom, epochs, device
):
    model.train()
    use_replay = not replay_buffer.is_empty()

    if use_replay:
        stats = replay_buffer.get_stats()
        print(f"  [Replay] Buffer: {stats['size']}/{stats['capacity']} | "
              f"Per domain: {stats['domain_sizes']}")
    else:
        print(f"  [Replay] Buffer empty — first timestep, no replay yet")

    for epoch in range(1, epochs + 1):
        total_lc, total_lr, total_ld, n = 0.0, 0.0, 0.0, 0

        for x, y, d in tqdm(loader, desc=f"    Epoch {epoch}/{epochs}", leave=False):
            x, y, d = x.to(device), y.to(device), d.to(device)

            # Binary source label for current batch
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

                # Update D on replay samples too
                if use_replay:
                    rb = replay_buffer.sample(BATCH_SIZE, device)
                    if rb is not None:
                        xr, yr, dr = rb
                        src_label_r = torch.zeros(xr.shape[0], dtype=torch.long, device=device)
                        with torch.no_grad():
                            shared_r = model.Es(xr)
                        ld = ld + loss_dom(model.D(shared_r), src_label_r)

                ld.backward()
                opt_D.step()

            # ── B. Update Extractor + Classifier with Replay ───────────────
            opt_EC.zero_grad()

            # Current domain loss
            sent, dom_s, dom_p, bin_label = model(x, d, alpha=1.0, is_source=True)
            lc     = loss_cls(sent, y)
            la     = loss_dom(dom_s, bin_label)
            lp     = loss_dom(dom_p, bin_label)
            l_curr = lc + LAMBDA_ADV * la + LAMBDA_PRIV * lp

            # Replay loss
            l_replay = torch.tensor(0.0, device=device)

            if use_replay:
                rb = replay_buffer.sample(BATCH_SIZE, device)
                if rb is not None:
                    xr, yr, dr = rb
                    sent_r, dom_s_r, _, bin_label_r = model(
                        xr, dr, alpha=1.0, is_source=True
                    )
                    lc_r     = loss_cls(sent_r, yr)
                    la_r     = loss_dom(dom_s_r, bin_label_r)
                    l_replay = lc_r + LAMBDA_ADV * la_r

            # L_total = L_curr + beta * L_replay  (Eq. 2.10)
            loss_total = l_curr + BETA_REPLAY * l_replay
            loss_total.backward()
            opt_EC.step()

            total_lc += lc.item()
            total_lr += l_replay.item()
            total_ld += ld.item()
            n += 1

        print(f"    Epoch {epoch}/{epochs} | "
              f"L_curr={total_lc/n:.4f} | "
              f"L_replay={total_lr/n:.4f} | "
              f"L_dom={total_ld/n:.4f}")


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
    print("EXPERIENCE REPLAY — Proposed Method")
    print("="*60)
    print(f"Device: {DEVICE} | Epochs/T: {EPOCHS_PER_T} | LR: {LR}")
    print(f"Beta: {BETA_REPLAY} | Buffer: {BUFFER_CAPACITY}")
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

    replay_buffer = ReplayBuffer(
        capacity=BUFFER_CAPACITY,
        num_domains=num_sources
    )

    results_over_time = []

    for t in range(1, num_sources + 1):
        domain_name = seq_loader.source_domains[t - 1]
        domain_id   = t - 1

        print(f"\n{'='*60}")
        print(f"TIMESTEP {t}/{num_sources} — New domain: '{domain_name}' (id={domain_id})")
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

        train_one_timestep_with_replay(
            model, train_loader, replay_buffer,
            opt_D, opt_EC, loss_cls, loss_dom, EPOCHS_PER_T, DEVICE
        )

        # Store to per-domain buffer using REAL domain_id
        print(f"\n  [Replay] Storing '{domain_name}' (id={domain_id}) to buffer...")
        replay_buffer.add_domain_data(train_loader, domain_id=domain_id, device=DEVICE)

        stats = replay_buffer.get_stats()
        print(f"  [Replay] Total: {stats['size']}/{stats['capacity']} | "
              f"Per domain: {stats['domain_sizes']}")

        accs = evaluate(model, test_loaders, DEVICE)

        print(f"\n  Results at Timestep {t}:")
        for d, acc in accs.items():
            tag = " (TARGET)" if d == TARGET_DOMAIN else ""
            print(f"    {d:20s}: {acc:.2f}%{tag}")

        results_over_time.append({
            "timestep"   : t,
            "new_domain" : domain_name,
            "accuracies" : accs,
            "buffer_size": len(replay_buffer)
        })

        if t > 1:
            print(f"\n  Forgetting Analysis:")
            for prev_d in seq_loader.source_domains[:t-1]:
                best = max(r["accuracies"][prev_d] for r in results_over_time
                           if prev_d in r["accuracies"])
                curr = accs[prev_d]
                icon = "✓" if curr >= best - 1.0 else "⚠"
                print(f"    {icon} {prev_d:20s}: "
                      f"best={best:.2f}% -> now={curr:.2f}% "
                      f"(delta={curr-best:+.2f}%)")

    forgetting = compute_forgetting(results_over_time, seq_loader.source_domains)
    final_accs = results_over_time[-1]["accuracies"]
    src_accs   = [final_accs[d] for d in seq_loader.source_domains]
    avg_src    = round(sum(src_accs) / len(src_accs), 4)

    print("\n" + "="*60)
    print("FINAL RESULTS — Experience Replay")
    print("="*60)
    for d, acc in final_accs.items():
        tag = " (TARGET)" if d == TARGET_DOMAIN else ""
        print(f"  {d:20s}: {acc:.2f}%{tag}")
    print(f"\n  Avg Source Accuracy : {avg_src:.2f}%")
    print(f"  Avg Forgetting Rate : {forgetting['average']:+.2f}%")

    if forgetting["average"] < 2.0:
        print("\n  Replay works! Forgetting successfully mitigated.")
    elif forgetting["average"] < 4.0:
        print("\n  Partial improvement. Good result!")
    else:
        print("\n  Limited improvement. Consider larger buffer or higher beta.")
    print("="*60)

    results = {
        "method"            : "Experience Replay (Proposed)",
        "configuration"     : {
            "epochs_per_t": EPOCHS_PER_T, "batch_size": BATCH_SIZE,
            "lr": LR, "lambda_adv": LAMBDA_ADV, "lambda_priv": LAMBDA_PRIV,
            "beta_replay": BETA_REPLAY, "buffer_capacity": BUFFER_CAPACITY
        },
        "source_domains"    : seq_loader.source_domains,
        "target_domain"     : TARGET_DOMAIN,
        "results_over_time" : results_over_time,
        "forgetting_metrics": forgetting,
        "final_accuracies"  : final_accs,
        "avg_source_acc"    : avg_src,
        "training_time_min" : round((time.time() - start) / 60, 2)
    }

    out_path = os.path.join(RESULTS_DIR, "results_replay.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "replay_final.pth"))
    print(f"\n✓ Results saved to: {out_path}")
    print(f"✓ Model saved to  : {CKPT_DIR}/replay_final.pth\n")


if __name__ == "__main__":
    main()

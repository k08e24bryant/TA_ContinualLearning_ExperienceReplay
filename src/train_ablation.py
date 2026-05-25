"""
Ablation Study — Experience Replay
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

Supports two ablation groups:
  Group B — Variasi Buffer Size  (beta fixed = 1.0)
  Group C — Variasi Beta         (buffer fixed = 500)
  Group D — Kombinasi custom     (both configurable)

Usage:
  python src/train_ablation.py --group B --variant B1
  python src/train_ablation.py --buffer_size 300 --beta 1.0 --run_id B2
  python src/train_ablation.py --buffer_size 500 --beta 2.0 --run_id C4

Author: Syarif Sanad - 5025221257
"""

import argparse
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
# Preset Variants
# ============================================================================
VARIANTS = {
    # Group B — Variasi Buffer Size (beta=1.0 fixed)
    "B1": {"buffer_size": 100,  "beta": 1.0, "group": "B"},
    "B2": {"buffer_size": 300,  "beta": 1.0, "group": "B"},
    "B3": {"buffer_size": 500,  "beta": 1.0, "group": "B"},   # baseline
    "B4": {"buffer_size": 1000, "beta": 1.0, "group": "B"},
    "B5": {"buffer_size": 2000, "beta": 1.0, "group": "B"},

    # Group C — Variasi Beta (buffer=500 fixed)
    "C1": {"buffer_size": 500, "beta": 0.1, "group": "C"},
    "C2": {"buffer_size": 500, "beta": 0.5, "group": "C"},
    "C3": {"buffer_size": 500, "beta": 1.0, "group": "C"},   # baseline
    "C4": {"buffer_size": 500, "beta": 2.0, "group": "C"},
    "C5": {"buffer_size": 500, "beta": 5.0, "group": "C"},
}

# ============================================================================
# Fixed Hyperparameters
# ============================================================================
DATA_DIR      = r"C:\ITS\SEMESTER 8\TA\dataset\processed_acl"
ALL_DOMAINS   = ["books", "dvd", "electronics", "kitchen"]
TARGET_DOMAIN = "kitchen"
RESULTS_DIR   = r"C:\ITS\SEMESTER 8\TA\results\ablation"
CKPT_DIR      = r"C:\ITS\SEMESTER 8\TA\checkpoints\ablation"
EXPERIMENTS_DIR = r"C:\ITS\SEMESTER 8\TA\experiments"

BATCH_SIZE   = 8
LR           = 0.0001
EPOCHS_PER_T = 10
N_CRITIC     = 5
LAMBDA_ADV   = 0.1
LAMBDA_PRIV  = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study — Experience Replay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Preset variant (shortcut)
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=list(VARIANTS.keys()),
        help="Preset variant ID (e.g. B1, C4). Overrides --buffer_size and --beta."
    )

    # Manual configuration
    parser.add_argument(
        "--buffer_size", type=int, default=500,
        help="Total replay buffer capacity"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Replay loss weight (L_total = L_curr + beta * L_replay)"
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Unique run identifier for output files (e.g. D1, custom_01)"
    )
    parser.add_argument(
        "--group", type=str, default="D",
        choices=["B", "C", "D"],
        help="Experiment group label"
    )

    # Optional overrides
    parser.add_argument("--epochs",     type=int,   default=EPOCHS_PER_T)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--save_ckpt",  action="store_true",
                        help="Save model checkpoint after training")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable tqdm progress bars (for batch runs)")

    return parser.parse_args()


# ============================================================================
# Helpers (same as train_replay.py)
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
    opt_D, opt_EC, loss_cls, loss_dom,
    epochs, beta, device, disable_progress=False
):
    model.train()
    use_replay = not replay_buffer.is_empty()

    for epoch in range(1, epochs + 1):
        total_lc, total_lr, total_ld, n = 0.0, 0.0, 0.0, 0

        for x, y, d in tqdm(loader,
                             desc=f"    Epoch {epoch}/{epochs}",
                             leave=False,
                             disable=disable_progress):
            x, y, d = x.to(device), y.to(device), d.to(device)
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

            sent, dom_s, dom_p, bin_label = model(x, d, alpha=1.0, is_source=True)
            lc     = loss_cls(sent, y)
            la     = loss_dom(dom_s, bin_label)
            lp     = loss_dom(dom_p, bin_label)
            l_curr = lc + LAMBDA_ADV * la + LAMBDA_PRIV * lp

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

            # L_total = L_curr + beta * L_replay
            loss_total = l_curr + beta * l_replay
            loss_total.backward()
            opt_EC.step()

            total_lc += lc.item()
            total_lr += l_replay.item()
            total_ld += ld.item()
            n += 1

        if not disable_progress:
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
# Main Training Loop
# ============================================================================
def run_experiment(run_id, buffer_size, beta, group, epochs,
                   batch_size, lr, save_ckpt, disable_progress):

    os.makedirs(RESULTS_DIR,     exist_ok=True)
    os.makedirs(CKPT_DIR,        exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    print("\n" + "="*65)
    print(f"ABLATION — Run: {run_id}  |  Group: {group}")
    print(f"Buffer={buffer_size}  |  Beta={beta}  |  Epochs/T={epochs}")
    print(f"Device: {DEVICE}  |  LR: {lr}")
    print("="*65 + "\n")

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
        capacity=buffer_size,
        num_domains=num_sources
    )

    results_over_time = []

    for t in range(1, num_sources + 1):
        domain_name = seq_loader.source_domains[t - 1]
        domain_id   = t - 1

        print(f"\n[T={t}] '{domain_name}' (id={domain_id})")

        train_loader, test_loaders = seq_loader.get_loader_at_timestep(
            t, batch_size=batch_size
        )

        opt_D  = optim.Adam(model.D.parameters(), lr=lr)
        opt_EC = optim.Adam(
            list(model.Es.parameters()) +
            list(model.Ep.parameters()) +
            list(model.C.parameters()), lr=lr
        )

        train_one_timestep_with_replay(
            model, train_loader, replay_buffer,
            opt_D, opt_EC, loss_cls, loss_dom,
            epochs, beta, DEVICE, disable_progress
        )

        replay_buffer.add_domain_data(train_loader, domain_id=domain_id, device=DEVICE)

        accs = evaluate(model, test_loaders, DEVICE)

        print(f"  Accuracies: " +
              " | ".join(f"{d}={v:.2f}%" for d, v in accs.items()))

        results_over_time.append({
            "timestep"   : t,
            "new_domain" : domain_name,
            "accuracies" : accs,
            "buffer_size": len(replay_buffer)
        })

    forgetting  = compute_forgetting(results_over_time, seq_loader.source_domains)
    final_accs  = results_over_time[-1]["accuracies"]
    src_accs    = [final_accs[d] for d in seq_loader.source_domains]
    avg_src     = round(sum(src_accs) / len(src_accs), 4)
    elapsed_min = round((time.time() - start) / 60, 2)

    print("\n" + "="*65)
    print(f"DONE [{run_id}] | Avg Src={avg_src:.2f}% | "
          f"Target={final_accs[TARGET_DOMAIN]:.2f}% | "
          f"Forget={forgetting['average']:+.2f}% | "
          f"Time={elapsed_min}min")
    print("="*65)

    results = {
        "run_id"            : run_id,
        "group"             : group,
        "method"            : "Experience Replay (Ablation)",
        "configuration"     : {
            "buffer_size"  : buffer_size,
            "beta"         : beta,
            "epochs_per_t" : epochs,
            "batch_size"   : batch_size,
            "lr"           : lr,
            "lambda_adv"   : LAMBDA_ADV,
            "lambda_priv"  : LAMBDA_PRIV
        },
        "source_domains"    : seq_loader.source_domains,
        "target_domain"     : TARGET_DOMAIN,
        "results_over_time" : results_over_time,
        "forgetting_metrics": forgetting,
        "final_accuracies"  : final_accs,
        "avg_source_acc"    : avg_src,
        "training_time_min" : elapsed_min
    }

    # Save per-run result
    out_path = os.path.join(RESULTS_DIR, f"results_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {out_path}")

    # Save per-run experiment doc
    doc_dir  = os.path.join(EXPERIMENTS_DIR, run_id)
    os.makedirs(doc_dir, exist_ok=True)
    doc_path = os.path.join(doc_dir, "config.json")
    with open(doc_path, "w") as f:
        json.dump(results["configuration"] | {"run_id": run_id, "group": group}, f, indent=2)

    # Save checkpoint if requested
    if save_ckpt:
        ckpt_path = os.path.join(CKPT_DIR, f"ablation_{run_id}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✓ Checkpoint: {ckpt_path}")

    return results


# ============================================================================
# Entry Point
# ============================================================================
def main():
    args = parse_args()

    # Resolve config: preset variant OR manual args
    if args.variant:
        cfg      = VARIANTS[args.variant]
        run_id   = args.variant
        buffer   = cfg["buffer_size"]
        beta     = cfg["beta"]
        group    = cfg["group"]
    else:
        buffer   = args.buffer_size
        beta     = args.beta
        group    = args.group
        run_id   = args.run_id or f"{group}_buf{buffer}_beta{beta}"

    run_experiment(
        run_id          = run_id,
        buffer_size     = buffer,
        beta            = beta,
        group           = group,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        save_ckpt       = args.save_ckpt,
        disable_progress= args.no_progress
    )


if __name__ == "__main__":
    main()

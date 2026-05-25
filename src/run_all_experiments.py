"""
Run All Ablation Experiments
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

Menjalankan semua variasi Group B dan C secara otomatis,
lalu menghasilkan summary table dan grafik ablation study.

Usage:
  python src/run_all_experiments.py              # Jalankan semua (B + C)
  python src/run_all_experiments.py --group B    # Hanya Group B
  python src/run_all_experiments.py --group C    # Hanya Group C
  python src/run_all_experiments.py --dry_run    # Lihat rencana tanpa eksekusi
  python src/run_all_experiments.py --skip B3 C3 # Lewati varian yang sudah ada

Author: Syarif Sanad - 5025221257
"""

import argparse
import json
import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Experiment Plan
# ============================================================================
GROUP_B = [
    {"run_id": "B1", "buffer_size": 100,  "beta": 1.0, "group": "B"},
    {"run_id": "B2", "buffer_size": 300,  "beta": 1.0, "group": "B"},
    {"run_id": "B3", "buffer_size": 500,  "beta": 1.0, "group": "B"},  # baseline
    {"run_id": "B4", "buffer_size": 1000, "beta": 1.0, "group": "B"},
    {"run_id": "B5", "buffer_size": 2000, "beta": 1.0, "group": "B"},
]

GROUP_C = [
    {"run_id": "C1", "buffer_size": 500, "beta": 0.1, "group": "C"},
    {"run_id": "C2", "buffer_size": 500, "beta": 0.5, "group": "C"},
    {"run_id": "C3", "buffer_size": 500, "beta": 1.0, "group": "C"},  # baseline
    {"run_id": "C4", "buffer_size": 500, "beta": 2.0, "group": "C"},
    {"run_id": "C5", "buffer_size": 500, "beta": 5.0, "group": "C"},
]

RESULTS_DIR   = r"C:\ITS\SEMESTER 8\TA\results\ablation"
PLOTS_DIR     = r"C:\ITS\SEMESTER 8\TA\results\ablation\plots"
MAIN_RESULTS  = r"C:\ITS\SEMESTER 8\TA\results"

# ============================================================================
# Argument Parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all ablation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--group", type=str, default="all",
        choices=["all", "B", "C"],
        help="Which experiment group to run"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="List of run_ids to skip (e.g. --skip B3 C3)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print plan without running experiments"
    )
    parser.add_argument(
        "--no_progress", action="store_true",
        help="Disable tqdm progress bars"
    )
    parser.add_argument(
        "--summary_only", action="store_true",
        help="Skip training, only generate summary from existing results"
    )
    return parser.parse_args()


# ============================================================================
# Run single experiment (in-process, not subprocess)
# ============================================================================
def run_single(cfg, disable_progress=False):
    """Import and call run_experiment directly (faster than subprocess)."""
    from train_ablation import run_experiment
    return run_experiment(
        run_id          = cfg["run_id"],
        buffer_size     = cfg["buffer_size"],
        beta            = cfg["beta"],
        group           = cfg["group"],
        epochs          = 10,
        batch_size      = 8,
        lr              = 0.0001,
        save_ckpt       = False,
        disable_progress= disable_progress
    )


# ============================================================================
# Load all results
# ============================================================================
def load_all_results(results_dir, run_ids):
    results = {}
    for run_id in run_ids:
        path = os.path.join(results_dir, f"results_{run_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[run_id] = json.load(f)
        else:
            print(f"  [Warning] Missing: results_{run_id}.json")
    return results


# ============================================================================
# Print Summary Table
# ============================================================================
def print_summary_table(results, group_label, group_configs, vary_key):
    print(f"\n{'='*75}")
    print(f"ABLATION SUMMARY — Group {group_label}")
    print(f"{'='*75}")

    header_key = "Buffer Size" if vary_key == "buffer_size" else "Beta"
    print(f"{'Run':<6} {header_key:<14} {'Avg Src Acc':>12} "
          f"{'Target Acc':>12} {'Avg Forget':>12} {'Time (min)':>12}")
    print("-"*75)

    for cfg in group_configs:
        rid = cfg["run_id"]
        val = cfg[vary_key]
        if rid not in results:
            print(f"{rid:<6} {str(val):<14} {'(missing)':>12}")
            continue
        r   = results[rid]
        avg_src = r.get("avg_source_acc", 0)
        target  = r.get("final_accuracies", {}).get("kitchen", 0)
        forget  = r.get("forgetting_metrics", {}).get("average", 0)
        elapsed = r.get("training_time_min", 0)
        marker  = " ← baseline" if val == (500 if vary_key == "buffer_size" else 1.0) else ""
        print(f"{rid:<6} {str(val):<14} {avg_src:>11.2f}% "
              f"{target:>11.2f}% {forget:>+11.2f}% {elapsed:>10.2f}{marker}")

    print("="*75)


# ============================================================================
# Save Combined Summary JSON
# ============================================================================
def save_combined_summary(results, group_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary = []
    for run_id, r in results.items():
        cfg = r.get("configuration", {})
        summary.append({
            "run_id"       : run_id,
            "group"        : r.get("group", "?"),
            "buffer_size"  : cfg.get("buffer_size"),
            "beta"         : cfg.get("beta"),
            "avg_source_acc" : r.get("avg_source_acc"),
            "target_acc"   : r.get("final_accuracies", {}).get("kitchen"),
            "avg_forgetting" : r.get("forgetting_metrics", {}).get("average"),
            "books_acc"    : r.get("final_accuracies", {}).get("books"),
            "dvd_acc"      : r.get("final_accuracies", {}).get("dvd"),
            "electronics_acc" : r.get("final_accuracies", {}).get("electronics"),
            "books_forget" : r.get("forgetting_metrics", {}).get("books"),
            "dvd_forget"   : r.get("forgetting_metrics", {}).get("dvd"),
            "training_time_min" : r.get("training_time_min"),
        })

    path = os.path.join(output_dir, f"summary_group_{group_label}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {path}")
    return summary


# ============================================================================
# Generate Ablation Plots
# ============================================================================
def generate_ablation_plots(results_b, results_c, plots_dir):
    """Generate ablation study plots using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [Warning] matplotlib not available, skipping plots")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # ── Plot B: Buffer Size vs Performance ────────────────────────────────
    if results_b:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Group B: Effect of Buffer Size (Beta=1.0)",
                     fontsize=14, fontweight="bold")

        buf_sizes = []
        avg_srcs, targets, forgets = [], [], []
        for cfg in GROUP_B:
            rid = cfg["run_id"]
            if rid not in results_b:
                continue
            r = results_b[rid]
            buf_sizes.append(cfg["buffer_size"])
            avg_srcs.append(r.get("avg_source_acc", 0))
            targets.append(r.get("final_accuracies", {}).get("kitchen", 0))
            forgets.append(r.get("forgetting_metrics", {}).get("average", 0))

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

        for ax, values, title, ylabel in zip(
            axes,
            [avg_srcs, targets, forgets],
            ["Avg Source Accuracy", "Target Accuracy (Kitchen)", "Avg Forgetting Rate"],
            ["Accuracy (%)", "Accuracy (%)", "Forgetting Rate (%)"]
        ):
            bars = ax.bar(range(len(buf_sizes)), values,
                         color=colors[:len(buf_sizes)], alpha=0.85, edgecolor="white")
            ax.set_xticks(range(len(buf_sizes)))
            ax.set_xticklabels([str(b) for b in buf_sizes])
            ax.set_xlabel("Buffer Size", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.05,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=9)

            # Highlight baseline (buffer=500 = index 2)
            if len(bars) > 2:
                bars[2].set_edgecolor("black")
                bars[2].set_linewidth(2)

        plt.tight_layout()
        path = os.path.join(plots_dir, "ablation_group_B.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: ablation_group_B.png")

    # ── Plot C: Beta vs Performance ────────────────────────────────────────
    if results_c:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Group C: Effect of Beta (Buffer=500)",
                     fontsize=14, fontweight="bold")

        betas = []
        avg_srcs, targets, forgets = [], [], []
        for cfg in GROUP_C:
            rid = cfg["run_id"]
            if rid not in results_c:
                continue
            r = results_c[rid]
            betas.append(cfg["beta"])
            avg_srcs.append(r.get("avg_source_acc", 0))
            targets.append(r.get("final_accuracies", {}).get("kitchen", 0))
            forgets.append(r.get("forgetting_metrics", {}).get("average", 0))

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

        for ax, values, title, ylabel in zip(
            axes,
            [avg_srcs, targets, forgets],
            ["Avg Source Accuracy", "Target Accuracy (Kitchen)", "Avg Forgetting Rate"],
            ["Accuracy (%)", "Accuracy (%)", "Forgetting Rate (%)"]
        ):
            bars = ax.bar(range(len(betas)), values,
                         color=colors[:len(betas)], alpha=0.85, edgecolor="white")
            ax.set_xticks(range(len(betas)))
            ax.set_xticklabels([str(b) for b in betas])
            ax.set_xlabel("Beta (β)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.05,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=9)

            # Highlight baseline (beta=1.0 = index 2)
            if len(bars) > 2:
                bars[2].set_edgecolor("black")
                bars[2].set_linewidth(2)

        plt.tight_layout()
        path = os.path.join(plots_dir, "ablation_group_C.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: ablation_group_C.png")

    # ── Plot Combined: Line chart for direct comparison ────────────────────
    if results_b and results_c:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Ablation Study — Combined Overview",
                     fontsize=14, fontweight="bold")

        # Left: Buffer size effect (forgetting)
        ax = axes[0]
        buf_sizes, forgets = [], []
        for cfg in GROUP_B:
            if cfg["run_id"] in results_b:
                r = results_b[cfg["run_id"]]
                buf_sizes.append(cfg["buffer_size"])
                forgets.append(r.get("forgetting_metrics", {}).get("average", 0))
        if buf_sizes:
            ax.plot(buf_sizes, forgets, "o-", color="#2196F3",
                    linewidth=2, markersize=8, label="Forgetting Rate")
            ax.axvline(x=500, color="gray", linestyle="--", alpha=0.5, label="Baseline (500)")
            ax.set_xlabel("Buffer Size", fontsize=11)
            ax.set_ylabel("Avg Forgetting Rate (%)", fontsize=11)
            ax.set_title("Buffer Size vs Forgetting", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Right: Beta effect (forgetting)
        ax = axes[1]
        betas, forgets = [], []
        for cfg in GROUP_C:
            if cfg["run_id"] in results_c:
                r = results_c[cfg["run_id"]]
                betas.append(cfg["beta"])
                forgets.append(r.get("forgetting_metrics", {}).get("average", 0))
        if betas:
            ax.plot(betas, forgets, "s-", color="#4CAF50",
                    linewidth=2, markersize=8, label="Forgetting Rate")
            ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (β=1.0)")
            ax.set_xlabel("Beta (β)", fontsize=11)
            ax.set_ylabel("Avg Forgetting Rate (%)", fontsize=11)
            ax.set_title("Beta vs Forgetting", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(plots_dir, "ablation_combined.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: ablation_combined.png")


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # Determine which groups to run
    if args.group == "B":
        plan = GROUP_B
    elif args.group == "C":
        plan = GROUP_C
    else:
        plan = GROUP_B + GROUP_C

    # Apply skip list
    skip_set = set(args.skip)
    plan     = [cfg for cfg in plan if cfg["run_id"] not in skip_set]

    # Check which already exist
    already_done = []
    to_run       = []
    for cfg in plan:
        path = os.path.join(RESULTS_DIR, f"results_{cfg['run_id']}.json")
        if os.path.exists(path):
            already_done.append(cfg["run_id"])
        else:
            to_run.append(cfg)

    # Print plan
    print("\n" + "="*65)
    print("ABLATION EXPERIMENT PLAN")
    print("="*65)
    print(f"Group filter : {args.group}")
    print(f"Skip list    : {list(skip_set) or 'none'}")
    print(f"Already done : {already_done or 'none'}")
    print(f"\nWill run ({len(to_run)} experiments):")
    for cfg in to_run:
        print(f"  {cfg['run_id']:4s} | buffer={cfg['buffer_size']:4d} | beta={cfg['beta']}")

    total_est = len(to_run) * 2.25
    print(f"\nEstimated time: ~{total_est:.0f} min ({len(to_run)} × ~2.25 min each)")
    print("="*65)

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running experiments.")
        return

    if args.summary_only:
        print("\n[SUMMARY ONLY] Skipping training...")
        to_run = []

    # ── Run experiments ────────────────────────────────────────────────────
    completed = []
    failed    = []

    if to_run:
        print(f"\nStarting {len(to_run)} experiments...\n")
        wall_start = time.time()

        for i, cfg in enumerate(to_run, 1):
            print(f"\n{'='*65}")
            print(f"[{i}/{len(to_run)}] Starting {cfg['run_id']} | "
                  f"buffer={cfg['buffer_size']} | beta={cfg['beta']}")
            print(f"{'='*65}")

            try:
                run_single(cfg, disable_progress=args.no_progress)
                completed.append(cfg["run_id"])
            except Exception as e:
                print(f"\n[ERROR] {cfg['run_id']} failed: {e}")
                failed.append(cfg["run_id"])

        wall_total = (time.time() - wall_start) / 60
        print(f"\n{'='*65}")
        print(f"TRAINING COMPLETE")
        print(f"  Completed : {completed}")
        print(f"  Failed    : {failed or 'none'}")
        print(f"  Wall time : {wall_total:.1f} min")
        print(f"{'='*65}")

    # ── Load and summarize all results ─────────────────────────────────────
    print("\nGenerating summary...")

    all_b_ids = [cfg["run_id"] for cfg in GROUP_B]
    all_c_ids = [cfg["run_id"] for cfg in GROUP_C]

    results_b = load_all_results(RESULTS_DIR, all_b_ids)
    results_c = load_all_results(RESULTS_DIR, all_c_ids)

    if results_b:
        print_summary_table(results_b, "B", GROUP_B, "buffer_size")
        save_combined_summary(results_b, "B", RESULTS_DIR)

    if results_c:
        print_summary_table(results_c, "C", GROUP_C, "beta")
        save_combined_summary(results_c, "C", RESULTS_DIR)

    # ── Generate plots ─────────────────────────────────────────────────────
    if results_b or results_c:
        print("\nGenerating ablation plots...")
        generate_ablation_plots(results_b, results_c, PLOTS_DIR)

    print("\n✓ All done!")
    print(f"  Results : {RESULTS_DIR}")
    print(f"  Plots   : {PLOTS_DIR}")


if __name__ == "__main__":
    main()

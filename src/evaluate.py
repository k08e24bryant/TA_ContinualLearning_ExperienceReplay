"""
Evaluation & Visualization
Sequential Multi-Source Domain Adaptation - Sentiment Analysis

Generates 3 comparison plots:
1. Accuracy per domain per timestep (line chart)
2. Forgetting Rate comparison — Naive vs Replay (bar chart)
3. Final accuracy comparison — all three methods (bar chart)

Author: Syarif Sanad - 5025221257
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
RESULTS_DIR = r"C:\ITS\SEMESTER 8\TA\results"
PLOTS_DIR   = r"C:\ITS\SEMESTER 8\TA\results\plots"

RESULT_FILES = {
    "Oracle (Joint)"   : "results_oracle.json",
    "Naive Sequential" : "results_naive.json",
    "Experience Replay": "results_replay.json",
}

COLORS = {
    "Oracle (Joint)"   : "#2196F3",   # Blue
    "Naive Sequential" : "#F44336",   # Red
    "Experience Replay": "#4CAF50",   # Green
}

SOURCE_DOMAINS = ["books", "dvd", "electronics"]
TARGET_DOMAIN  = "kitchen"

# Display labels (capitalize properly)
DOMAIN_LABELS = {
    "books"      : "Books",
    "dvd"        : "DVD",
    "electronics": "Electronics",
    "kitchen"    : "Kitchen\n(TARGET)"
}


# ============================================================================
# Load Results
# ============================================================================
def load_results(results_dir, result_files):
    data = {}
    for method, filename in result_files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"  [Warning] Not found: {path}")
            continue
        with open(path, "r") as f:
            data[method] = json.load(f)
        print(f"  ✓ Loaded: {filename}")
    return data


# ============================================================================
# Plot 1: Accuracy per Domain per Timestep
# ============================================================================
def plot_accuracy_over_time(data, source_domains, target_domain, save_dir):
    all_domains = source_domains + [target_domain]
    n_domains   = len(all_domains)

    fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 5))
    fig.suptitle(
        "Accuracy per Domain over Sequential Timesteps",
        fontsize=14, fontweight="bold", y=1.02
    )

    num_timesteps = len(source_domains)

    for ax, domain in zip(axes, all_domains):
        # Determine y-axis range per domain
        all_vals = []

        for method, result in data.items():
            if method == "Oracle (Joint)":
                val = result["final_accuracies"].get(domain)
                if val:
                    all_vals.append(val)
            else:
                for entry in result.get("results_over_time", []):
                    if domain in entry["accuracies"]:
                        all_vals.append(entry["accuracies"][domain])

        if all_vals:
            y_min = max(50, min(all_vals) - 5)
            y_max = min(100, max(all_vals) + 3)
        else:
            y_min, y_max = 50, 100

        for method, result in data.items():
            if method == "Oracle (Joint)":
                final_acc = result["final_accuracies"].get(domain)
                if final_acc is not None:
                    ax.axhline(
                        y=final_acc,
                        color=COLORS[method],
                        linestyle="--",
                        linewidth=2,
                        label=method
                    )
            else:
                timesteps, accs = [], []
                for entry in result.get("results_over_time", []):
                    if domain in entry["accuracies"]:
                        timesteps.append(entry["timestep"])
                        accs.append(entry["accuracies"][domain])

                if timesteps:
                    ax.plot(
                        timesteps, accs,
                        color=COLORS[method],
                        marker="o",
                        linewidth=2,
                        markersize=7,
                        label=method
                    )

        title = DOMAIN_LABELS.get(domain, domain.capitalize())
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(1, num_timesteps + 1))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "plot1_accuracy_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: plot1_accuracy_over_time.png")


# ============================================================================
# Plot 2: Forgetting Rate Comparison
# ============================================================================
def plot_forgetting_rate(data, source_domains, save_dir):
    # Only domains that CAN be forgotten (all except last)
    # Last domain cannot be forgotten since nothing comes after it
    domains_with_forgetting = source_domains[:-1]  # books, dvd

    seq_methods = {
        m: d for m, d in data.items()
        if m != "Oracle (Joint)" and "forgetting_metrics" in d
    }

    if not seq_methods:
        print("  [Warning] No sequential results, skipping plot 2")
        return

    x     = np.arange(len(domains_with_forgetting))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (method, result) in enumerate(seq_methods.items()):
        forgetting = result["forgetting_metrics"]
        values     = [forgetting.get(d, 0) for d in domains_with_forgetting]
        offset     = (i - len(seq_methods) / 2 + 0.5) * width

        bars = ax.bar(
            x + offset, values,
            width,
            label=method,
            color=COLORS[method],
            alpha=0.85,
            edgecolor="white"
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    ax.set_title(
        "Forgetting Rate per Domain\n(Lower is Better)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Domain", fontsize=11)
    ax.set_ylabel("Forgetting Rate (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DOMAIN_LABELS.get(d, d.capitalize()) for d in domains_with_forgetting],
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add average forgetting annotation
    avg_texts = []
    for method, result in seq_methods.items():
        avg = result["forgetting_metrics"].get("average", 0)
        avg_texts.append(f"{method}: avg = {avg:.2f}%")

    ax.text(
        0.02, 0.97,
        "\n".join(avg_texts),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    # Add note about electronics
    ax.text(
        0.5, -0.15,
        "* Electronics excluded: last domain has no subsequent training to cause forgetting",
        transform=ax.transAxes,
        fontsize=8, ha="center", color="gray", style="italic"
    )

    plt.tight_layout()
    path = os.path.join(save_dir, "plot2_forgetting_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: plot2_forgetting_rate.png")


# ============================================================================
# Plot 3: Final Accuracy Comparison
# ============================================================================
def plot_final_accuracy(data, source_domains, target_domain, save_dir):
    all_domains = source_domains + [target_domain]
    x     = np.arange(len(all_domains))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (method, result) in enumerate(data.items()):
        final_accs = result.get("final_accuracies", {})
        values     = [final_accs.get(d, 0) for d in all_domains]
        offset     = (i - len(data) / 2 + 0.5) * width

        bars = ax.bar(
            x + offset, values,
            width,
            label=method,
            color=COLORS[method],
            alpha=0.85,
            edgecolor="white"
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.set_title(
        "Final Accuracy Comparison — All Methods",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Domain", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DOMAIN_LABELS.get(d, d.capitalize()) for d in all_domains],
        fontsize=10
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Zoom y-axis to show differences clearly
    all_vals = []
    for result in data.values():
        all_vals.extend(result.get("final_accuracies", {}).values())
    if all_vals:
        ax.set_ylim(max(50, min(all_vals) - 5), min(100, max(all_vals) + 3))

    plt.tight_layout()
    path = os.path.join(save_dir, "plot3_final_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: plot3_final_accuracy.png")


# ============================================================================
# Summary Table
# ============================================================================
def print_summary_table(data, source_domains, target_domain):
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Method':<25} {'Avg Src Acc':>12} {'Target Acc':>12} {'Avg Forget':>12}")
    print("-"*70)

    for method, result in data.items():
        avg_src = result.get("avg_source_acc", 0)
        target  = result.get("final_accuracies", {}).get(target_domain, 0)
        forget  = result.get("forgetting_metrics", {}).get("average", 0)

        forget_str = f"{forget:+.2f}%" if forget != 0 else "    N/A"
        print(f"{method:<25} {avg_src:>11.2f}% {target:>11.2f}% {forget_str:>12}")

    print("="*70)
    print("\nNotes:")
    print("  Oracle      = upper bound (joint training, no forgetting possible)")
    print("  Naive       = lower bound (sequential, no continual learning)")
    print("  Replay      = proposed method (sequential + experience replay)")
    print("  Avg Forget  = average BWT across old source domains")
    print("="*70 + "\n")


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("EVALUATION & VISUALIZATION")
    print("="*60 + "\n")

    print("Loading results...")
    data = load_results(RESULTS_DIR, RESULT_FILES)

    if not data:
        print("\n[Error] No result files found.")
        print("Run training scripts first:")
        print("  python src/train_oracle.py")
        print("  python src/train_naive.py")
        print("  python src/train_replay.py")
        return

    print(f"\nLoaded {len(data)} methods: {list(data.keys())}\n")

    print_summary_table(data, SOURCE_DOMAINS, TARGET_DOMAIN)

    print("Generating plots...")

    print("\n[Plot 1] Accuracy over time...")
    plot_accuracy_over_time(data, SOURCE_DOMAINS, TARGET_DOMAIN, PLOTS_DIR)

    print("\n[Plot 2] Forgetting rate comparison...")
    plot_forgetting_rate(data, SOURCE_DOMAINS, PLOTS_DIR)

    print("\n[Plot 3] Final accuracy comparison...")
    plot_final_accuracy(data, SOURCE_DOMAINS, TARGET_DOMAIN, PLOTS_DIR)

    print(f"\n✓ All plots saved to: {PLOTS_DIR}")
    print("  - plot1_accuracy_over_time.png")
    print("  - plot2_forgetting_rate.png")
    print("  - plot3_final_accuracy.png")
    print("\n✓ Evaluation complete!\n")


if __name__ == "__main__":
    main()

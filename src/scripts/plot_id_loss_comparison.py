"""Plot the four training-loss arms; no separate ID measurements."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "image-gnn-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = {"ce": ("CE baseline", "#3267ac"), "mle": ("CE + MLE", "#da7435"),
        "twonn": ("CE + TwoNN", "#2c916b"),
        "corrint": ("CE + smooth correlation", "#9554ad")}


def plot_results(root):
    root = Path(root)
    summary = []
    for name in ("mnist", "fmnist", "pathmnist"):
        folder = root / name
        if not (folder / "manifest.json").exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        direction = manifest.get("id_direction", "higher")
        sign = "+" if direction == "lower" else "-"
        seeds = [seed for seed in manifest["arguments"]["seeds"]
                 if all((folder / f"{arm}_seed{seed}/complete.json").exists() for arm in ARMS)]
        if not seeds:
            continue
        histories = {}
        for arm in ARMS:
            histories[arm] = []
            for seed in seeds:
                run = folder / f"{arm}_seed{seed}"
                with (run / "history.csv").open() as f:
                    histories[arm].append(list(csv.DictReader(f)))
                summary.append(json.loads((run / "complete.json").read_text()))
        panels = [("train_ce", "Training cross-entropy (eval mode)"),
                  ("test_ce", "Test cross-entropy"),
                  ("train_accuracy", "Training accuracy (eval mode)"),
                  ("test_accuracy", "Test accuracy"),
                  ("train_objective", "Training objective (during updates)"),
                  ("train_penalty", f"ID loss term: {sign}log estimate (unweighted)")]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (key, title) in zip(axes.flat, panels):
            for arm, (label, color) in ARMS.items():
                histories_for_arm = histories[arm]
                values = np.array([[float(row[key]) for row in h] for h in histories_for_arm])
                epochs = [int(row["epoch"]) for row in histories_for_arm[0]]
                mean = values.mean(axis=0)
                std = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
                ax.plot(epochs, mean, label=label, color=color, marker="." if len(epochs) == 1 else None)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=.15)
            ax.set(title=title, xlabel="Epoch")
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        settings = manifest["arguments"]
        prefix = "SMOKE TEST — " if settings["limit"] else ""
        fig.suptitle(f"{prefix}{name.upper()}: encourage {direction} ID; CE {sign} lambda*log(ID); lambda={settings['lambda_id']}\n"
                     f"{len(seeds)} paired seeds; mean +/- sample SD")
        fig.tight_layout(rect=(0, .045, 1, .94))
        fig.text(.5, .015, "Compare prediction quality with test CE and accuracy; total objectives have different offsets. No offline ID evaluation.",
                 ha="center", fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"loss_comparison.{ext}", dpi=180)
        plt.close(fig)
    if summary:
        with (root / "final_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(f"Four-arm plots and paired final metrics: {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/id_loss_comparison_lower_id")
    plot_results(parser.parse_args().results_dir)

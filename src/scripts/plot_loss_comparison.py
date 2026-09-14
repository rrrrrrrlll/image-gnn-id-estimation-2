"""Render completed paired experiments as PNG/PDF and summary CSV."""
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


def plot_results(root):
    root = Path(root)
    summary = []
    for dataset in ("mnist", "fmnist", "pathmnist"):
        folder = root / dataset
        completed = {}
        for method in ("ce", "ldreg"):
            completed[method] = {p.parent.name.split("seed")[-1]: p.parent
                                 for p in folder.glob(f"{method}_seed*/complete.json")}
        seeds = sorted(completed["ce"].keys() & completed["ldreg"].keys())
        if not seeds:
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        direction = manifest.get("id_direction", "higher")
        sign = "+" if direction == "lower" else "-"
        histories = {}
        for method in completed:
            histories[method] = []
            for seed in seeds:
                run = completed[method][seed]
                with (run / "history.csv").open() as f:
                    histories[method].append(list(csv.DictReader(f)))
                summary.append(json.loads((run / "complete.json").read_text()))
        panels = [("train_ce", "Training cross-entropy (eval mode)"),
                  ("test_ce", "Test cross-entropy"), ("test_accuracy", "Test accuracy"),
                  ("train_objective", "Training objective (during updates)"),
                  ("train_log_id", "Training mean log(local ID)"),
                  ("test_log_id", "Test mean log(local ID)")]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, (metric, title) in zip(axes.flat, panels):
            for method, color in (("ce", "#3267ac"), ("ldreg", "#da7435")):
                records = histories[method]
                values = np.array([[float(row[metric]) for row in h] for h in records])
                epochs = [int(row["epoch"]) for row in records[0]]
                avg = values.mean(axis=0)
                std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(avg)
                label = "Cross-entropy" if method == "ce" else f"CE {sign} lambda*mean(log ID) (lambda={records[0][0]['lambda_id']})"
                ax.plot(epochs, avg, label=label, color=color)
                ax.fill_between(epochs, avg - std, avg + std, color=color, alpha=.16)
            ax.set(title=title, xlabel="Epoch")
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        settings = json.loads((folder / "manifest.json").read_text())["arguments"]
        prefix = "SMOKE TEST — " if settings["limit"] else ""
        fig.suptitle(f"{prefix}{dataset.upper()}: encourage {direction} ID ({len(seeds)} seeds; mean +/- sample SD)")
        fig.tight_layout(rect=(0, .035, 1, .95))
        fig.text(.5, .012, "Total objectives have different offsets; compare classification quality using test CE and accuracy.", ha="center", fontsize=10)
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"loss_comparison.{extension}", dpi=180)
        plt.close(fig)
    if summary:
        with (root / "final_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(f"Plots and paired final metrics written under {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/loss_comparison_lower_id")
    plot_results(parser.parse_args().results_dir)

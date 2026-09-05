# Image-GNN Intrinsic Dimension Estimation (Fork 2)

This is a fork of `image-gnn-id-estimation` that extends the same VAE-based
image-to-graph learning pipeline to a wider set of datasets, several of
which arrive as **pre-computed embeddings supplied by the base paper's
authors** rather than being trained from scratch in this repository.

Datasets in this fork:

| Dataset | Origin | Entry point |
|---|---|---|
| MNIST | Trained here (VAE sweep) | Step 1 |
| FashionMNIST (FMNIST) | Trained here (VAE sweep) | Step 1 |
| PathMNIST | Trained here (VAE sweep) | Step 1 |
| CIFAR10 | Trained here (VAE sweep) | Step 1 |
| OCTMNIST | Config scaffolding only — **no data currently present in this fork** | n/a |
| FER2013 | Pre-computed embeddings from the paper's authors | Step 6 |
| CelebA-Gender (`celeba_gb`) | Pre-computed embeddings from the paper's authors | Step 6 |
| CelebA-Smiling (`celeba_sb`) | Pre-computed embeddings from the paper's authors | Step 6 |

`celeba_gb` and `celeba_sb` are **the same underlying face images and the
same VAE latent features** — only the classification label differs
(gender vs. smiling). They are two separate datasets for GNN/MLP
training and for the k value in `train_gap_curve.py` / graph
construction, but they are one dataset for Step 12 (intrinsic dimension
estimation), since ID estimation only looks at the latent features and
never the label — running Step 12 on `celeba_gb` and `celeba_sb`
separately is expected to produce identical numbers.

Because MNIST/FMNIST/PathMNIST/CIFAR10 are trained here while
FER2013/CelebA arrive pre-embedded, the pipeline has two entry points that
converge at Step 6:

```text
MNIST, FMNIST, PathMNIST, CIFAR10          FER2013, CelebA-Gender, CelebA-Smiling
        │                                              │
        ▼                                              │
Raw Images                                              │
    │                                                   │
    ▼                                                   │
CNN-VAE Training with W&B Sweep                         │
    │                                                   │
    ▼                                                   │
Best VAE Checkpoint                                      │
    │                                                   │
    ▼                                                   │
Latent Embeddings  ◄─────────────────────────────────────┘
    (data/<Dataset>Embeddings/<name>_{train,test}_embeddings.npy,
     supplied directly for FER2013/CelebA instead of generated)
    │
    ▼
SMSL Parsing
    │
    ▼
KNN Graph Construction
    │
    ▼
GNN Training / Evaluation  +  Non-Graph MLP Baseline
    │
    ▼
Intrinsic Dimension Estimation
```

Each dataset is processed independently through the complete pipeline.

---

## 1. Environment

Same environment as fork 1:

```text
Python       3.13
PyTorch      2.11.0+cu128
CUDA         12.8
PyG          2.8.0.post1
```

Install the main dependencies:

```powershell
python -m pip install numpy scipy pyyaml tqdm matplotlib wandb medmnist scikit-dimension annoy
```

Install the PyG extension packages matching PyTorch 2.11 + CUDA 12.8:

```powershell
python -m pip install --no-cache-dir pyg_lib torch_sparse torch_scatter `
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

Verify the environment:

```powershell
python -c "import torch; import pyg_lib; import torch_sparse; import torch_scatter; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:

```text
2.11.0+cu128
True
```

---

## 2. Configuration Layout

```text
config/
├── datasets/                  # VAE dataset configs -- only for the 4 trained-here datasets
│   ├── mnist.yaml
│   ├── fmnist.yaml
│   ├── octmnist.yaml          # scaffolding only, no data in this fork
│   └── pathmnist.yaml
│   └── cifar10.yaml
│
├── knn/                       # KNN graph input -- all 7 usable datasets
│   ├── mnist.yaml
│   ├── fmnist.yaml
│   ├── octmnist.yaml          # scaffolding only, no data in this fork
│   ├── pathmnist.yaml
│   ├── cifar10.yaml
│   ├── fer2013.yaml
│   ├── celeba_gb.yaml
│   └── celeba_sb.yaml
│
├── gnn/                        # GNN training input (points at the built KNN graph .pkl)
│   ├── mnist.yaml
│   ├── fmnist.yaml
│   ├── octmnist.yaml
│   ├── pathmnist.yaml
│   ├── cifar10.yaml
│   ├── fer2013.yaml
│   ├── celeba_gb.yaml
│   └── celeba_sb.yaml
│
├── mlp/                        # Non-graph MLP baseline input (points at raw, non-SMSL embeddings)
│   ├── mnist.yaml
│   ├── fmnist.yaml
│   ├── octmnist.yaml
│   ├── pathmnist.yaml
│   ├── cifar10.yaml
│   ├── fer2013.yaml
│   ├── celeba_gb.yaml
│   └── celeba_sb.yaml
│
├── models/
│   ├── cnnvae_gray.yaml
│   ├── cnnvae_rgb.yaml
│   ├── gnn.yaml
│   └── mlp.yaml
│
├── vae_training_config.yaml
├── gnn_training_config.yaml
└── wandb_cnnvae_sweep_config.yaml
```

Generated outputs are stored under:

```text
src/scripts/checkpoints/       # VAE checkpoints (4 trained-here datasets only)
data/*Embeddings/              # top-level VAE latent embeddings + smsl_embeddings/ subfolder
data/*Graph/                   # KNN graph .pkl files
results/id_estimation/         # Step 12 output, one CSV per dataset
results/gap_curve/             # Step 11a output, one CSV per dataset
results/mlp_baseline/          # MLP-baseline output, one CSV per dataset
results/train_gnn/             # SLURM .out logs, moved here manually
analysis/gap_curve_vs_id.ipynb # notebook comparing GNN vs. MLP vs. estimated ID
```

---

# Execution Guide

## Step 1 — Configure VAE Training

**Applies only to MNIST, FMNIST, PathMNIST, CIFAR10.** FER2013 and CelebA
skip Steps 1–5 entirely — their `data/<Dataset>Embeddings/<name>_{train,test}_embeddings.npy`
files were supplied pre-computed and already have the `[latent | label]`
shape Step 5 would otherwise produce.

Open:

```text
config/vae_training_config.yaml
```

Recommended real-experiment configuration:

```yaml
learning_rate: 0.0001
batch_size: 128
num_epochs: 50
save_model: True
loss: torch_vae_loss
```

---

## Step 2 — Configure the CNN-VAE Sweep

Open:

```text
config/wandb_cnnvae_sweep_config.yaml
```

Example:

```yaml
program: train_model.py
name: CNNVAE_arch_optimization
method: grid

metric:
  goal: minimize
  name: test_loss

parameters:
  latent_size:
    values: [364, 256, 128, 64, 32, 16, 8]

  block_1:
    values: [1, 2, 3, 4, 5]

  block_2:
    values: [1, 2, 3, 4, 5]

  block_3:
    values: [1, 2, 3, 4, 5]
```

`in_channels` is dataset-dependent and should not be included in the sweep.

```text
MNIST       → 1 channel
FMNIST      → 1 channel
PathMNIST   → 3 channels
CIFAR10     → 3 channels
```

Use:

```text
config/models/cnnvae_gray.yaml
config/models/cnnvae_rgb.yaml
```

---

## Step 3 — Create a W&B Sweep

Log in:

```powershell
wandb login --relogin
```

Create a new sweep for each dataset — a separate sweep per dataset, not one shared sweep:

```powershell
wandb sweep --project image-gnn config/wandb_cnnvae_sweep_config.yaml
```

Each call prints a sweep ID. `train_model.py` takes the sweep ID as a required command-line argument:

```text
--sweep_id <SWEEP_ID>      # required
--project <PROJECT_NAME>   # optional, defaults to "image-gnn"
```

Note the sweep ID for each dataset before moving to Step 4.

---

## Step 4 — Train the CNN-VAE

### MNIST

```powershell
python src/scripts/train_model.py `
    -d config/datasets/mnist.yaml `
    -m config/models/cnnvae_gray.yaml `
    -t config/vae_training_config.yaml `
    --sweep_id YOUR_MNIST_SWEEP_ID
```

### FMNIST

```powershell
python src/scripts/train_model.py `
    -d config/datasets/fmnist.yaml `
    -m config/models/cnnvae_gray.yaml `
    -t config/vae_training_config.yaml `
    --sweep_id YOUR_FMNIST_SWEEP_ID
```

### PathMNIST

```powershell
python src/scripts/train_model.py `
    -d config/datasets/pathmnist.yaml `
    -m config/models/cnnvae_rgb.yaml `
    -t config/vae_training_config.yaml `
    --sweep_id YOUR_PATHMNIST_SWEEP_ID
```

### CIFAR10

```powershell
python src/scripts/train_model.py `
    -d config/datasets/cifar10.yaml `
    -m config/models/cnnvae_rgb.yaml `
    -t config/vae_training_config.yaml `
    --sweep_id YOUR_CIFAR10_SWEEP_ID
```

Each command above points at its own dataset's sweep ID, so all four can be submitted and run at the same time — for example as separate SLURM jobs (`slurm/sweep_mnist.sh`, etc.).

Each W&B run saves its best checkpoint locally:

```text
src/scripts/checkpoints/<dataset>/<dataset>_<wandb_run_id>_best.pt
```

---

## Step 5 — Generate Latent Embeddings

**Applies only to MNIST, FMNIST, PathMNIST, CIFAR10.** For FER2013 and
CelebA, this step's output already exists — go straight to Step 6.

`gen_graph_embedding.py` automatically selects the checkpoint with the lowest stored `test_loss` for the selected dataset.

```powershell
python src/scripts/gen_graph_embedding.py -d config/datasets/mnist.yaml
python src/scripts/gen_graph_embedding.py -d config/datasets/fmnist.yaml
python src/scripts/gen_graph_embedding.py -d config/datasets/pathmnist.yaml
python src/scripts/gen_graph_embedding.py -d config/datasets/cifar10.yaml
```

Example MNIST output:

```text
data/MNISTEmbeddings/
├── mnist_train_embeddings.npy
└── mnist_test_embeddings.npy
```

Each row contains:

```text
latent features + class label
```

For FER2013 and CelebA, the equivalent files already exist at:

```text
data/FER2013Embeddings/fer2013_{train,test}_embeddings.npy
data/CELEBAEmbeddings/celeba_gb_{train,test}_embeddings.npy
data/CELEBAEmbeddings/celeba_sb_{train,test}_embeddings.npy
```

---

## Step 6 — Parse Embeddings for SMSL

Run for every dataset (all 7):

```powershell
python src/scripts/parse_graph_embedding.py -d config/datasets/mnist.yaml
python src/scripts/parse_graph_embedding.py -d config/datasets/fmnist.yaml
python src/scripts/parse_graph_embedding.py -d config/datasets/pathmnist.yaml
python src/scripts/parse_graph_embedding.py -d config/datasets/cifar10.yaml
```

FER2013 and CelebA don't have a `config/datasets/*.yaml` (no VAE training
config was needed for them), so their SMSL-parsed files were produced
separately and already exist under:

```text
data/FER2013Embeddings/smsl_embeddings/
data/CELEBAEmbeddings/gb_smsl_embeddings/
data/CELEBAEmbeddings/sb_smsl_embeddings/
```

The parsed representation is:

```text
[latent features | class label | SMSL flag]
```

`parse_graph_embedding.py` always appends `_pca_` to the filenames it
writes ("kept for compatibility with the existing repository
configuration" per its own comment) — **this is a pure VAE-embedding
passthrough with no dimensionality reduction.** See the important note in
Step 7 below: for several datasets in this fork, a genuinely
PCA-reduced file with the same `_pca_` suffix also exists, produced by a
separate step, and is easy to confuse with the passthrough one.

## Step 7 — Configure the KNN Graph Input

**This step was the source of a real bug in this fork's history — read this
section before changing anything here.**

Every `smsl_embeddings/` folder in this fork contains, depending on the
dataset, either 4 files (no true PCA reduction ever happened) or 8 files
(a second, genuinely-reduced set of files also exists, sharing the same
`_pca_` suffix as the harmless passthrough files from Step 6). The
`_pca_` suffix by itself does **not** tell you whether a file is reduced —
you have to check which of the two patterns the dataset falls into:

| Dataset | Files in `smsl_embeddings/` | Correct file for Step 7 | Dims (correct) | The `_pca_` file's actual dims |
|---|---:|---|---:|---:|
| MNIST | 8 | `mnist_smsl_{train,test}_embeddings.npy` | 130 (128 latent + label + flag) | 5 (**reduced** to 3 latent) |
| FMNIST | 8 | `fmnist_smsl_{train,test}_embeddings.npy` | 258 (256 latent + label + flag) | 5 (**reduced** to 3 latent) |
| FER2013 | 8 | `fer2013_smsl_{train,test}_embeddings.npy` | 66 (64 latent + label + flag) | 5 (**reduced** to 3 latent) |
| PathMNIST | 8 | `pathmnist_smsl_{train,test}_embeddings.npy` | 258 (256 latent + label + flag) | 10 (**reduced** to 8 latent) |
| CIFAR10 | 8 | `cifar10_smsl_{train,test}_embeddings.npy` | 4098 (4096 latent + label + flag) | 5 (**reduced** to 3 latent) |
| CelebA (gb/sb) | 4 | `celeba_smsl_{train,test}_embeddings.npy` | 130 (128 latent + label + flag) | — no `_pca_` variant exists at all |
| OCTMNIST | 0 | n/a | n/a — no data in this fork | n/a |

**The rule for every dataset in this fork: use the file WITHOUT `_pca_` in
the name.** The `_pca_`-suffixed file, where one exists, is a real,
separately-generated dimensionality reduction (very likely via
`gen_pca_embedding.py`'s hardcoded `PCAModel(n_components=8)`, confirmed
for PathMNIST and consistent with the other reduced datasets) and using it
as the KNN graph input silently trains the whole downstream pipeline on a
3–8 dimensional projection instead of the real latent space — this is
exactly what happened in this fork's config before it was corrected.

Example, now correct:

```text
config/knn/mnist.yaml
```

```yaml
train:
  base_dir: data/MNISTEmbeddings/smsl_embeddings/mnist_smsl_train_embeddings.npy

test:
  base_dir: data/MNISTEmbeddings/smsl_embeddings/mnist_smsl_test_embeddings.npy
```

The graph builder uses:

```python
full_ds[:, :-2]
```

for KNN distances, so:

```text
latent features → used for KNN distance
class label     → excluded
SMSL flag       → excluded
```

This is dimension-agnostic — `gen_knn_graphs.py` never needs to change
when you point it at a different-width file.

---

## Step 8 — Configure Annoy

The Annoy tree count is set in:

```text
src/scripts/gen_knn_graphs.py
```

Find:

```python
model = kNNModel(
    ds=full_ds[:, :-2],
    n_trees=...
)
```

`n_trees=100000` is used uniformly across all datasets in this fork, matching fork 1's convention.

---

## Step 9 — Construct the KNN Graph

`k = 100` for every dataset.

```powershell
python src/scripts/gen_knn_graphs.py -d config/knn/mnist.yaml     -k 100 --dataset_name mnist
python src/scripts/gen_knn_graphs.py -d config/knn/fmnist.yaml    -k 100 --dataset_name fmnist
python src/scripts/gen_knn_graphs.py -d config/knn/pathmnist.yaml -k 100 --dataset_name pathmnist
python src/scripts/gen_knn_graphs.py -d config/knn/cifar10.yaml   -k 100 --dataset_name cifar10
python src/scripts/gen_knn_graphs.py -d config/knn/fer2013.yaml   -k 100 --dataset_name fer2013
python src/scripts/gen_knn_graphs.py -d config/knn/celeba_gb.yaml -k 100 --dataset_name celeba_gb
python src/scripts/gen_knn_graphs.py -d config/knn/celeba_sb.yaml -k 100 --dataset_name celeba_sb
```

SLURM launchers are provided for every dataset above:

```text
slurm/gen_knn_graph_mnist.sh
slurm/gen_knn_graph_fmnist.sh
slurm/gen_knn_graph_pathmnist.sh
slurm/gen_knn_graph_cifar10.sh
slurm/gen_knn_graph_fer2013.sh
slurm/gen_knn_graph_celeba_gb.sh
slurm/gen_knn_graph_celeba_sb.sh
```

This step is CPU-only (Annoy + numpy, no `.cuda()` calls in
`gen_knn_graphs.py`), so these launchers deliberately do not request a GPU
(`--gres=gpu:1` is omitted) to avoid reserving one for work that never
uses it.

Example output:

```text
data/MNISTGraph/
├── mnist_train_knn_graph-100.pkl
└── mnist_test_knn_graph-100.pkl
```

**A caveat specific to CIFAR10 and CelebA:** the Gaussian edge-weight
kernel (`ker_width=5`) and the hard `0.75` similarity threshold in
`gen_knn_graphs.py` are fixed constants applied identically to every
dataset. CIFAR10's raw embedding (4096 dims, large per-dimension scale)
and CelebA's (128 dims but a larger vector norm than MNIST's own 128-dim
embedding) both produce typical nearest-neighbor Euclidean distances large
enough that most of their k=100 neighbor weights land at or below 0.75 and
get zeroed out — CIFAR10 far more severely than CelebA. This is a real
property of these two embeddings' scale relative to a threshold tuned
around the other datasets' smaller distances, not a bug in the script.
CIFAR10's 4096-dim input also makes `gen_knn_graphs.py`'s per-point Annoy
query loop noticeably slower (~6 it/s observed vs. 40+ it/s for the other
datasets) — budget SLURM wall-time accordingly.

## Step 10 — Configure GNN Training

Open:

```text
config/gnn_training_config.yaml
```

Current configuration in this fork (bumped from the original `num_epochs: 6`):

```yaml
learning_rate: 0.01
batch_size: 256
num_epochs: 50
save_model: False
loss: torch_ce_loss
```

Use one generic GNN configuration:

```text
config/models/gnn.yaml
```

```yaml
model: GNNModel

args:
    hidden_size: 32
    gnn_conv: GCNConv
    gnn_conv_args: {}
    layers: 2
    dropout: 0.5
```

`in_features`/`out_size` are inferred automatically from the graph's feature
dimension and class count — no code change needed when a dataset's KNN
graph input dimension changes (e.g. after the Step 7 fix above).

This same `config/gnn_training_config.yaml` is reused as-is for the MLP
baseline (Step 11b) and gap-curve runs (Step 11a), so `num_epochs: 50`
applies to all three.

---

## Step 11 — Train the GNN

```powershell
python src/scripts/train.py -d config/gnn/mnist.yaml     -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train.py -d config/gnn/fmnist.yaml    -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train.py -d config/gnn/pathmnist.yaml -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train.py -d config/gnn/cifar10.yaml   -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train.py -d config/gnn/fer2013.yaml   -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train.py -d config/gnn/celeba_gb.yaml -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train.py -d config/gnn/celeba_sb.yaml -m config/models/gnn.yaml -t config/gnn_training_config.yaml
```

SLURM launchers: `slurm/train_gnn_<dataset>.sh` for all 7 datasets (also
`train_gnn_octmnist.sh`, currently unusable — no data). These request a
GPU and are currently budgeted at `--time=01:30:00`, calibrated against a
real observed ~40-minute completion time for a 50-epoch run — cifar10
and celeba have not yet been separately re-verified against that budget
post-fix and may need adjustment once you have a real number for them.

Record at minimum:

```text
Train Loss
Test Loss
Train Accuracy
Test Accuracy
```

**Fix applied in this fork to `Trainer.parse_grow_graph()` /
`train_eval_grow_graph()`** (in `src/scripts/training/train.py`): the
original implementation trained on a single fixed 6,000-node subsample
(`n0=6000, n_increases=1`) regardless of dataset size, and never actually
grew the graph despite the name. It now builds a sequence of subgraphs at
fractions `(0.1, 0.25, 0.5, 0.75, 1.0)` of each dataset's own eligible
training-node count — consistent with `train_eval_gap_curve()`'s own
convention — and splits the configured `num_epochs` budget evenly across
those growth steps instead of multiplying it by the number of steps. This
makes Step 11 meaningfully comparable across datasets that range from
~28k (FER2013) to ~163k (CelebA) eligible nodes. This function still only
prints its results to stdout — it does not write a CSV — so `.out` logs
under `results/train_gnn/` are the only record of its output; use Step 11a
below if you need machine-readable results.

---

## Step 11a — Generalization-Gap-vs-Graph-Size Curve (optional, for the intrinsic-dimension analysis)

`Trainer.train_eval_gap_curve()`, run through `src/scripts/train_gap_curve.py`,
trains fresh models from scratch at each of several graph-size fractions
(10/25/50/75/100% of eligible training nodes by default), evaluates each
once on the held-out test set, and appends one row per (size, seed) to
`results/gap_curve/<dataset>.csv`.

```powershell
python src/scripts/train_gap_curve.py -d config/gnn/mnist.yaml     -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train_gap_curve.py -d config/gnn/fmnist.yaml    -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train_gap_curve.py -d config/gnn/pathmnist.yaml -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train_gap_curve.py -d config/gnn/cifar10.yaml   -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train_gap_curve.py -d config/gnn/fer2013.yaml   -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train_gap_curve.py -d config/gnn/celeba_gb.yaml -m config/models/gnn.yaml -t config/gnn_training_config.yaml
python src/scripts/train_gap_curve.py -d config/gnn/celeba_sb.yaml -m config/models/gnn.yaml -t config/gnn_training_config.yaml
```

Optional flags:

```text
--size-fractions   comma-separated fractions of eligible training nodes (default 0.1,0.25,0.5,0.75,1.0)
--num-seeds        independent fresh-model repeats per size (default 3)
--results-dir      output directory for the per-dataset CSV (default results/gap_curve)
```

SLURM launchers: `slurm/train_gap_curve_<dataset>.sh` for all 7 datasets
(one job per dataset, `--time=08:00:00`, calibrated conservatively rather
than measured).

`results/gap_curve/<dataset>.csv` is **append-only** — re-running this
step after a KNN-graph rebuild (e.g. after fixing Step 7's file choice)
will mix stale and fresh rows in the same file under the same dataset
name unless you move or delete the old CSV first.

---

## Step 11b — Non-Graph MLP Baseline

`Trainer.train_eval_mlp_baseline()`, run through
`src/scripts/train_mlp_baseline.py`, trains a plain MLP directly on the
Step 5 raw (non-SMSL, non-graph) embeddings — no KNN graph involved at
all — and appends one row to `results/mlp_baseline/<dataset>.csv`. This
exists to answer whether the graph structure itself is adding anything
over a graph-free baseline.

```powershell
python src/scripts/train_mlp_baseline.py -d config/mlp/mnist.yaml     -m config/models/mlp.yaml -t config/gnn_training_config.yaml
python src/scripts/train_mlp_baseline.py -d config/mlp/fmnist.yaml    -m config/models/mlp.yaml -t config/gnn_training_config.yaml
python src/scripts/train_mlp_baseline.py -d config/mlp/pathmnist.yaml -m config/models/mlp.yaml -t config/gnn_training_config.yaml
python src/scripts/train_mlp_baseline.py -d config/mlp/cifar10.yaml   -m config/models/mlp.yaml -t config/gnn_training_config.yaml
python src/scripts/train_mlp_baseline.py -d config/mlp/fer2013.yaml   -m config/models/mlp.yaml -t config/gnn_training_config.yaml
python src/scripts/train_mlp_baseline.py -d config/mlp/celeba_gb.yaml -m config/models/mlp.yaml -t config/gnn_training_config.yaml
python src/scripts/train_mlp_baseline.py -d config/mlp/celeba_sb.yaml -m config/models/mlp.yaml -t config/gnn_training_config.yaml
```

SLURM launchers: `slurm/train_mlp_<dataset>.sh` for all 7 datasets
(`--time=08:00:00`, not yet verified against a real observed runtime).

Because `config/mlp/*.yaml` points at plain (non-SMSL) top-level
embeddings, and FER2013/CelebA/PathMNIST never had those top-level files
generated by a local Step 5 run, this fork includes small "adapter" files
— the SMSL-parsed file with its trailing SMSL-flag column stripped off —
at `data/<Dataset>Embeddings/<name>_{train,test}_embeddings.npy` so
`config/mlp/*.yaml` and Step 12 have a proper raw-embeddings file to read.
This does not affect Step 7/9 — the KNN graph never reads these adapter
files.

`analysis/gap_curve_vs_id.ipynb` compares Step 11a's `fraction=1.0` rows
against Step 11b's MLP results as the GNN-vs-no-graph comparison (Step
11's own `train_eval_grow_graph()` output isn't used for this since it
writes no CSV — see the note in Step 11).

---

## Step 12 — Estimate Intrinsic Dimension

Use the original raw embedding file (top-level `data/<Dataset>Embeddings/`,
not the `smsl_embeddings/` subfolder).

```powershell
python src/scripts/gen_graph_id.py -e data/MNISTEmbeddings/mnist_train_embeddings.npy
python src/scripts/gen_graph_id.py -e data/FMNISTEmbeddings/fmnist_train_embeddings.npy
python src/scripts/gen_graph_id.py -e data/PathMNISTEmbeddings/pathmnist_train_embeddings.npy
python src/scripts/gen_graph_id.py -e data/CIFAR10Embeddings/cifar10_train_embeddings.npy
python src/scripts/gen_graph_id.py -e data/FER2013Embeddings/fer2013_train_embeddings.npy
python src/scripts/gen_graph_id.py -e data/CELEBAEmbeddings/celeba_gb_train_embeddings.npy
python src/scripts/gen_graph_id.py -e data/CELEBAEmbeddings/celeba_sb_train_embeddings.npy
```

The script reports:

```text
Levina-Bickel MLE
Two-NN
Correlation Dimension
```

Results land in `results/id_estimation/<label>.csv`, where `<label>` is
parsed from the input filename. This step never reads `config/knn/*.yaml`
or the `smsl_embeddings/` folder, so it was unaffected by the Step 7 bug
for every dataset except CIFAR10, whose entire original embedding import
into this fork turned out to be wrong (not a Step-7-style naming issue)
and has since been replaced — its stale results were moved to
`results/id_estimation/stale_wrong_cifar10_import/` rather than deleted.

As noted above, `celeba_gb` and `celeba_sb` share identical latent
features, so their ID estimates are expected to match exactly.

---

# Recommended Execution Order

Run one complete dataset at a time.

For MNIST, FMNIST, PathMNIST, CIFAR10:

```text
Raw images
  ↓
VAE sweep (Steps 1-4)
  ↓
best checkpoint
  ↓
latent embeddings (Step 5)
  ↓
SMSL parsing (Step 6)
  ↓
KNN graph (Steps 7-9)
  ↓
GNN (Step 11) + gap curve (Step 11a) + MLP baseline (Step 11b)
  ↓
intrinsic dimension (Step 12)
  ↓
record results
```

For FER2013, CelebA-Gender, CelebA-Smiling (embeddings already supplied):

```text
Supplied latent embeddings
  ↓
SMSL parsing (Step 6, if not already done)
  ↓
KNN graph (Steps 7-9)
  ↓
GNN (Step 11) + gap curve (Step 11a) + MLP baseline (Step 11b)
  ↓
intrinsic dimension (Step 12)
  ↓
record results
```

Whenever a dataset's Step 7 input file changes (for example, after fixing
which file — `_pca_` or plain — Step 7 points at), Step 9's KNN graph,
Step 11's GNN training, and Step 11a's gap curve all need rerunning for
that dataset; Step 11b (MLP baseline) and Step 12 (ID estimation) do not,
since neither depends on the KNN graph.

---

# Results

| Dataset | Best VAE Test Loss | Latent Size | GNN Test Accuracy | MLP Test Accuracy | MLE ID | TwoNN ID | Corr. ID |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | | 128 | | | | | |
| FMNIST | | 256 | | | | | |
| PathMNIST | | 256 | | | | | |
| CIFAR10 | | 4096 | | | | | |
| FER2013 | | 64 | | | | | |
| CelebA-Gender | | 128 | | | | | |
| CelebA-Smiling | | 128 | | | | | |

Pull the accuracy and ID columns from `results/gap_curve/`,
`results/mlp_baseline/`, and `results/id_estimation/` respectively —
`analysis/gap_curve_vs_id.ipynb` assembles the same numbers into the
comparison plots.

Also record:

```text
W&B sweep ID (trained-here datasets only)
W&B best run ID
checkpoint filename
KNN k (100 for all datasets)
Annoy n_trees (100000 for all datasets)
GNN epochs (50)
```

---

# Reproducibility

For a controlled comparison across datasets:

1. Use the same GNN architecture for every dataset.
2. Use the same KNN `k` — 100 across all datasets (Step 9).
3. Use the same Annoy `n_trees` — 100000 across all datasets (Step 8).
4. Keep training settings fixed (`config/gnn_training_config.yaml`) unless a dataset-specific change is intentional.
5. Double-check Step 7's file choice per dataset before trusting a result — the `_pca_` suffix does not reliably indicate whether a file is reduced in this fork (see Step 7).
6. Remember `celeba_gb`/`celeba_sb` share one underlying embedding — don't treat their ID estimates as independent evidence.
7. Record W&B sweep and run IDs for the trained-here datasets.
8. Keep smoke-test outputs separate from final experiment outputs.
9. Use fixed random seeds if exact reproducibility is required.

---

## Pipeline Summary

```text
   Raw Dataset                              Pre-computed
  (MNIST/FMNIST/                             Embeddings
  PathMNIST/CIFAR10)                    (FER2013/CelebA gb+sb)
        │                                       │
        ▼                                       │
┌───────────────┐                               │
│ CNN-VAE Sweep │                               │
└───────┬───────┘                               │
        │                                       │
        ▼                                       │
 Best Checkpoint                                │
        │                                       │
        ▼                                       │
┌───────────────┐                               │
│ VAE Embedding │  ◄────────────────────────────┘
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ SMSL Parsing  │
└───────┬───────┘
        │
        ▼
┌────────────────────────┐
│  KNN Graph Input Choice │   ← Step 7: plain file, not "_pca_"
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │   KNN Graph   │
    └───────┬───────┘
            │
   ┌────────┴────────┐
   ▼                  ▼
┌─────┐        ┌──────────────┐
│ GNN │        │ MLP Baseline │  (no graph, raw embeddings)
└──┬──┘        └──────┬───────┘
   │                  │
   └────────┬─────────┘
            ▼
     Classification
     Comparison (analysis/gap_curve_vs_id.ipynb)

                    +

             VAE Embeddings
                    │
                    ▼
          Intrinsic Dimension
      MLE / TwoNN / Correlation
```

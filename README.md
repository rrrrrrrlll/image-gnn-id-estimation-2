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

`n_trees=50` is used uniformly across all datasets in this fork.

**Fix applied in this fork to the tree count itself:** this was previously `n_trees=100000`,
inherited from fork 1's convention. Under the `kNNModel.__init__()` build-order bug described
below, that value never actually mattered -- Annoy built its trees over an empty index regardless
of `n_trees`, so an extreme value was silently free. Once the build-order fix below made Annoy
genuinely build that many trees over the full dataset, `n_trees=100000` OOM-killed the SLURM job
for every single dataset, including the smallest ones -- it's roughly 1000x Annoy's typical usage
range (tens to a few hundred trees). `n_trees=50` is a standard, well-supported value with no such
cost.

**Fix applied in this fork to `kNNModel.__init__()`** (in `src/scripts/models/models.py`): the
original implementation called `self.ann.build(n_trees)` *before* the `add_item()` loop that
populates the index. Annoy does not raise an error for this, but the search trees end up built
over an empty index, so `get_nns_by_item()` silently returns the wrong neighbors — verified
directly against a standalone Annoy index, where the wrong-order build missed a point's true
nearest neighbor entirely on a small test case. `add_item()` now runs for every point first, and
`build()` is called once afterward. **Every KNN graph generated before this fix is unreliable and
must be regenerated** (Step 9) — this is upstream of the edge-weight threshold below, so it
affects all seven datasets regardless of that fix's status.

**Fix applied in this fork to `kNNModel.__call__()`** (also in `src/scripts/models/models.py`):
Annoy's `get_nns_by_item()` includes the query point itself in its own neighbor list (it has
distance 0 to itself), and the original `__call__()` passed that straight through with no
filtering. `build_edge_index()` in `gen_knn_graphs.py` then built a genuine self-loop edge `(i, i)`
for every node `i`, with `weight = exp(-0 / ker_width**2) = 1.0` — confirmed as the source of the
weight=1.0 spike visible in the "before thresholding" histograms. It also silently cost each node
one of its `k` requested neighbor slots, so every node was really only getting `k-1` genuine
neighbors. `__call__()` now queries for `k+1` neighbors and drops the one equal to the query
index, restoring `k` genuine neighbors and removing the spurious weight=1.0 edges. **Every KNN
graph generated before this fix has one spurious self-loop edge per node and one fewer genuine
neighbor per node than intended, and must be regenerated** (Step 9).

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

**Fix applied in this fork to the edge-weight threshold in `gen_knn_graphs.py`:** the original
implementation zeroed out any edge weight below a hard, fixed `0.75` constant (with the Gaussian
kernel's `ker_width=5`), applied identically to every dataset. CIFAR10's raw embedding (4096 dims,
large per-dimension scale) and CelebA's (128 dims but a larger vector norm than MNIST's own
128-dim embedding) both produce typical nearest-neighbor Euclidean distances large enough that
most of their k=100 neighbor weights landed at or below 0.75 and got zeroed out — CIFAR10 far more
severely than CelebA — while MNIST/FMNIST/FER2013/PathMNIST kept nearly all of theirs. That was a
real property of these embeddings' scale relative to one threshold tuned around the
smaller-distance datasets, not a one-off bug, so a single fixed cutoff can't work for all seven.

The threshold was then changed to that run's own mean edge weight (`edge_weight.mean().item()`),
and after that to `edge_weight.mean() - edge_weight.std()`. Both were dataset-adaptive but still
sensitive to the shape of each dataset's own distribution: for MNIST/FMNIST/PathMNIST, the mean
landed at or near the distribution's own peak (confirmed from the before/after threshold
histograms `gen_knn_graphs.py` saves under `results/edge_weights/`), so cutting there discarded
most edges rather than "roughly half." Both mean-based versions were also skewed upward by the
weight=1.0 self-loop edges described in Step 8, before that bug was fixed.

The threshold is now the 50th percentile (median) of that run's own edge-weight distribution:
`np.percentile(edge_weight.numpy(), 50)`. A percentile threshold sidesteps both problems — it
guarantees a specific, dataset-independent fraction of edges survive (the top half by weight)
regardless of the distribution's shape, and it is no longer pulled around by the self-loop spike
now that that spike is gone. **This changes graph density for all seven datasets** — any graph
generated before this fix (including ones built under the intermediate plain-mean or
mean-std versions, and any built before the Step 8 self-loop fix) does not reflect it and needs to
be regenerated for results to be comparable across datasets.

**Fix applied in this fork to actually drop the thresholded edges:** every version of the
threshold above (`0.75`, plain mean, mean - std, and now the percentile) only ever zeroed out
`edge_weight` for the edges below it -- none of them removed those edges from `edge_index`, so
the graph saved to disk kept its full, untrimmed k-neighbor topology regardless of thresholding.
This went unnoticed because `GCNConv`'s weighted aggregation does correctly treat a zero-weight
edge as contributing nothing, but `NeighborLoader` -- used for both `sample_subgraph()`'s bounded
`num_neighbors=[10, 10]` training sampling (Step 11/11a) and the unbounded `[-1, -1]` used for
evaluation -- samples purely from `edge_index` topology and has no notion of `edge_weight`, so
training's bounded 10-per-hop draw was wasting roughly half its budget on structurally-present
but functionally dead edges. This was confirmed as the likely cause of GNN accuracy trailing the
graph-free MLP baseline (Step 11b) by a wide margin on MNIST/FMNIST/PathMNIST, and of the
generalization gap growing with graph size instead of shrinking. `gen_knn_graphs.py` now filters
`edge_index`/`edge_weight` down to `edge_weight > 0` right after thresholding, so the saved
graph's actual structure matches what thresholding was meant to produce. **This changes graph
density for all seven datasets again** — every KNN graph needs to be regenerated once more, and
every downstream gap-curve/MLP-baseline/ID-estimation result rerun, for results to reflect it.

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

**Fix applied in this fork to the evaluation `NeighborLoader` in both `train_eval_grow_graph()`
and `train_eval_gap_curve()`:** the GNN in `config/models/gnn.yaml` has `layers: 2`, so its second
graph-conv layer needs a true 2-hop neighborhood to aggregate over. Training already samples 2
hops (`sample_subgraph()`'s `NeighborLoader(..., num_neighbors=[10, 10])`), but both functions'
`test_dl` was built with a single-entry `num_neighbors=[-1]` — only 1 hop. The model's second
layer at evaluation time was aggregating over whatever nodes happened to be incidentally present
in that 1-hop batch, not the neighborhood it was trained to expect. Both are now
`num_neighbors=[-1, -1]`. This is the most likely explanation for generalization-gap curves that
were flat or increasing with graph size instead of decreasing (observed on MNIST/FMNIST/PathMNIST)
— results generated before this fix should be treated the same way as pre-Annoy-fix results: not
reliable for the final comparison.

**Fix applied in this fork to which rows `GNNModel.forward()` scores, in `models/models.py`:**
it used to `return out[batch.mask.bool()]`; a commented-out line right above it,
`# return out[:batch.batch_size]`, shows the standard, correct alternative was known but not
used. A `NeighborLoader` batch's first `batch.batch_size` rows are always the true seed nodes
being predicted; anything after that is sampled 1-/2-hop context, pulled in only to support
message-passing into the seeds, not meant to be scored itself. `batch.mask` is a leftover
whole-graph train/test flag — since `sample_subgraph()` only keeps edges between already-sampled
training nodes, that mask was true for nearly the entire local training batch (not just its
seeds), and for test batches it also picked up any test node reachable as *another* seed's
neighbor. Scoring those extra rows mixed real seed-node predictions with predictions for nodes
whose own 2-hop neighborhoods were never fully expanded — diluting the training signal and
corrupting both train and test accuracy, worse as the graph gets denser. `forward()` now returns
`out[:batch.batch_size]`, and every loss/accuracy computation in `train_eval_grow_graph()` and
`train_eval_gap_curve()` slices `batch.y[:batch.batch_size]` to match (`train_eval_mlp_baseline()`
doesn't use `NeighborLoader` at all, so it's unaffected and unchanged).

**Fix applied in this fork to add `model.train()`/`model.eval()` calls:** none of `Trainer`'s
methods ever switched the model between training and evaluation mode — `nn.Module` defaults to
training mode and nothing changed that, so `nn.Dropout` (`config/models/gnn.yaml`'s `dropout: 0.5`)
and `GNNBasicBlock`'s `nn.BatchNorm1d` stayed in training behavior during the "test" loop too:
test accuracy was being computed with dropout still randomly zeroing units and BatchNorm still
using per-batch statistics, not a clean inference pass. `model.train()` now runs before each
epoch's training loop and `model.eval()` before its test loop, in `train()`,
`train_eval_grow_graph()`, `train_eval_gap_curve()`, and `train_eval_mlp_baseline()`.

**Both of the above changes affect every dataset's results (Step 11, Step 11a, and Step 11b).**
Results generated before this fix should be treated the same way as pre-Annoy-fix results: not
reliable for the final comparison — everything needs rerunning once more, on top of the KNN-graph
edge-pruning fix in Step 9.

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
--num-seeds        independent fresh-model repeats per size (default 3). Either a single
                   int (same count at every fraction) or a comma-separated list the same
                   length as --size-fractions, one seed count per fraction -- useful since
                   small fractions sample noisier subgraphs and benefit from more seeds
                   while large fractions are already fairly stable with few.
--results-dir      output directory for the per-dataset CSV (default results/gap_curve)
```

Example pairing a log-spaced, small-n-skewed fraction list (for relating the gap curve
to intrinsic dimension) with more seeds at the noisier, smaller fractions:

```powershell
python src/scripts/train_gap_curve.py -d config/gnn/mnist.yaml -m config/models/gnn.yaml -t config/gnn_training_config.yaml `
    --size-fractions 0.005,0.01,0.02,0.05,0.1,0.25,0.5,1.0 `
    --num-seeds 10,10,10,10,5,5,5,1
```

SLURM launchers: `slurm/train_gap_curve_<dataset>.sh` for all 7 datasets
(one job per dataset, `--time=08:00:00`, calibrated conservatively rather
than measured).

`results/gap_curve/<dataset>.csv` is **append-only** — re-running this
step after a KNN-graph rebuild (e.g. after fixing Step 7's file choice)
will mix stale and fresh rows in the same file under the same dataset
name unless you move or delete the old CSV first.

**Regularized vs. non-regularized comparison (for Block 6 of `gap_curve_vs_id.ipynb`):**
`gap_curve_vs_id.ipynb`'s Block 6 compares each dataset's generalization-gap curve under
the default model config against the same curve with regularization removed, to check
whether it's regularization -- not graph size -- that's actually controlling the gap.
That comparison needs a second set of gap-curve results, run with a second model config:

```text
config/models/gnn_noreg.yaml   # identical to config/models/gnn.yaml except dropout: 0
```

Only `dropout` is toggled here -- `weight_decay` is hardcoded to `0.0` in
`Trainer.train_eval_gap_curve()`'s optimizer for every run regardless of model config, so it
is identical (0.0) in both the regularized and non-regularized runs and is not part of this
comparison.

```powershell
python src/scripts/train_gap_curve.py -d config/gnn/mnist.yaml     -m config/models/gnn_noreg.yaml -t config/gnn_training_config.yaml --results-dir results/gap_curve_noreg
python src/scripts/train_gap_curve.py -d config/gnn/fmnist.yaml    -m config/models/gnn_noreg.yaml -t config/gnn_training_config.yaml --results-dir results/gap_curve_noreg
python src/scripts/train_gap_curve.py -d config/gnn/pathmnist.yaml -m config/models/gnn_noreg.yaml -t config/gnn_training_config.yaml --results-dir results/gap_curve_noreg
python src/scripts/train_gap_curve.py -d config/gnn/cifar10.yaml   -m config/models/gnn_noreg.yaml -t config/gnn_training_config.yaml --results-dir results/gap_curve_noreg
python src/scripts/train_gap_curve.py -d config/gnn/fer2013.yaml   -m config/models/gnn_noreg.yaml -t config/gnn_training_config.yaml --results-dir results/gap_curve_noreg
python src/scripts/train_gap_curve.py -d config/gnn/celeba_gb.yaml -m config/models/gnn_noreg.yaml -t config/gnn_training_config.yaml --results-dir results/gap_curve_noreg
python src/scripts/train_gap_curve.py -d config/gnn/celeba_sb.yaml -m config/models/gnn_noreg.yaml -t config/gnn_training_config.yaml --results-dir results/gap_curve_noreg
```

SLURM launchers: `slurm/train_gap_curve_<dataset>_noreg.sh` for all 7 datasets, identical to
each dataset's `slurm/train_gap_curve_<dataset>.sh` except for the model config and
`--results-dir` above (same `--time`/`--mem` per dataset as the regularized launcher).
Results land in `results/gap_curve_noreg/<dataset>.csv` -- same append-only caveat as
`results/gap_curve/` above, and the same schema, so `gap_curve_vs_id.ipynb`'s existing loader
reads it with no changes once the CSVs exist.

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

`gap_curve_vs_id.ipynb`'s Block 5 compares Step 11a's `fraction=1.0` rows
against Step 11b's MLP results as the GNN-vs-no-graph comparison (Step
11's own `train_eval_grow_graph()` output isn't used for this since it
writes no CSV — see the note in Step 11), saving a two-panel test-accuracy
and generalization-gap bar chart to `analysis/block5_gnn_vs_mlp.png`.

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

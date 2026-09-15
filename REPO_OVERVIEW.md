# DLSSNet — Repository Overview

Research codebase for **DLSSNet**, a deep-learning model for EEG motor-imagery classification on the **BCI Competition IV dataset 2a** (4-class: left hand / right hand / foot / tongue). This is the code accompanying an EEG/BCI research project (NITT), not a packaged library — it is a personal experiment repo with training scripts, baseline model implementations, and analysis notebooks.

The committed `README.md` is a 4-line stub ("Detailed version Readme would be available soon"), so this document reconstructs the project from the source.

## 1. What the project does

It classifies single-trial motor-imagery EEG (22 channels, ~438 timepoints/trial after cropping) into 4 classes, and compares a novel architecture (DLSSNet) against several published baselines, training one model **per subject** (subject-dependent, not cross-subject) on the 9 subjects of BCI IV-2a.

Beyond classification accuracy, the repo's real focus is **interpretability**: DLSSNet is built so its internal representations can be inspected — pooled time frames, learned "state"/basis vectors, spatial topomaps, t-SNE separability, and K-means clustering of the learned states.

## 2. The DLSSNet model (`Model/DLSSNet/`)

Defined in `model_zhengjiao.py` / `layers_zhengjiao.py` ("zhengjiao" = 正交, Chinese for *orthogonal*, referring to the orthogonality constraint described below). Pipeline:

1. **PatchEmbedding** — two constrained-conv blocks (`Conv2dWithConstraint`, the EEGNet-style max-norm conv) reminiscent of EEGNet/Conformer's shallow net:
   - *TaskPotentialConvBlock*: a depthwise-style conv across all channels (spatial filter).
   - *TemporalStateConvBlock*: a temporal conv with stride, turning the trial into a sequence of `num_frame` "state" embeddings of dimension `d_model`.
2. **Encoder** — a stack of Transformer-like `EncoderLayer`s, each combining:
   - A **learnable graph-U-Net-style pooling gate** (`Pool`/`p_filter`, ported from Graph U-Net) that scores each time frame and produces a soft/hard attention **mask** — effectively learning which frames are informative and suppressing/attending selectively to the rest (a form of adaptive temporal pooling).
   - Standard scaled dot-product multi-head self-attention (`MultiHeadAttention_Enc`) applied under that mask.
   - A position-wise feed-forward block.
3. **ClassificationTransHead** — appends `statenum` learnable, **orthogonally-initialized** "summary tokens" (basis states) to the sequence and lets them attend (via `nn.MultiheadAttention`) to the encoded frames; the resulting basis-state vectors are concatenated and passed through an MLP to produce class logits. `statenum` (how many basis states to learn) is a swept hyperparameter (`GroupTraining.py` iterates over `data_config.statenum`, e.g. `np.arange(23,24)` or wider ranges).
4. **DecoderLayer** — cross-attends the *masked, embedded* input back onto the learned basis states and reconstructs it, i.e. the basis states must be sufficient to reconstruct the (pooled) embedded signal — an autoencoder-style regularizer on what the basis states capture.
5. **Orthogonality loss** (`utils/My_loss.py:reg_loss`) — penalizes `‖WWᵀ − I‖_F` on the basis-state weight matrix, pushing the learned basis states to be mutually orthogonal (distinct, non-redundant "prototypes").

Total training loss = classification cross-entropy + `dec_loss_alpha` × decoder MSE reconstruction loss + orthogonality regularization (see `Trainer/trainermodel_decoder_zhengjiaoyueshu_single_withouttest.py:myloss`).

This is why the checkpoint files are literally named `Conv+TransFormer+GPoolingV2+ClassTransHead+decoder+正交约束_best_params.pkl` — that name *is* the architecture description (Conv frontend + Transformer + Graph-Pooling-v2 + Classification-Transformer-Head + Decoder + orthogonality constraint).

## 3. Baseline models compared against (`Model/`)

| Folder | Model | Notes |
|---|---|---|
| `ConFormer/`, `ConFormer_forAnalysis/` | EEG Conformer | Conv "shallownet" + Transformer encoder + classification head |
| `baseLine_MATT/` | M-ATT | Riemannian/SPD-matrix attention model (`spd.py`, `signal2spd`), from CECNL/MAtt |
| `baseline_FBCNet/` | FBCNet, EEGNet, DeepConvNet | `networks.py` implements all three (credited to Ravikiran Mane) |

Pretrained weights for all of these (9 subject-folds each) are checked into `Netweights/` (DLSSNet, EEGNet, EEGconformer/EEGConformerwithoutpooling, deepconv, matt) as `.pkl` files.

## 4. Data pipeline (`DataLoader/`)

- `GetBci2a.py` loads the **preprocessed** BCI IV-2a `.mat` files (`BCIC_S<NN>_T.mat` / `_E.mat`) — the processed format from the [CECNL/MAtt](https://github.com/CECNL/MAtt) repo, as stated in the README.
- Crops each trial to timepoints `[124:562]` (motor-imagery window).
- `getAllDataloader_withouttest` merges the official train+test sessions and re-splits them into train/validation only (no held-out test set) — this is the loader actually used by the training scripts.
- `split_train_valid_set` does a per-class, ratio-based split (not random shuffle) to keep class balance.
- `interaug` (in `DataLoader/DataLoader.py`, used by the trainer when `data_augment_inTraining=True`) implements segmentation-and-recombination augmentation (mixing time segments across trials of the same class), a well-known EEG data-augmentation trick.

## 5. Training entry points

- **`SingleTraining.py`** — trains one DLSSNet-variant model per subject with a fixed `statenum`, subject-dependent (loops over `data_config.subs`), writes TensorBoard logs + checkpoints per subject-fold.
- **`GroupTraining.py`** — same idea but additionally sweeps `data_config.statenum` per subject, so for every subject it trains one model per candidate number-of-basis-states and records which `statenum` gave the best validation accuracy (this is how the paper likely justifies its choice of state count).
- Both scripts follow the same shape: read `args_*` config class → build `DataLoader` → build `Net` → optimizer/scheduler/loss from strings via `eval(...)` on the config → hand off to a `Trainer` class → dump text summaries + TensorBoard scalars per fold.
- Config classes (`Model/*/args_*.py`) are plain classes (not argparse/YAML) with **hardcoded absolute paths** to data/log directories (e.g. `/home/wangkaixuan/Code/data/BCICIV_2a_mat/`, `/disks/HDD2/wangkaixuan/ExperimentResult/...`, and `C:/2023Experiment/...`, `C:/同步文件夹/...` inside the notebooks) — these are the original author's machine paths and must be edited before the code will run elsewhere.

## 6. Analysis / visualization

- **`415Analysis数据可分性.ipynb`** / **`..._baseline.ipynb`** ("data separability analysis") — loads trained weights, extracts pooled/embedded features, and runs t-SNE to visually compare class separability between DLSSNet and baselines.
- **`VisualizePatternsandClusterCenters.ipynb`** — extracts the learned spatial (`TaskPotentialConvBlock`) and temporal (`TemporalStateConvBlock`) filter weights and plots them as EEG scalp topomaps via `mne` (`Visualization/plot_topomap.py`); also runs K-means (`kmeans_clustering.py`, added in the most recent feature commit) on the pooled "state" representations to find prototype clusters, using silhouette score to pick K, then reconstructs cluster-center patterns and does an SVD-based decomposition of the topomap. Outputs: `kmeans_cluster_centers0*.npy`, `kmeans_cluster_labels0*.npy`, `kmeans_clustering_results.png`, `kmeans_silhouette_scores.png`, `cluster_label_distribution.png`.
- **`250214统计试次准确.ipynb`** ("per-trial accuracy statistics") — loads every model's checkpoint per subject, runs inference, and writes per-trial correctness to `results/predicts.xlsx`.
- **`Analysis/computCMandCI.ipynb`** — reads `results/predicts.xlsx`, builds confusion matrices per model/subject with `pycm`, and computes binomial confidence intervals on accuracy (`scipy.stats.binom`) — i.e. proper statistical comparison of DLSSNet vs. baselines, not just point-accuracy.
- **`results/*.html`** — per-subject, per-model interactive plots (likely Plotly, e.g. training curves or attention/probability visualizations) for DLSSNet, EEG Conformer, EEGNet, deepconvNet, and matt.

## 7. Repository structure

```
DataLoader/     BCI IV-2a .mat loading, train/val split, augmentation
Model/          DLSSNet + baselines (ConFormer, MATT, FBCNet/EEGNet/DeepConvNet)
Trainer/        Per-experiment training loops (loss, epoch loop, checkpointing)
Metrics/        AverageMeter, accuracy helper
Visualization/  MNE topomap plotting
utils/          orthogonality loss, misc data/arg/visualize utils
Netweights/     Pretrained .pkl checkpoints, 9 subject-folds x 6 model variants
results/        Per-subject/per-model HTML plots + predicts/results .xlsx
Analysis/       Confusion-matrix + confidence-interval notebook
*.ipynb (root)  Separability (t-SNE), pattern/cluster visualization, trial-accuracy stats
kmeans_*.py/.npy/.png   K-means clustering of learned states (most recent feature)
SingleTraining.py, GroupTraining.py   Top-level training entry points
```

## 8. Notable issues / things a new contributor should know

- **No environment/dependency file** — no `requirements.txt`, `environment.yml`, or `setup.py`. Inferred dependencies from imports: `torch`, `torchmetrics`, `einops`, `scipy`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `mne`, `pycm`, `openpyxl`, `tensorboard`.
- **No LICENSE file.**
- **Hardcoded, machine-specific absolute paths** throughout configs and notebooks (Linux paths like `/home/wangkaixuan/...`, `/disks/HDD2/...`, and Windows paths like `C:/2023Experiment/...`) — the repo is not runnable out-of-the-box; a user must edit `data_path`, `savepath`, and notebook `path`/`savepath` variables first.
- **Both top-level training scripts currently have unresolved imports** in this snapshot:
  - `SingleTraining.py` imports `Model.GPoolingwithdecoderForAnalysis3...` — this directory does not exist anywhere in the repo.
  - `GroupTraining.py` imports `Trainer.trainermodel_multistatenum` — no such file exists (only `Trainer/trainermodel_decoder_zhengjiaoyueshu_single_withouttest.py` is present as source).
  - Corroborating evidence: several `__pycache__/*.pyc` files reference source modules that no longer exist in the repo at all (e.g. `Model/DLSSNet/{args_BCI,layers,model}.py`, `Trainer/{trainer,trainer_matt,trainermodel_decoder}.py`), meaning earlier versions of these files existed and were deleted from the git history/working tree without updating the scripts that import them.
  - **Net effect:** as committed, neither `SingleTraining.py` nor `GroupTraining.py` will run; only the notebooks (which import the still-present `Model.DLSSNet.model_zhengjiao` / `Model.DLSSNet.args_zjBCIsingletrain_val` directly) and `kmeans_clustering.py` are currently self-consistent.
- **Compiled `.pyc` files and large binary checkpoints (`Netweights/**/*.pkl`) are committed to git** — this bloats the repository; typically these would be in `.gitignore`.
- **Mixed Chinese/English** naming, comments, and variable names throughout (this is a single-author research repo, not intended as a public library).
- **No test suite** — correctness is judged entirely via the analysis notebooks/results, not unit tests.

## 9. Recent activity (git log)

```
48ee537 Update README.md
3a36b36 Update README.md
befdf32 feat: 添加K-means聚类分析功能并可视化结果   (add K-means clustering + visualization)
d61dc9d update statistic results
678f09b update
24399a6 update
4729a76 Update README.md
7ce205a Initial commit
```
The most recent substantive change is the K-means clustering analysis (`kmeans_clustering.py` + its outputs), used inside `VisualizePatternsandClusterCenters.ipynb` to cluster DLSSNet's learned "state" representations and inspect whether they form interpretable, separable prototypes.

## 10. Published paper vs. this repository

This repo is the official implementation for **"Interpretable dual-layer state space net for capturing fine-grained temporal features in EEG-based motor imagery BCIs"** — Kaixuan Wang, Cong Xu, Wenhao Jiang, Shihang Ding, Qinglong Liu, Lin Ma, Haifeng Li (Harbin Institute of Technology), *Biomedical Signal Processing and Control* 120 (2026) 109826. Confirmed via: the paper's stated GitHub link (`github.com/ReidWa/DLSSNet.git`) matches this repo's commit author handle `ReidWa`, and first author "Kaixuan Wang" matches the `wangkaixuan` path baked into `Model/DLSSNet/args_*.py`. The paper is included at `researach_papers/Interpretable dual-layer state space net for capturing fine-grained temporal features in EEG-based motor imagery BCIs.pdf` (note: folder is misspelled "researach_papers", and is currently untracked in git).

### What the paper claims

DLSSNet treats EEG decoding as a two-layer state-space / microstate-analysis problem. Layer 1 extracts short spatiotemporal "Functional Patterns" (FPs) per electrode group; Layer 2 spans a "Functional State Space" over multiple FPs — critically, **no temporal pooling is used**, so full time resolution is preserved (their "TIoF" = Temporal Integrity of Features metric; DLSSNet is the only compared model that scores 100% on it). Three named modules feed a classifier:

| Paper module | Code equivalent |
|---|---|
| Functional State Space Representation (FSSR) | `Embedding` (`PatchEmbedding`): spatial conv (`TaskPotentialConvBlock`) → temporal conv (`TemporalStateConvBlock`, kernel=16, stride=1) → pointwise conv (`project`). Paper fixes k=60 kernels, matching the released checkpoints' `d_model=60` exactly. |
| Task-Relevance Evaluation (TRE) | `Pool` class in `layers_zhengjiao.py` — sigmoid-scales each state by task-relevance and thresholds (`p_filter`, `low_p`) to separate task-relevant vs. task-irrelevant states. |
| Attention-Based Embedding (ABE) — contextual embedding + subspace encode/decode | `ClassificationTransHead` (its orthogonally-initialized `summary_token` = the paper's trainable orthonormal subspace vectors `S`) + `DecoderLayer` (cross-attention reconstruction, paper Eq. A.5). |
| Classifier | `self.fc` MLP inside `ClassificationTransHead`. |

Reported results: 78.51% avg accuracy on BCI IV-2a, 95.49% on the High-Gamma Dataset (HGD), beating EEGNet/DeepConvNet/M-ATT/EEG Conformer/MSFCNet/AMFTCNet baselines; an ablation study (removing TRE, removing ABE, replacing ABE with plain attention) shows every module contributes.

The paper's real contribution is a **post-hoc interpretability protocol** (Fig. 2): extract Functional State sequences for correctly-classified trials → K-means cluster them into task-relevant prototype states + one task-irrelevant cluster (K chosen via SSE-elbow combined with task-relevance-probability criteria, Appendix B.1) → reconstruct each prototype as an EEG scalp topomap via a "feature inversion" weighted-sum-of-kernels + SVD (Appendix B.2/B.3) → show these topomaps match known Sensorimotor Rhythm (SMR) patterns (C3/C4 contralateral activation for hand tasks, bilateral/central for tongue/foot) and are stable across subjects (Fig. 10).

### Verified: the released checkpoints are the exact ones behind Table 1

Evaluating each `Netweights/DLSSNet/ckpl/FoldNN/*_best_params.pkl` checkpoint with `DataLoader.GetBci2a.getAllDataloader_toT_E` (train on session **T**, test on session **E** — a true cross-session split) reproduces the paper's Table 1 per-subject accuracies exactly: S01 85.76%, S02 61.81%, S03 92.36%, S04 71.88%, S05 70.14% (off by one trial out of 288 — rounding), S06 57.29%, S07 93.40%, S08 88.19%, S09 85.76%, avg 78.51%.

This matters because `SingleTraining.py` — the actual training entry point — calls a **different** loader, `getAllDataloader_withouttest`, which pools sessions T+E together and re-splits them class-balanced in-distribution, rather than the true cross-session T-train/E-test protocol the paper describes in Section 3.3 ("trained on the first recording session... tested on... the second session"). `getAllDataloader_toT_E` (the function that actually matches the paper and reproduces the released checkpoints) exists in the same `GetBci2a.py` file but is never called by any entry-point script — effectively dead code from the training scripts' point of view, despite being what actually produced the paper's numbers.

### Other concrete gaps between the paper and the current code

- **No HGD (High-Gamma Dataset) assets.** Table 2's 95.49% result has no corresponding data or checkpoints in this repo — only BCI IV-2a is reproducible locally.
- **No dedicated ablation scripts/checkpoints** for the FTC/FAC/FToAC variants in Fig. 4 — reproducing that experiment requires hand-modifying the model to drop/replace modules.
- **K-selection in `kmeans_clustering.py` doesn't match the paper's stated method.** Appendix B.1 selects the cluster count K via an SSE-elbow criterion *combined with* task-relevance-probability thresholds (Eq. B.1–B.2 — a joint, multi-criterion selection). The repo's `find_optimal_k()` only maximizes silhouette score — simpler, and not the paper-specified criterion. This is notable because it's the most recently added feature in the repo (the K-means clustering commit) and doesn't implement the published method for choosing K.
- **The Contextual Embedding Block's task-relevance mask isn't applied where the paper describes it.** Appendix A.3 / Eq. A.3 says self-attention in the ABE module should mask out task-irrelevant states during contextual embedding. In code, `ClassificationTransHead.forward()` calls `self.classATTN(x, x, x)` with no mask argument; the `mask_index` computed by `Pool` (TRE) is only ever used to build the decoder's reconstruction target (`y = embeded_x * mask_index`), never as an attention mask in the classification head itself.

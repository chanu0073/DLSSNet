"""
Cross-validated hyperparameter tuning for DLSSNet -- session T only.

Why this exists: choosing hyperparameters on the single 36-trial validation
split does not work. Measured on this repo's own 27 augmented runs, the
within-subject correlation between validation and test accuracy is
r = +0.106 (p = 0.60), and selecting the best-validation seed per subject
scored 70.33% on test versus 70.45% for simply averaging the seeds -- i.e.
selection on 36 trials was worth -0.12 points. A sweep decided that way
picks the luckiest configuration, not the best one.

This harness averages over k stratified folds of session T instead, so the
decision uses all 288 trials (binomial std err 7.2 -> 2.6 points at p=0.75).
With --folds 8 each fold is 252 train / 36 validation, exactly the split
sizes of the reporting protocol, so the chosen hyperparameters transfer.

Session E is never opened: getCVFolds() cannot read it, and the trainer is
called with testloader=None so no test number exists to report. This script
CANNOT tell you a test accuracy -- that is the point. Once a config wins
here, evaluate it with TrainCorrectedProtocol.py on all 9 subjects x 3
seeds, touching session E once.

Usage:
    DLSSNET_DATA_PATH=/path/to/BCICIV_2a_mat DLSSNET_GPU=0 \\
        python TuneCV.py --configs baseline small_dmodel --subjects 3 4 6

    python TuneCV.py --list                    # show configurations
    python TuneCV.py                           # every config, default subjects
    python TuneCV.py --folds 4                 # cheaper, noisier (std err 3.6)

Defaults to subjects 3 / 4 / 6 -- easy / middling / hard, by the mean over
six architectures (81.3 / 59.7 / 47.8). Tuning only on easy subjects would
hide changes that fail where the headroom actually is.
"""
import argparse
import json
import os
import random
import sys

import numpy as np

for _name, _repl in [("long", int), ("ulong", int), ("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _name):
        setattr(np, _name, _repl)

import torch

from Model.DLSSNet.args_corrected_protocol import data_config as BASE
from Model.DLSSNet.model_zhengjiao import Net
from DataLoader.GetBci2a import getCVFolds
from Trainer.trainer_corrected_protocol import CorrectedProtocolTrainer
from utils.gpu_limit import cap_gpu_memory

# Each entry overrides attributes of args_corrected_protocol.data_config.
# "baseline" (no overrides) is the 70.45% configuration, so every sweep
# carries its own control.
CONFIGS = {
    "baseline": {},
    # Capacity: MAtt wins this benchmark with 0.009M parameters against
    # DLSSNet's 0.496M, and every capacity ADDITION tried here has lost
    # (Mamba -5.54, SPD -4.64). If 252 trials/subject is the binding
    # constraint, shrinking should help.
    "small_dmodel": {"d_model": 32},
    "small_states": {"statenum": 12},
    "small_both": {"d_model": 32, "statenum": 12},
    "tiny": {"d_model": 24, "statenum": 8},
    # Optimisation: lr/dropout came from the paper, tuned under the LEAKY
    # protocol, so there is no reason they suit this one.
    "lr_5e4": {"lr": 5e-4, "optimizer_parm": {"lr": 5e-4, "betas": (0.5, 0.999)}},
    "lr_1e4": {"lr": 1e-4, "optimizer_parm": {"lr": 1e-4, "betas": (0.5, 0.999)}},
    "dropout_6": {"dropout": 0.6},
    # Augmentation strength: S&R at aug_rate 1.0 was worth +3.51 points, the
    # only change that has ever helped. More of it may help more.
    "aug_2x": {"aug_rate": 2.0},
    "aug_seg16": {"aug_segments": 16},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_config(overrides, extra):
    # A subclass, so every `cfg.attr` lookup in the trainer/model still works.
    return type("TunedConfig", (BASE,), {**overrides, **extra})


def build_model(cfg):
    return Net(
        num_channels=cfg.num_channel, len_window=cfg.len_window, d_model=cfg.d_model,
        frame_stride=cfg.frame_stride, num_frame=cfg.num_frame, num_head=cfg.num_head,
        encoder_num_layers=cfg.encoder_num_layers, low_p=cfg.low_p, dropout=cfg.dropout,
        transformerparwiseforward_dimrat=cfg.transformerparwiseforward_dimrat,
        statenum=cfg.statenum, num_class=cfg.num_class,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--subjects", nargs="+", type=int, default=[3, 4, 6])
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=None, help="override (changes conditions vs the final protocol)")
    ap.add_argument("--patience", type=int, default=None, help="override (changes conditions vs the final protocol)")
    ap.add_argument("--out", default=os.environ.get("DLSSNET_TUNE_OUT", "TuneResults"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, ov in CONFIGS.items():
            print(f"  {name:14s} {ov if ov else '(unchanged -- the 70.45% control)'}")
        return

    extra = {}
    if args.max_epochs is not None:
        extra["max_epochs"] = args.max_epochs
    if args.patience is not None:
        extra["patience"] = args.patience
    if extra:
        print(f"WARNING: {extra} differs from the reporting protocol; a config that wins", flush=True)
        print("         under a shorter budget may not win under the full one.", flush=True)

    device = torch.device(f"cuda:{BASE.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}", flush=True)
    cap_gpu_memory(device)
    print(f"data_path: {BASE.data_path}", flush=True)
    print(f"tuning on subjects {args.subjects}, {args.folds}-fold, seed {args.seed}", flush=True)
    print("session E is never loaded by this script\n", flush=True)

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "cv_results.json")
    rows = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            rows = json.load(f)
        print(f"resuming: {len(rows)} fold-run(s) already done\n", flush=True)
    done = {(r["config"], r["subject"], r["fold"]) for r in rows}

    for cname in args.configs:
        cfg = make_config(CONFIGS[cname], extra)
        for subid in args.subjects:
            folds = getCVFolds(subid, args.folds, BASE.data_path, BASE.batch_size)
            for f, (trainloader, validloader) in enumerate(folds):
                if (cname, subid, f) in done:
                    continue
                set_seed(args.seed)
                model = build_model(cfg).to(device)
                opt = eval(cfg.optimizer)(model.parameters(), **cfg.optimizer_parm)
                trainer = CorrectedProtocolTrainer(model, device, cfg)
                ckpt = os.path.join(args.out, "ckpl", cname, f"s{subid:02d}_f{f}.pkl")
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                # testloader=None -> no test number is computed at all
                res = trainer.run(trainloader, validloader, None, opt, ckpt_path=ckpt, log_every=10**9)
                print(
                    f"[{cname}] subject{subid:02d} fold{f}: val={res['best_val_acc']:.4f}"
                    f"@epoch{res['best_epoch']} ({res['elapsed_s']:.0f}s)", flush=True
                )
                rows.append(dict(
                    config=cname, subject=int(subid), fold=int(f),
                    val_acc=float(res["best_val_acc"]), best_epoch=int(res["best_epoch"]),
                    elapsed_s=float(res["elapsed_s"]),
                ))
                tmp = results_path + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(rows, fh, indent=2)
                os.replace(tmp, results_path)

    summarise(rows, args, results_path)


def summarise(rows, args, results_path):
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config"], {}).setdefault(r["subject"], []).append(r["val_acc"])

    lines = [f"CV tuning: {args.folds}-fold on session T, subjects {args.subjects}, seed {args.seed}",
             "(validation only -- session E never touched; re-run the winner with "
             "TrainCorrectedProtocol.py for a test number)", ""]
    ranked = []
    for cname, per_sub in by_cfg.items():
        sub_means = [np.mean(v) for v in per_sub.values()]
        ranked.append((np.mean(sub_means), cname, per_sub))
    ranked.sort(reverse=True)

    base = next((m for m, c, _ in ranked if c == "baseline"), None)
    header = f"{'config':14s} {'mean val':>9s}  " + "  ".join(f"S{s}" for s in args.subjects)
    lines.append(header + ("   vs baseline" if base is not None else ""))
    lines.append("-" * (len(header) + 16))
    for mean, cname, per_sub in ranked:
        cells = "  ".join(f"{np.mean(per_sub.get(s, [np.nan]))*100:4.1f}" for s in args.subjects)
        delta = "" if base is None else ("  (control)" if cname == "baseline" else f"  {(mean-base)*100:+5.2f}")
        lines.append(f"{cname:14s} {mean*100:9.2f}  {cells}{delta}")
    n = args.folds * len(args.subjects)
    lines += ["", f"each mean averages {n} fold-runs; binomial std err at p=0.75 is "
                  f"~{np.sqrt(.75*.25/(36*args.folds))*100:.1f} points per subject.",
              "treat gaps smaller than that as ties and prefer the simpler config."]
    out = "\n".join(lines)
    with open(os.path.join(os.path.dirname(results_path), "cv_summary.txt"), "w") as f:
        f.write(out + "\n")
    print("\n" + out, flush=True)


if __name__ == "__main__":
    main()

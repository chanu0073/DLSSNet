"""
Re-run the in-repo baselines under the corrected protocol.

The published DLSSNet comparison table is not usable as-is: its numbers come
from a protocol where session E served as BOTH the early-stopping signal and
the reported result (see TrainCorrectedProtocol.py). This script retrains
each baseline under the corrected 3-way split -- same data, same S&R
augmentation, same epochs/patience/seeds/optimizer as DLSSNet -- so the
comparison table is legitimate.

Usage:
    DLSSNET_DATA_PATH=/path/to/BCICIV_2a_mat DLSSNET_GPU=0 \\
        python TrainBaselines.py --models eegnet deepconvnet -- 1 2 3

    python TrainBaselines.py                      # all models, all subjects, all seeds
    python TrainBaselines.py --models matt        # one model, all subjects
    python TrainBaselines.py --list               # show available models

Results go to ExperimentResults_Baselines/<model>/{results.json,summary.txt}.
Each model keeps its own directory, so models can be run as separate
parallel processes without clobbering each other.
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

from Model.DLSSNet.args_baselines import data_config
from DataLoader.GetBci2a import getAllDataloader
from Trainer.trainer_baseline import BaselineTrainer, FilterBank
from utils.gpu_limit import cap_gpu_memory


def _eegnet(cfg):
    from Model.baseline_FBCNet.networks import eegNet

    return eegNet(nChan=cfg.num_channel, nTime=cfg.timepoint, nClass=cfg.num_class)


def _deepconvnet(cfg):
    from Model.baseline_FBCNet.networks import deepConvNet

    return deepConvNet(nChan=cfg.num_channel, nTime=cfg.timepoint, nClass=cfg.num_class)


def _fbcnet(cfg):
    from Model.baseline_FBCNet.networks import FBCNet

    return FBCNet(nChan=cfg.num_channel, nTime=cfg.timepoint, nClass=cfg.num_class, nBands=9)


def _conformer(cfg):
    from Model.ConFormer.Conformer import Conformer

    return Conformer(emb_size=40, depth=6, n_classes=cfg.num_class)


def _matt(cfg):
    from Model.baseLine_MATT.mAtt import mAtt_bci

    return mAtt_bci(epochs=3)  # m=3 temporal segments, as in the MAtt paper for BCI IV-2a


# adapt: fixes each model's input convention. riemannian: needs MAtt's
# Stiefel-manifold optimizer, since its SPD transforms are constrained.
BASELINES = {
    "eegnet": dict(build=_eegnet, adapt=lambda x: x.unsqueeze(1)),
    "deepconvnet": dict(build=_deepconvnet, adapt=None),
    "conformer": dict(build=_conformer, adapt=None),
    "matt": dict(build=_matt, adapt=None, riemannian=True),
    "fbcnet": dict(build=_fbcnet, adapt="filterbank"),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, spec):
    opt = eval(data_config.optimizer)(model.parameters(), **data_config.optimizer_parm)
    if not spec.get("riemannian"):
        return opt
    from Model.baseLine_MATT.optimizer import MixOptimizer

    class CheckpointableMixOptimizer(MixOptimizer):
        # MixOptimizer wraps a normal optimizer but exposes only
        # zero_grad/step; the checkpoint-resume path also needs state.
        def state_dict(self):
            return self.optimizer.state_dict()

        def load_state_dict(self, sd):
            self.optimizer.load_state_dict(sd)

    return CheckpointableMixOptimizer(opt)


def run_model(name, spec, subs, device):
    out_dir = os.path.join(data_config.Result_PATH, name)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.json")
    summary_path = os.path.join(out_dir, "summary.txt")

    all_results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"[{name}] resuming: {len(all_results)} run(s) already done", flush=True)
    completed = {(r["subject"], r["seed"]) for r in all_results}

    adapt = spec["adapt"]
    if adapt == "filterbank":
        adapt = FilterBank().to(device)

    for subid in subs:
        for seed in data_config.seeds:
            if (subid, seed) in completed:
                print(f"[{name}] subject {subid:02d} seed {seed}: already completed, skipping", flush=True)
                continue

            print(f"=== [{name}] subject {subid:02d} seed {seed} ===", flush=True)
            set_seed(seed)

            trainloader, validloader, testloader = getAllDataloader(
                subject=subid, ratio=data_config.ratio, data_path=data_config.data_path, bs=data_config.batch_size
            )
            model = spec["build"](data_config).to(device)
            # The vendored MAtt/SPD code hardcodes torch.device('cpu') and
            # forces intermediate tensors back to CPU inside forward(), which
            # breaks GPU training. Retarget those attributes instead of
            # editing the vendored baseline. No-op for the other models.
            for mod in model.modules():
                if getattr(mod, "dev", None) is not None:
                    mod.dev = device
                if getattr(mod, "device", None) is not None:
                    mod.device = device
            optimizer = make_optimizer(model, spec)

            ckpt_dir = os.path.join(data_config.MODEL_PATH, name, f"subject{subid:02d}_seed{seed}")
            os.makedirs(ckpt_dir, exist_ok=True)

            trainer = BaselineTrainer(model, device, data_config, adapt=adapt)
            result = trainer.run(
                trainloader, validloader, testloader, optimizer,
                ckpt_path=os.path.join(ckpt_dir, "resume_checkpoint.pkl"),
            )

            torch.save(
                {
                    "net_state_dict": result["model_state"],
                    "best_epoch": result["best_epoch"],
                    "best_val_acc": result["best_val_acc"],
                    "test_acc": result["test_acc"],
                },
                os.path.join(ckpt_dir, "best_params.pkl"),
            )
            print(
                f"[{name}] subject{subid:02d} seed{seed}: best_val_acc={result['best_val_acc']:.4f}"
                f"@epoch{result['best_epoch']} TEST_ACC={result['test_acc']:.4f} ({result['elapsed_s']:.0f}s)",
                flush=True,
            )

            all_results.append(
                dict(
                    subject=int(subid), seed=int(seed),
                    best_val_acc=float(result["best_val_acc"]), best_epoch=int(result["best_epoch"]),
                    test_acc=float(result["test_acc"]), elapsed_s=float(result["elapsed_s"]),
                )
            )
            tmp = results_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(all_results, f, indent=2)
            os.replace(tmp, results_path)

    per_subject = {}
    for r in all_results:
        per_subject.setdefault(r["subject"], []).append(r["test_acc"])
    lines, means = [], []
    for subid in sorted(per_subject):
        accs = np.array(per_subject[subid])
        lines.append(f"subject{subid:02d}: {accs.mean():.4f} +/- {accs.std():.4f} (n={len(accs)})")
        means.append(accs.mean())
    lines += ["", f"OVERALL AVG (mean of per-subject means): {np.mean(means):.4f}"]
    summary = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\n[{name}]\n{summary}\n", flush=True)
    return np.mean(means) if means else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(BASELINES), choices=list(BASELINES))
    ap.add_argument("--list", action="store_true", help="list available models and exit")
    ap.add_argument("subjects", nargs="*", type=int, help="subject ids (default: all)")
    args = ap.parse_args()

    if args.list:
        print("available models:", ", ".join(BASELINES))
        return

    device = torch.device(f"cuda:{data_config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}", flush=True)
    cap_gpu_memory(device)
    print(f"data_path: {data_config.data_path}", flush=True)
    print(f"results under: {data_config.Result_PATH}", flush=True)

    subs = args.subjects or list(data_config.subs)
    overall = {name: run_model(name, BASELINES[name], subs, device) for name in args.models}

    print("=== overall (mean of per-subject means) ===", flush=True)
    for name, acc in overall.items():
        print(f"  {name:12s} {acc:.4f}", flush=True)


if __name__ == "__main__":
    main()

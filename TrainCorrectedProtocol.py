"""
Legitimate re-baseline for DLSSNet on BCI IV-2a.

SingleTraining.py (via DataLoader.GetBci2a.getAllDataloader_withouttest) pools
sessions T+E and re-splits them in-distribution -- not the cross-session
protocol the paper describes. Even the paper-matching loader
(getAllDataloader_toT_E) uses session E as BOTH the model-selection signal
(checkpointing "best" whenever session-E accuracy improves) AND the number
that gets reported -- i.e. no independent held-out test set at all.

This script uses DataLoader.GetBci2a.getAllDataloader(), which does a correct
3-way split: session T -> train/val (val used only for early stopping),
session E -> held out, touched exactly once, for the reported "test_acc".

Usage:
    DLSSNET_DATA_PATH=/path/to/BCICIV_2a_mat DLSSNET_GPU=0 \
        python TrainCorrectedProtocol.py [subject_ids...]

    # e.g. train subjects 1-3 only:
    python TrainCorrectedProtocol.py 1 2 3

    # with no subject args: trains every subject in data_config.subs, all seeds
"""
import json
import os
import random
import sys

import numpy as np

# Guards against a broken scipy/numpy pairing seen on some machines (an old
# scipy build referencing numpy aliases removed in numpy>=1.24, e.g.
# `np.long`). Harmless no-op on a clean environment -- only sets an attribute
# if it's actually missing.
for _name, _repl in [("long", int), ("ulong", int), ("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _name):
        setattr(np, _name, _repl)

import torch

from Model.DLSSNet.args_corrected_protocol import data_config
from Model.DLSSNet.model_zhengjiao import Net
from DataLoader.GetBci2a import getAllDataloader
from Trainer.trainer_corrected_protocol import CorrectedProtocolTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model():
    return Net(
        num_channels=data_config.num_channel,
        len_window=data_config.len_window,
        d_model=data_config.d_model,
        frame_stride=data_config.frame_stride,
        num_frame=data_config.num_frame,
        num_head=data_config.num_head,
        encoder_num_layers=data_config.encoder_num_layers,
        low_p=data_config.low_p,
        dropout=data_config.dropout,
        transformerparwiseforward_dimrat=data_config.transformerparwiseforward_dimrat,
        statenum=data_config.statenum,
        num_class=data_config.num_class,
    )


def main():
    device = torch.device(f"cuda:{data_config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}", flush=True)
    print(f"data_path: {data_config.data_path}", flush=True)
    print(f"results will be saved under: {data_config.Result_PATH}", flush=True)

    subs = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else list(data_config.subs)
    if data_config.bad_subs is not None:
        subs = [s for s in subs if s not in data_config.bad_subs]

    os.makedirs(data_config.Result_PATH, exist_ok=True)
    results_path = os.path.join(data_config.Result_PATH, "results.json")
    summary_path = os.path.join(data_config.Result_PATH, "summary.txt")

    all_results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"resuming: loaded {len(all_results)} already-completed run(s) from {results_path}", flush=True)

    completed = {(r["subject"], r["seed"]) for r in all_results}

    for subid in subs:
        for seed in data_config.seeds:
            if (subid, seed) in completed:
                print(f"=== subject {subid:02d} seed {seed}: already completed, skipping ===", flush=True)
                continue

            print(f"=== subject {subid:02d} seed {seed} ===", flush=True)
            set_seed(seed)

            trainloader, validloader, testloader = getAllDataloader(
                subject=subid,
                ratio=data_config.ratio,
                data_path=data_config.data_path,
                bs=data_config.batch_size,
            )

            model = build_model().to(device)
            optimizer = eval(data_config.optimizer)(model.parameters(), **data_config.optimizer_parm)

            ckpt_dir = os.path.join(data_config.MODEL_PATH, f"subject{subid:02d}_seed{seed}")
            os.makedirs(ckpt_dir, exist_ok=True)
            resume_ckpt_path = os.path.join(ckpt_dir, "resume_checkpoint.pkl")

            trainer = CorrectedProtocolTrainer(model, device, data_config)
            result = trainer.run(trainloader, validloader, testloader, optimizer, ckpt_path=resume_ckpt_path)

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
                f"subject{subid:02d} seed{seed}: best_val_acc={result['best_val_acc']:.4f}"
                f"@epoch{result['best_epoch']} TEST_ACC={result['test_acc']:.4f} "
                f"({result['elapsed_s']:.0f}s)",
                flush=True,
            )

            all_results.append(
                dict(
                    # plain int()/float() -- data_config.subs is a numpy array,
                    # so subid can be numpy.int64, and result values can be
                    # numpy scalars too; none of those are JSON-serializable.
                    subject=int(subid),
                    seed=int(seed),
                    best_val_acc=float(result["best_val_acc"]),
                    best_epoch=int(result["best_epoch"]),
                    test_acc=float(result["test_acc"]),
                    elapsed_s=float(result["elapsed_s"]),
                )
            )

            # Write atomically -- a crash/error mid-dump must never leave a
            # truncated results.json behind, since that file is read back on
            # every resume attempt (a corrupt file would break resuming the
            # very run this is meant to protect).
            tmp_results_path = results_path + ".tmp"
            with open(tmp_results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            os.replace(tmp_results_path, results_path)

    # Summary: mean +/- std per subject across seeds, then overall average.
    per_subject = {}
    for r in all_results:
        per_subject.setdefault(r["subject"], []).append(r["test_acc"])

    lines = []
    subject_means = []
    for subid in sorted(per_subject):
        accs = np.array(per_subject[subid])
        lines.append(f"subject{subid:02d}: {accs.mean():.4f} +/- {accs.std():.4f} (n={len(accs)})")
        subject_means.append(accs.mean())
    lines.append("")
    lines.append(f"OVERALL AVG (mean of per-subject means): {np.mean(subject_means):.4f}")

    summary = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()

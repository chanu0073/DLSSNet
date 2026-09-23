"""
Merge the results.json files written by parallel training processes into one
combined results.json + summary.txt.

Running the sweep as several concurrent processes is the only way to use a
big shared GPU efficiently here: the model is small enough (~0.5M params,
batch 10) that a single process leaves the GPU idle between kernel launches,
so runs overlap almost for free. Each process must write to its own
DLSSNET_SAVE_PATH, otherwise they overwrite each other's results.json --
this script stitches those back together afterwards.

Usage:
    python merge_results.py results_aug_1_2_3 results_aug_4_5_6 results_aug_7_8_9
    python merge_results.py results_aug_*            # shell glob works too
    python merge_results.py results_aug_* -o ExperimentResults_Merged
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result directories to merge")
    ap.add_argument("-o", "--out", default="ExperimentResults_Merged", help="output directory")
    args = ap.parse_args()

    merged, seen = [], {}
    for d in args.dirs:
        path = os.path.join(d, "results.json")
        if not os.path.exists(path):
            print(f"  skipping {d}: no results.json")
            continue
        with open(path) as f:
            rows = json.load(f)
        print(f"  {d}: {len(rows)} run(s)")
        for r in rows:
            key = (r["subject"], r["seed"])
            if key in seen:
                # Same subject/seed in two directories: the runs are
                # independent, so silently averaging them would be wrong.
                print(f"    WARNING: duplicate subject{key[0]:02d} seed{key[1]} (also in {seen[key]}) -- keeping the first")
                continue
            seen[key] = d
            merged.append(r)

    if not merged:
        print("nothing to merge")
        return

    os.makedirs(args.out, exist_ok=True)
    merged.sort(key=lambda r: (r["subject"], r["seed"]))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(merged, f, indent=2)

    per_subject = {}
    for r in merged:
        per_subject.setdefault(r["subject"], []).append(r["test_acc"])

    lines, subject_means = [], []
    for subid in sorted(per_subject):
        accs = np.array(per_subject[subid])
        lines.append(f"subject{subid:02d}: {accs.mean():.4f} +/- {accs.std():.4f} (n={len(accs)})")
        subject_means.append(accs.mean())
    lines.append("")
    lines.append(f"OVERALL AVG (mean of per-subject means): {np.mean(subject_means):.4f}")
    summary = "\n".join(lines)

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nmerged {len(merged)} run(s) over {len(per_subject)} subject(s) -> {args.out}/\n")
    print(summary)


if __name__ == "__main__":
    main()

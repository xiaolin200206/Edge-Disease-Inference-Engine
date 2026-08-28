#!/usr/bin/env python3
"""
c04_architecture_stats.py - Reviewer 1, revision 2, Comment 4.

Formal statistical comparison of the three architectures on the paired seeds,
and an explicit separation of the two sources of uncertainty the reviewer asks
to be distinguished:

  (i)  training stochasticity   - seed-to-seed variation, the same validation
                                  set re-scored by independently initialised
                                  runs;
  (ii) validation-sample uncertainty - the same checkpoint re-scored on
                                  bootstrap resamples of the 126 validation
                                  images.

The point of the exercise is not to find a significant difference. It is to
report what the design can and cannot resolve. With three paired seeds the
exact sign-flip permutation test has eight possible outcomes, so the smallest
attainable two-sided p-value is 0.25: no architectural difference of any
magnitude could be declared significant at alpha = 0.05 under this design.
That is a property of the number of seeds, not of the effect, and it is
reported as such.

Inputs are the aggregate metrics of Table 2, transcribed here so the script is
runnable without the checkpoints (which are not released); they match
data/training_logs/multiseed/rtdetr_3seeds.json for RT-DETR-l.

Usage:  python c04_architecture_stats.py [--out results_c04.json]
"""

import argparse
import itertools
import json
import math

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- Table 2 ---
MAP50 = {
    "YOLOv8s":   {0: 0.3483, 1: 0.3947, 2: 0.4251, 3: 0.4662, 4: 0.4389},
    "YOLOv11s":  {0: 0.4311, 1: 0.4317, 2: 0.3706},
    "RT-DETR-l": {0: 0.4593, 1: 0.4638, 2: 0.4359},
}
MAP5095 = {
    "YOLOv8s":   {0: 0.2748, 1: 0.2692, 2: 0.2991, 3: 0.2782, 4: 0.2980},
    "YOLOv11s":  {0: 0.2983, 1: 0.2696, 2: 0.2848},
    "RT-DETR-l": {0: 0.3554, 1: 0.3445, 2: 0.3422},
}

# Bootstrap interval on aggregate mAP@0.5 reported in Section 4.2.2, used only
# to contrast the two uncertainty sources.
BOOTSTRAP_MAP50_CI = (0.286, 0.621)


def exact_sign_flip(d):
    """Two-sided exact permutation test on paired differences.

    Under the null the sign of each paired difference is exchangeable, so the
    reference distribution is the 2**n sign assignments enumerated in full.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    stats_ = [abs(np.sum(d * np.array(s)) / n)
              for s in itertools.product([1, -1], repeat=n)]
    p = sum(1 for s in stats_ if s >= obs - 1e-12) / len(stats_)
    return p, len(stats_), 1.0 / len(stats_) * 2 if n else float("nan")


def paired_block(a_name, b_name, table, seeds):
    a = np.array([table[a_name][s] for s in seeds])
    b = np.array([table[b_name][s] for s in seeds])
    d = a - b
    p_perm, n_perm, _ = exact_sign_flip(d)
    t, p_t = stats.ttest_rel(a, b)
    try:
        w, p_w = stats.wilcoxon(a, b, mode="exact")
    except Exception:
        w, p_w = float("nan"), float("nan")
    sd = d.std(ddof=1)
    # Cohen's dz, and the difference that WOULD have been detectable at
    # alpha = 0.05 given this sd and this n - reported to make the power
    # limitation concrete rather than rhetorical.
    tcrit = stats.t.ppf(0.975, len(d) - 1)
    mde = tcrit * sd / math.sqrt(len(d))
    ci = (d.mean() - tcrit * sd / math.sqrt(len(d)),
          d.mean() + tcrit * sd / math.sqrt(len(d)))
    return {
        "pair": f"{a_name} - {b_name}",
        "seeds": list(seeds),
        "per_seed_difference": [round(float(x), 4) for x in d],
        "mean_difference": round(float(d.mean()), 4),
        "sd_difference": round(float(sd), 4),
        "ci95_difference": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "paired_t": round(float(t), 3),
        # Six decimals, not four. The paper quotes three, and a value stored
        # at four can round differently from the true one when a reader
        # formats it again: p = 0.7985388 stores as 0.7985, which reformats
        # to 0.798 while the true value gives 0.799.
        "p_paired_t": round(float(p_t), 6),
        "wilcoxon_W": None if math.isnan(w) else float(w),
        "p_wilcoxon_exact": None if math.isnan(p_w) else round(float(p_w), 4),
        "p_exact_sign_flip": round(float(p_perm), 4),
        "n_permutations": n_perm,
        "min_attainable_two_sided_p": round(2.0 / n_perm, 4),
        "cohen_dz": round(float(d.mean() / sd), 3) if sd else None,
        "minimum_detectable_difference_alpha05": round(float(mde), 4),
    }


def spread_block(name, table):
    v = np.array(list(table[name].values()), dtype=float)
    return {
        "architecture": name,
        "n_seeds": len(v),
        "mean": round(float(v.mean()), 4),
        "sd": round(float(v.std(ddof=1)), 4),
        "range": [round(float(v.min()), 4), round(float(v.max()), 4)],
        "spread": round(float(v.max() - v.min()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_c04.json")
    a = ap.parse_args()

    shared = (0, 1, 2)
    out = {"metric": "mAP@0.5", "shared_seeds": list(shared)}

    out["per_architecture_spread"] = [spread_block(k, MAP50) for k in MAP50]
    out["paired_comparisons"] = [
        paired_block("YOLOv11s", "YOLOv8s", MAP50, shared),
        paired_block("RT-DETR-l", "YOLOv8s", MAP50, shared),
        paired_block("RT-DETR-l", "YOLOv11s", MAP50, shared),
    ]
    out["paired_comparisons_map5095"] = [
        paired_block("YOLOv11s", "YOLOv8s", MAP5095, shared),
        paired_block("RT-DETR-l", "YOLOv8s", MAP5095, shared),
        # The third pairing was omitted from earlier versions of this script.
        # It matters: its parametric p is 0.009, smaller than the 0.030 the
        # manuscript had described as the only parametric p below 0.05. Leaving
        # a comparison out made a claim about the set look stronger than the
        # set supports, which is the failure mode this paper is about.
        paired_block("RT-DETR-l", "YOLOv11s", MAP5095, shared),
    ]

    # ------------------------------------------------ uncertainty contrast ---
    v8 = np.array(list(MAP50["YOLOv8s"].values()))
    seed_sd = float(v8.std(ddof=1))
    boot_w = BOOTSTRAP_MAP50_CI[1] - BOOTSTRAP_MAP50_CI[0]
    out["uncertainty_decomposition"] = {
        "training_stochasticity": {
            "definition": "sd of aggregate mAP@0.5 across independently seeded "
                          "runs of one architecture, same validation set",
            "yolov8s_sd_5_seeds": round(seed_sd, 4),
            "yolov8s_range_5_seeds": round(float(v8.max() - v8.min()), 4),
            "implied_95pct_interval_width": round(2 * 1.96 * seed_sd, 4),
        },
        "validation_sample_uncertainty": {
            "definition": "95% bootstrap interval for one checkpoint over "
                          "2,000 image-level resamples of the 126 validation "
                          "images (Section 4.2.2)",
            "ci95": list(BOOTSTRAP_MAP50_CI),
            "width": round(boot_w, 4),
        },
        "ratio_validation_to_training": round(boot_w / (2 * 1.96 * seed_sd), 2),
        "nominal_architecture_difference": 0.017,
        "reading": "The validation-sample interval is wider than the seed "
                   "interval by the ratio above, and both exceed the nominal "
                   "architecture difference. The two are independent and are "
                   "not additive; neither alone leaves room to resolve 0.017.",
    }

    print(json.dumps(out, indent=2))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

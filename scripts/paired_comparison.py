# парное сравнение двух серий запусков (для абляций).
# пары собираются по (последовательность, фрагмент, зерно), r = ATE_B / ATE_A,
# G = exp(mean(ln r)) - среднее геометрическое, G < 1 значит B точнее.
# p-value - критерий Уилкоксона по ln r.
# зёрна одного фрагмента зависимы, поэтому ещё проверка по фрагментам
# (ln r усреднён по зёрнам) и доверительный интервал G бутстрепом по фрагментам
#   python scripts/paired_comparison.py A_dir B_dir [--mode-a sync] [--mode-b sync]
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SEQS = ["V1_01_easy", "V1_02_medium", "MH_01_easy", "MH_05_difficult"]


def load(root, mode):
    out = {}
    for f in Path(root).glob("*/metrics.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        mt = re.match(rf"f(\d+)_{mode}_s(\d+)$", m["tag"])
        if mt:
            out[(m["sequence"], int(mt[1]), int(mt[2]))] = m["ate_rmse_cm"]
    return out


ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--mode-a", default="sync")
ap.add_argument("--mode-b", default="sync")
ap.add_argument("--label-a", default="A")
ap.add_argument("--label-b", default="B")
ap.add_argument("--json", default=None)
a = ap.parse_args()

A, B = load(a.a, a.mode_a), load(a.b, a.mode_b)
by_seq = defaultdict(list)
for k in sorted(set(A) & set(B)):
    by_seq[k[0]].append((A[k], B[k]))

res = {}
print(f"{a.label_a} -> {a.label_b}")
print(f"{'последовательность':<18}{'пар':>5}{'мед A,см':>10}{'мед B,см':>10}{'G=B/A':>8}{'B лучше':>9}{'p (Уилкоксон)':>15}")
all_lr = []
for seq in SEQS + ["все"]:
    if seq == "все":
        pairs = [p for s in SEQS for p in by_seq.get(s, [])]
    else:
        pairs = by_seq.get(seq, [])
    if not pairs:
        continue
    x = np.array(pairs)
    lr = np.log(x[:, 1] / x[:, 0])
    p = float(wilcoxon(lr).pvalue) if len(lr) >= 6 and np.any(lr != 0) else float("nan")
    r = {"n": len(lr), "med_a": float(np.median(x[:, 0])), "med_b": float(np.median(x[:, 1])),
         "G": float(np.exp(lr.mean())), "b_better": int((lr < 0).sum()), "p": p}
    res[seq] = r
    print(f"{seq:<18}{r['n']:>5}{r['med_a']:>10.1f}{r['med_b']:>10.1f}{r['G']:>8.2f}{r['b_better']:>5}/{r['n']:<3}{p:>15.3f}")

# то же по фрагментам: зёрна усредняем, бутстреп по фрагментам
keys = sorted(set(A) & set(B))
frags = sorted({k[:2] for k in keys})
lr_by = {f: np.array([np.log(B[k] / A[k]) for k in keys if k[:2] == f]) for f in frags}
fm = np.array([lr_by[f].mean() for f in frags])
rng = np.random.default_rng(0)
boot = [np.concatenate([lr_by[frags[j]] for j in rng.integers(0, len(frags), len(frags))]).mean()
        for _ in range(10000)]
lo, hi = (float(v) for v in np.exp(np.percentile(boot, [2.5, 97.5])))
p_f = float(wilcoxon(fm).pvalue) if len(fm) >= 6 and np.any(fm != 0) else float("nan")
res["по фрагментам"] = {"n": len(fm), "G": float(np.exp(fm.mean())), "b_better": int((fm < 0).sum()),
                        "p": p_f, "ci95": [lo, hi]}
print(f"по фрагментам: n = {len(fm)}, B лучше на {int((fm < 0).sum())}, p = {p_f:.4f}; "
      f"95%-й доверительный интервал G: [{lo:.2f}; {hi:.2f}]")
if a.json:
    Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

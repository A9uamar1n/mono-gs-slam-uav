# основная таблица по фрагментам для каждой последовательности и режима:
# ATE фрагмента = среднее по зёрнам, дальше медиана и диапазон по фрагментам;
# запуск успешный, если ATE <= 5% пути; RPE, PSNR и время - медианы по запускам
#   python scripts/table_fragments.py [outputs/stage5] [--ok-pct 5] [--json out.json]
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SEQS = ["V1_01_easy", "V1_02_medium", "MH_01_easy", "MH_05_difficult"]

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="outputs/stage5")
ap.add_argument("--ok-pct", type=float, default=5.0)
ap.add_argument("--json", default=None)
a = ap.parse_args()

runs = defaultdict(list)   # (seq, mode) -> список запусков
for f in Path(a.root).glob("*/metrics.json"):
    m = json.loads(f.read_text(encoding="utf-8"))
    mt = re.match(r"f(\d+)_(sync|async)_s(\d+)$", m["tag"])
    if not mt:
        continue
    lag = (m.get("async") or {}).get("lag_ms_median")
    runs[(m["sequence"], mt[2])].append({
        "frag": int(mt[1]), "seed": int(mt[3]), "ate": m["ate_rmse_cm"], "pct": m["ate_percent_of_path"],
        "path": m["path_length_m"], "rpe1": m["rpe1_trans_cm"], "rpe1r": m["rpe1_rot_deg"],
        "rpe10r": m["rpe10_rot_deg"], "psnr": m.get("psnr_test"), "trk": m["track_ms"]["median"],
        "trk95": m["track_ms"]["p95"], "lag": lag, "nkf": m["n_keyframes"],
        "map_kf": (m.get("map_ms_per_kf") or {}).get("median"),
    })

out = {}
hdr = (f"{'последовательность':<17}{'режим':<7}{'фр.':>4}{'зап.':>5}{'ATE мед,см':>11}{'диапазон,см':>14}"
       f"{'% пути мед':>11}{'успех':>7}{'RPE1,см':>8}{'RPE1,°':>7}{'RPE10,°':>8}{'PSNR':>6}{'трек,мс':>8}{'p95':>5}{'отст,мс':>8}")
print(hdr)
for seq in SEQS:
    for mode in ("sync", "async"):
        rs = runs.get((seq, mode), [])
        if not rs:
            continue
        frags = defaultdict(list)
        for r in rs:
            frags[r["frag"]].append(r)
        fa = {k: float(np.mean([r["ate"] for r in v])) for k, v in frags.items()}
        fp = {k: float(np.mean([r["pct"] for r in v])) for k, v in frags.items()}
        ok = np.mean([r["pct"] <= a.ok_pct for r in rs])
        med = lambda key: float(np.median([r[key] for r in rs if r[key] is not None])) if any(r[key] is not None for r in rs) else None  # noqa: E731
        v = list(fa.values())
        row = {"n_frag": len(frags), "n_runs": len(rs), "ate_med_cm": float(np.median(v)),
               "ate_min_cm": min(v), "ate_max_cm": max(v), "pct_med": float(np.median(list(fp.values()))),
               "success": float(ok), "rpe1_cm": med("rpe1"), "rpe1_deg": med("rpe1r"), "rpe10_deg": med("rpe10r"),
               "psnr": med("psnr"), "track_ms": med("trk"), "track_p95": med("trk95"), "lag_ms": med("lag"),
               "map_kf_ms": med("map_kf"), "per_fragment": {k: {"ate_cm": fa[k], "pct": fp[k],
               "seeds": [r["ate"] for r in sorted(frags[k], key=lambda r: r["seed"])]} for k in sorted(fa)}}
        out[f"{seq}/{mode}"] = row
        lg = f"{row['lag_ms']:>8.0f}" if row["lag_ms"] is not None else f"{'-':>8}"
        print(f"{seq:<17}{mode:<7}{row['n_frag']:>4}{row['n_runs']:>5}{row['ate_med_cm']:>11.1f}"
              f"{row['ate_min_cm']:>7.1f}–{row['ate_max_cm']:<6.1f}{row['pct_med']:>11.2f}{row['success'] * 100:>6.0f}%"
              f"{row['rpe1_cm']:>8.2f}{row['rpe1_deg']:>7.2f}{row['rpe10_deg']:>8.2f}{row['psnr']:>6.1f}"
              f"{row['track_ms']:>8.1f}{row['track_p95']:>5.0f}{lg}")

print("\nпо фрагментам (ATE, см: среднее по зёрнам [зёрна])")
for k, row in out.items():
    print(k)
    for f, d in row["per_fragment"].items():
        print(f"   {f:>5}: {d['ate_cm']:7.1f} ({d['pct']:.2f}%)  [" + ", ".join(f"{x:.1f}" for x in d["seeds"]) + "]")
if a.json:
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

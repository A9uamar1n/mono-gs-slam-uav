# таблица по полным полётам. Одна ATE на весь полёт смешивает локальную точность
# и дрейф масштаба, поэтому ещё считаем ATE по окнам (400 кадров = 20 с, шаг 200),
# каждое окно выравниваем отдельно, и размах масштаба по окнам.
# запуски с одним тегом без _sN объединяются (среднее и разброс по зёрнам)
#   python scripts/table_full_flights.py outputs/stage5 "*full*" [--win 400] [--step 200]
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsslam.eval.trajectory import align_trajectory, ate  # noqa: E402


def load_tum(path):
    a = np.loadtxt(path)
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix()
    T[:, :3, 3] = a[:, 1:4]
    return a[:, 0], T


def window_stats(est, gt, win, step):
    ates, scales = [], []
    for s in range(0, max(1, len(est) - win + 1), step):
        e, g = est[s:s + win], gt[s:s + win]
        if len(e) < win // 2:
            break
        ates.append(ate(e, g, with_scale=True).rmse)
        scales.append(align_trajectory(e, g, with_scale=True)[1])
    ates, scales = np.array(ates), np.array(scales)
    return ates, scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("pattern", nargs="?", default="*")
    ap.add_argument("--win", type=int, default=400)
    ap.add_argument("--step", type=int, default=200)
    ap.add_argument("--json", default=None, help="записать сводку в файл")
    args = ap.parse_args()

    runs = []
    for d in sorted(Path(args.root).glob(args.pattern)):
        mf = d / "metrics.json"
        if not mf.exists() or not (d / "est_tum.txt").exists():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        _, est = load_tum(d / "est_tum.txt")
        _, gt = load_tum(d / "gt_tum.txt")
        wa, ws = window_stats(est, gt, args.win, args.step)
        r = {
            "run": d.name, "group": re.sub(r"_s\d+$", "", d.name),
            "ate_cm": m["ate_rmse_cm"], "ate_pct": m["ate_percent_of_path"],
            "rpe1_cm": m["rpe1_trans_cm"], "rpe10_cm": m["rpe10_trans_cm"], "rpe10_deg": m["rpe10_rot_deg"],
            "win_ate_med_cm": float(np.median(wa) * 100), "win_ate_max_cm": float(wa.max() * 100),
            "scale_ratio": float(np.percentile(ws, 90) / np.percentile(ws, 10)),   # P90/P10, чтобы одно окно не портило
            "path_m": m["path_length_m"], "n_kf": m["n_keyframes"], "psnr": m.get("psnr_test"),
            "track_med_ms": m["track_ms"]["median"], "track_p95_ms": m["track_ms"]["p95"],
            "map_kf_ms": m["map_ms_per_kf"]["median"] if m.get("map_ms_per_kf") else None,
            "lag_med_ms": (m.get("async") or {}).get("lag_ms_median"),
            "lag_max_ms": (m.get("async") or {}).get("lag_ms_max"),
            "n_frames": m["n_frames"], "n_eval": m["n_eval"],
        }
        runs.append(r)

    cols = [("ate_cm", "ATE,см", 8, 1), ("ate_pct", "%пути", 7, 2), ("win_ate_med_cm", "окно мед", 9, 1),
            ("win_ate_max_cm", "окно макс", 10, 1), ("scale_ratio", "масшт.разм", 11, 2),
            ("rpe1_cm", "RPE1,см", 8, 2), ("rpe10_deg", "RPE10,°", 8, 2), ("psnr", "PSNR", 7, 2),
            ("track_med_ms", "трек,мс", 8, 1), ("lag_med_ms", "отст,мс", 8, 0)]

    def fmt(v, w, p):
        return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"

    print(f"{'запуск':<40}" + "".join(f"{h:>{w}}" for _, h, w, _ in cols))
    for r in runs:
        print(f"{r['run']:<40}" + "".join(fmt(r[k], w, p) for k, _, w, p in cols))

    groups = defaultdict(list)
    for r in runs:
        groups[r["group"]].append(r)
    print("\nсреднее ± разброс (стд) по зёрнам")
    print(f"{'группа':<34}{'n':>3}" + "".join(f"{h:>{w + 6}}" for _, h, w, _ in cols[:7]))
    summary = {}
    for g, rs in groups.items():
        line = f"{g:<34}{len(rs):>3}"
        summary[g] = {"n": len(rs)}
        for k, _, w, p in cols[:7]:
            v = np.array([r[k] for r in rs if r[k] is not None], float)
            if len(v):
                line += f"{v.mean():>{w}.{p}f}±{v.std():<5.{p}f}"
                summary[g][k] = [float(v.mean()), float(v.std()), [float(x) for x in v]]
            else:
                line += f"{'-':>{w + 6}}"
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps({"runs": runs, "groups": summary}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()

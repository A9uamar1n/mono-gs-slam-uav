# бюджет времени по модулям из timing_ms всех запусков серии.
# по модулю: медиана по запускам и сколько раз вызывается на кадр.
# время на кадр t_frame = t_track + (N_kf / N) * t_kf,
# RTF = T_wall / (N / 20 Гц), если меньше 1 - успеваем за камерой
#   python scripts/table_timing.py [outputs/stage5] [--mode sync] [--json out.json]
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="outputs/stage5")
ap.add_argument("--mode", default="sync", choices=["sync", "async"])
ap.add_argument("--fps", type=float, default=20.0)
ap.add_argument("--json", default=None)
a = ap.parse_args()

mods = defaultdict(lambda: {"med": [], "p95": [], "per_frame": []})
rtf, vram, kf_rate, t_frame = [], [], [], []
n_runs = 0
for f in sorted(Path(a.root).glob("*/metrics.json")):
    m = json.loads(f.read_text(encoding="utf-8"))
    if not re.match(rf"f\d+_{a.mode}_s\d+$", m["tag"]):
        continue
    n_runs += 1
    N = m["n_frames"]
    for k, v in (m.get("timing_ms") or {}).items():
        if v["n"] and np.isfinite(v["median"]):
            mods[k]["med"].append(v["median"])
            mods[k]["p95"].append(v["p95"])
            mods[k]["per_frame"].append(v["n"] / N)
    rtf.append(m["wall_s"] / (N / a.fps))
    if m.get("vram_peak_mb"):
        vram.append(m["vram_peak_mb"])
    kr = m["n_keyframes"] / N
    kf_rate.append(kr)
    tkf = (m.get("timing_ms") or {}).get("маппинг (ключевой кадр)", {}).get("median")
    if tkf and np.isfinite(tkf):
        t_frame.append(m["track_ms"]["median"] + kr * tkf)

out = {"n_runs": n_runs, "modules": {}}
print(f"режим {a.mode}, запусков {n_runs}")
print(f"{'модуль':<30}{'мед, мс':>9}{'p95, мс':>9}{'вызовов/кадр':>14}{'мс/кадр':>9}")
for k, v in sorted(mods.items(), key=lambda kv: -np.median(kv[1]["med"]) * np.mean(kv[1]["per_frame"])):
    med, p95, pf = float(np.median(v["med"])), float(np.median(v["p95"])), float(np.mean(v["per_frame"]))
    out["modules"][k] = {"median_ms": med, "p95_ms": p95, "calls_per_frame": pf, "ms_per_frame": med * pf}
    print(f"{k:<30}{med:>9.1f}{p95:>9.1f}{pf:>14.3f}{med * pf:>9.1f}")
out.update({"kf_per_frame": float(np.median(kf_rate)), "t_frame_ms": float(np.median(t_frame)) if t_frame else None,
            "rtf_median": float(np.median(rtf)), "rtf_p90": float(np.percentile(rtf, 90)),
            "vram_peak_mb_max": max(vram) if vram else None})
print(f"\nключевых кадров на кадр: {out['kf_per_frame']:.3f}")
if t_frame:
    print(f"t_frame = t_track + (N_kf/N)*t_kf: медиана {out['t_frame_ms']:.1f} мс (бюджет камеры {1000 / a.fps:.0f} мс)")
print(f"RTF = T_wall/(N/f): медиана {out['rtf_median']:.2f}, P90 {out['rtf_p90']:.2f}")
if vram:
    print(f"пиковая видеопамять: {out['vram_peak_mb_max']:.0f} МБ")
if a.json:
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

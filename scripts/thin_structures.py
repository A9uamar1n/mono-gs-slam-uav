# насколько карта передаёт тонкие структуры (стойки, кабели, рейки).
# карта и позы из map.pt (run_slam.py --save-map), берём каждый 5-й неключевой кадр.
# маска M по кадру I: top-hat T(I) = max(I - open(I), close(I) - I) с ядром k x k,
# порог tau, и оставляем только вытянутые области (длина >= l_min, ширина <= w_max).
# метрики (по кадрам, потом среднее):
#   PSNR в маске и вне её;
#   C = sum_M T(R) / sum_M T(I) - сколько контраста осталось в рендере R (1 - всё, 0 - размыто);
#   Rec_M - доля границ Кэнни из маски, которые нашлись в рендере (допуск 1 пикс.);
#   если есть скан (V1) - AbsRel глубины в маске и вне её, для карты и для Depth Anything
#   (сеть выравниваем прямо по скану, т.е. ей даём фору)
#   python scripts/thin_structures.py outputs/thin/<запуск> data/euroc/<посл.> [--fig out.png]
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsslam.data.euroc import EurocSequence  # noqa: E402
from gsslam.data.gt_depth import ScanDepth, load_scan  # noqa: E402
from gsslam.map.gaussians import GaussianMap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("root")
ap.add_argument("--k", type=int, default=7, help="ядро top-hat, пиксели (376x240)")
ap.add_argument("--tau", type=float, default=0.08)
ap.add_argument("--l-min", type=float, default=12.0)
ap.add_argument("--w-max", type=float, default=3.5)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--fig", default=None)
a = ap.parse_args()

run = Path(a.run)
mp = torch.load(run / "map.pt", weights_only=False)
H, W = mp["height"], mp["width"]
seq = EurocSequence(a.root, scale=mp["scale"], start=mp["start"], end=mp["start"] + int(max(mp["ids"])) + 1)
gmap = GaussianMap.load(mp["params"])
K = torch.as_tensor(mp["K"], dtype=torch.float32, device="cuda")
ids = list(mp["ids"])
kf = set(mp["kf_ids"])
test = [j for j, i in enumerate(ids) if i not in kf][::a.every]
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.k, a.k))
scan_pts = load_scan(a.root)
scan = ScanDepth(scan_pts) if scan_pts is not None else None
mono = None
if scan is not None:
    from gsslam.depth.mono import MonoDepth
    mono = MonoDepth()


def tophat(g):
    return np.maximum(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, ker), cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, ker))


def thin_mask(g):
    t = tophat(g)
    m0 = (t > a.tau).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m0, connectivity=8)
    m = np.zeros_like(m0, dtype=bool)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] < 8:
            continue
        ys, xs = np.nonzero(lab == c)
        (_, _), (w_, h_), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        L = max(w_, h_, 1.0)
        if L >= a.l_min and st[c, cv2.CC_STAT_AREA] / L <= a.w_max:
            m |= lab == c
    border = np.zeros_like(m)
    border[8:-8, 8:-8] = True
    return m & border, t


def psnr_m(x, y, m):
    if m.sum() < 20:
        return np.nan
    return float(-10 * np.log10(max(((x - y)[m] ** 2).mean(), 1e-12)))


def edges(g):
    return cv2.Canny((np.clip(g, 0, 1) * 255).astype(np.uint8), 30, 90) > 0


rows, best = [], None
with torch.no_grad():
    for j in test:
        i = ids[j]
        T = torch.as_tensor(mp["poses"][j], dtype=torch.float32, device="cuda")
        rgb, d, alpha, _ = gmap.render(K, T, W, H)
        ex = mp["exposure"].get(i, (0.0, 0.0))
        R = (np.exp(ex[0]) * rgb[..., 0] + ex[1]).clamp(0, 1).cpu().numpy().astype(np.float32)
        img = seq.read_image(i)
        I = img[..., 0].astype(np.float32)
        M, tI = thin_mask(I)
        cov = alpha.cpu().numpy() > 0.95
        M &= cov
        rest = cov & ~M
        if M.sum() < 30:
            continue
        tR = tophat(R)
        eI, eR = edges(I), cv2.dilate(edges(R).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        r = {"frame": int(i), "mask_frac": float(M.mean()),
             "psnr_M": psnr_m(R, I, M), "psnr_rest": psnr_m(R, I, rest),
             "C": float(tR[M].sum() / max(tI[M].sum(), 1e-9)),
             "C_rest": float(tR[rest & (tI > a.tau)].sum() / max(tI[rest & (tI > a.tau)].sum(), 1e-9)),
             "rec_M": float((eI & M & eR).sum() / max((eI & M).sum(), 1)),
             "rec_rest": float((eI & rest & eR).sum() / max((eI & rest).sum(), 1))}
        if scan is not None:
            Tg = torch.as_tensor(mp["gt_poses"][j], dtype=torch.float32, device="cuda")
            ds, ok = scan.render(K, Tg, W, H)
            ds, ok = ds.cpu().numpy(), ok.cpu().numpy()
            dm = d.cpu().numpy()
            v = ok & cov & (dm > 1e-3)
            s = np.median(ds[v] / dm[v])
            dm_s = dm * s
            p = mono(img).float().cpu().numpy()   # обратная глубина от сети
            A = np.stack([p[v], np.ones(v.sum())], 1)
            st_, *_ = np.linalg.lstsq(A, 1.0 / ds[v], rcond=None)
            dn = 1.0 / np.maximum(st_[0] * p + st_[1], 1e-3)
            for nm, dd in (("map", dm_s), ("dav2", dn)):
                for reg, mm in (("M", M & v), ("rest", rest & v)):
                    e = np.abs(dd[mm] - ds[mm]) / ds[mm]
                    r[f"absrel_{nm}_{reg}"] = float(e.mean()) if mm.sum() > 20 else np.nan
                    r[f"d25_{nm}_{reg}"] = float((e < 0.25).mean()) if mm.sum() > 20 else np.nan
            r["_depth"] = (dm_s, dn, ds, ok)
        rows.append(r)
        if best is None or r["mask_frac"] > best[0]["mask_frac"]:
            best = (r, I, R, M)

keys = [k for k in rows[0] if not k.startswith("_") and k != "frame"]
summ = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
summ["n_frames"] = len(rows)
print(f"{run.name}: кадров {len(rows)}, доля маски {summ['mask_frac'] * 100:.1f}%")
print(f"  PSNR в маске {summ['psnr_M']:.2f} дБ, вне {summ['psnr_rest']:.2f} дБ")
print(f"  контраст C в маске {summ['C']:.2f}, вне маски (текстура) {summ['C_rest']:.2f}")
print(f"  полнота границ в маске {summ['rec_M']:.2f}, вне {summ['rec_rest']:.2f}")
if "absrel_map_M" in summ:
    print(f"  AbsRel карта: в маске {summ['absrel_map_M']:.3f}, вне {summ['absrel_map_rest']:.3f}; "
          f"DA2: в маске {summ['absrel_dav2_M']:.3f}, вне {summ['absrel_dav2_rest']:.3f}")
    print(f"  δ25 карта: {summ['d25_map_M']:.3f} / {summ['d25_map_rest']:.3f}; DA2: {summ['d25_dav2_M']:.3f} / {summ['d25_dav2_rest']:.3f}")
(run / "thin.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")

if a.fig and best is not None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, I, R, M = best
    ov = np.repeat(I[..., None], 3, 2).copy()
    ov[M] = [1.0, 0.2, 0.2]
    panels = [("кадр", I, "gray"), ("маска тонких структур", ov, None), ("рендер карты", R, "gray"),
              ("|рендер − кадр|", np.abs(R - I), "magma")]
    if "_depth" in r:
        dm_s, dn, ds, ok = r["_depth"]
        vmax = np.percentile(ds[ok], 98)
        panels += [("глубина карты", dm_s, "turbo"), ("Depth Anything V2", dn, "turbo"),
                   ("лазерный скан", np.where(ok, ds, np.nan), "turbo")]
    n = len(panels)
    cols = 4 if n > 4 else 2
    fig, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 2.8 * ((n + cols - 1) // cols)), squeeze=False)
    for k, (t, x, cm) in enumerate(panels):
        A_ = ax[k // cols, k % cols]
        kw = {"vmin": 0, "vmax": vmax} if cm == "turbo" else ({"vmin": 0, "vmax": 1} if cm == "gray" else {})
        A_.imshow(x, cmap=cm, **kw)
        A_.set_title(t, fontsize=9)
    for A_ in ax.flat:
        A_.axis("off")
    fig.suptitle(f"{run.name}, кадр {r['frame']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=130)

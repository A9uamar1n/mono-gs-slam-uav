# маппинг при известных (эталонных) позах: строим карту по фрагменту с глубиной от сети
# и проверяем её на отложенных кадрах между ключевыми. Так проверялся маппинг отдельно от трекинга
#   python scripts/mapping_gt_poses.py data/euroc/MH_05_difficult --start 600 --n-frames 400 --tag full
#   python scripts/mapping_gt_poses.py data/euroc/MH_05_difficult --start 600 --n-frames 400 --lambda-depth 0 --lambda-smooth 0 --tag no_depth
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsslam.data.euroc import EurocSequence  # noqa: E402
from gsslam.data.gt_depth import ScanDepth, load_scan  # noqa: E402
from gsslam.depth.align import apply_alignment, confidence_map, fit_scale_shift  # noqa: E402
from gsslam.depth.mono import MonoDepth  # noqa: E402
from gsslam.geometry.sparse import SparseTriangulator  # noqa: E402
from gsslam.map.deblur import BlurMLP  # noqa: E402
from gsslam.map.gaussians import GaussianMap  # noqa: E402
from gsslam.map.losses import (_gauss_window, color_loss, depth_loss, psnr,  # noqa: E402
                               smooth_loss, ssim)
from gsslam.utils.timing import Timers  # noqa: E402


def to_t(x, dev):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


def colorize(d, lo, hi):
    x = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    return cv2.applyColorMap((255 - x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=600)
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--kf-step", type=int, default=10)
    ap.add_argument("--iters-per-kf", type=int, default=60)
    ap.add_argument("--final-iters", type=int, default=600)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--lambda-ssim", type=float, default=0.2)
    ap.add_argument("--lambda-depth", type=float, default=0.5)
    ap.add_argument("--lambda-smooth", type=float, default=0.5)
    ap.add_argument("--no-conf", action="store_true", help="не использовать карту уверенности")
    ap.add_argument("--init", choices=["dense", "sparse"], default="dense",
                    help="dense: по выровненной нейросетевой глубине; sparse: только по триангулированным точкам")
    ap.add_argument("--sparse-px", type=float, default=6.0, help="размер гауссианы в пикселях при --init sparse")
    ap.add_argument("--depth-mode", choices=["anchor", "realign"], default="anchor",
                    help="anchor: цель D* фиксирована выравниванием по опорным точкам; "
                         "realign: на каждом шаге предсказание сети заново выравнивается по текущей глубине карты")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="outputs/mapping")
    ap.add_argument("--deblur", action="store_true", help="Deblurring MLP при обучении (этап 2)")
    ap.add_argument("--deblur-lr", type=float, default=1e-3)
    ap.add_argument("--deblur-dq", type=float, default=0.2, help="предел поправки кватерниона (0 -- только масштаб)")
    ap.add_argument("--deblur-reg", type=float, default=0.0, help="штраф за среднюю поправку log-масштаба")
    ap.add_argument("--save-map", action="store_true", help="сохранить карту в map.pt (для трекинга)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    timers = Timers()

    seq = EurocSequence(args.root, scale=args.scale, start=args.start, end=args.start + args.n_frames)
    W, H = seq.width, seq.height
    K_np = seq.K
    K = to_t(K_np, dev)
    has_gt = [i for i in range(len(seq)) if seq.gt_T_wc[i] is not None]
    kf_ids = [i for i in has_gt if i % args.kf_step == 0]
    test_ids = [i + args.kf_step // 2 for i in kf_ids[:-1] if seq.gt_T_wc[i + args.kf_step // 2] is not None]
    print(f"{Path(args.root).name}: кадры {args.start}..{args.start + len(seq) - 1}, {W}x{H}, "
          f"ключевых {len(kf_ids)}, отложенных {len(test_ids)}")

    # ключевые кадры: глубина сети, опорные точки, выравнивание
    mono = MonoDepth(device="cuda")
    tri = SparseTriangulator()
    frames = {i: seq[i] for i in sorted(set(kf_ids + test_ids))}
    feats = {i: tri.features(frames[i].image) for i in frames}

    kfs = []
    align_stats = []
    for n, i in enumerate(kf_ids):
        fr = frames[i]
        with timers("глубина (сеть)"):
            pred = mono(fr.image)
        pred_np = pred.cpu().numpy()
        with timers("опорные точки"):
            neigh = [(feats[kf_ids[j]], frames[kf_ids[j]].T_wc)
                     for j in (n - 2, n - 1, n + 1, n + 2) if 0 <= j < len(kf_ids)]
            sd = tri.sparse_depth(feats[i], K_np, fr.T_wc, neigh)
        with timers("выравнивание + уверенность"):
            a = fit_scale_shift(pred_np, sd.uv, sd.depth)
            if a.ok:
                d_star, valid = apply_alignment(pred_np, a)
                conf = confidence_map(fr.image[:, :, 0], d_star, valid)
            else:
                d_star = np.zeros((H, W), np.float32)
                valid = np.zeros((H, W), bool)
                conf = np.zeros((H, W), np.float32)
        g8 = (fr.image[:, :, 0] * 255).astype(np.uint8)
        edges = cv2.dilate(cv2.Canny(g8, 50, 150), np.ones((3, 3), np.uint8))
        align_stats.append((a.ok, a.n_points, a.rel_err))
        kfs.append({
            "pred": pred.float(), "sparse_uv": to_t(sd.uv, dev), "sparse_d": to_t(sd.depth, dev),
            "id": i, "image": to_t(fr.image, dev), "T_wc": to_t(fr.T_wc, dev),
            "depth": to_t(d_star, dev), "valid": torch.as_tensor(valid, device=dev),
            "conf": torch.ones(H, W, device=dev) if args.no_conf else to_t(conf, dev),
            "non_edge": to_t((edges == 0).astype(np.float32), dev), "ok": a.ok,
        })
    ok = np.array([s[0] for s in align_stats])
    rel = np.array([s[2] for s in align_stats])[ok]
    print(f"выравнивание глубины: успешно {ok.sum()}/{len(ok)} ключевых кадров, "
          f"медианная отн. ошибка на опорных точках {np.median(rel) * 100:.1f}% "
          f"(худший кадр {rel.max() * 100:.1f}%)")
    del mono
    torch.cuda.empty_cache()

    # строим карту
    gmap = GaussianMap()
    win = _gauss_window(device=dev)

    def realign(kf, d, alpha):
        # выравниваем выход сети по текущей глубине карты (в обратной глубине)
        with torch.no_grad():
            m = (alpha > 0.9) & (d > 0.05)
            if m.sum() < 500:
                return kf["depth"], kf["valid"]
            x, y = kf["pred"], 1.0 / d.clamp(min=0.05)
            w = kf["conf"] * m
            for _ in range(2):   # второй раз уже с весами Хьюбера
                sw = w.sum()
                mx, my = (w * x).sum() / sw, (w * y).sum() / sw
                var = (w * (x - mx) ** 2).sum()
                s_ = (w * (x - mx) * (y - my)).sum() / var.clamp(min=1e-12)
                t_ = my - s_ * mx
                r = (y - (s_ * x + t_)).abs()
                k = 1.345 * 1.4826 * r[m].median().clamp(min=1e-9)
                w = kf["conf"] * m * torch.where(r <= k, torch.ones_like(r), k / r.clamp(min=1e-12))
            if not bool(s_ > 0):
                return kf["depth"], kf["valid"]
            v = s_ * x + t_
            valid = v > 1.0 / 40.0
            return torch.where(valid, 1.0 / v.clamp(min=1e-6), torch.zeros_like(v)), valid

    for n_, kf in enumerate(kfs):
        kf["n"] = n_
    blur = None
    if args.deblur:
        blur = BlurMLP(len(kfs), max_dq=args.deblur_dq).to(dev)
        blur_opt = torch.optim.Adam(blur.parameters(), lr=args.deblur_lr)

    def step(kf):
        with timers("рендер (вперёд)"):
            if blur is not None:
                if blur.center is None:
                    blur.set_bounds(gmap.params["means"])
                q_b, ls_b, dls = blur(gmap.params["means"], gmap.params["quats"], gmap.params["log_scales"], kf["n"])
                rgb, d, alpha, _ = gmap.render(K, kf["T_wc"], W, H, quats=q_b, log_scales=ls_b)
            else:
                dls = None
                rgb, d, alpha, _ = gmap.render(K, kf["T_wc"], W, H)
        with timers("потери + назад"):
            loss = color_loss(rgb, kf["image"], args.lambda_ssim, win)
            if args.lambda_depth > 0 and kf["ok"]:
                if args.depth_mode == "realign":
                    target, valid = realign(kf, d.detach(), alpha.detach())
                else:
                    target, valid = kf["depth"], kf["valid"]
                loss = loss + args.lambda_depth * depth_loss(
                    d, target, kf["conf"], valid & (alpha.detach() > 0.3))
            if args.lambda_smooth > 0:
                loss = loss + args.lambda_smooth * smooth_loss(d, kf["non_edge"])
            if dls is not None and args.deblur_reg > 0:
                loss = loss + args.deblur_reg * dls.mean()
            gmap.opt.zero_grad(set_to_none=True)
            if blur is not None:
                blur_opt.zero_grad(set_to_none=True)
            loss.backward()
            gmap.opt.step()
            if blur is not None:
                blur_opt.step()
        return float(loss)

    for n, kf in enumerate(kfs):
        if kf["ok"]:
            with timers("добавление гауссиан"):
                if len(gmap) == 0:
                    mask = kf["valid"]
                else:
                    with torch.no_grad():
                        _, d, alpha, _ = gmap.render(K, kf["T_wc"], W, H)
                    mask = kf["valid"] & ((alpha < 0.5) | (d > 1.2 * kf["depth"]))
                if args.init == "dense":
                    gmap.add_from_depth(kf["image"], kf["depth"], mask, K, kf["T_wc"], args.stride)
                else:
                    # только по триангулированным точкам (как инициализация по SfM)
                    uv, dd = kf["sparse_uv"], kf["sparse_d"]
                    ui = uv[:, 0].round().long().clamp(0, W - 1)
                    vi = uv[:, 1].round().long().clamp(0, H - 1)
                    keep = torch.ones_like(dd, dtype=torch.bool) if len(gmap) == 0 else (alpha[vi, ui] < 0.5)
                    uu, vv, dk = uv[keep, 0], uv[keep, 1], dd[keep]
                    pc = torch.stack([(uu - K[0, 2]) / K[0, 0] * dk, (vv - K[1, 2]) / K[1, 1] * dk, dk], 1)
                    pw = pc @ kf["T_wc"][:3, :3].T + kf["T_wc"][:3, 3]
                    gmap.add(pw, dk * args.sparse_px / K[0, 0], kf["image"][vi[keep], ui[keep]])
        if len(gmap) == 0:
            continue
        lo = max(0, n - args.window + 1)
        for _ in range(args.iters_per_kf):
            j = random.randint(lo, n) if (random.random() < 0.7 or lo == 0) else random.randint(0, lo - 1)
            step(kfs[j])
        if (n + 1) % 5 == 0:
            gmap.prune()
    for it in range(args.final_iters):
        step(kfs[random.randrange(len(kfs))])
        if (it + 1) % 200 == 0:
            gmap.prune()
    gmap.prune()
    print(f"гауссиан в карте: {len(gmap)}")

    # оценка на отложенных кадрах
    out =Path(args.out) / f"{Path(args.root).name}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    if args.save_map:
        torch.save({"params": {k: p.detach().cpu() for k, p in gmap.params.items()},
                    "K": K_np, "width": W, "height": H, "scale": args.scale,
                    "start": args.start, "n_frames": args.n_frames, "kf_ids": kf_ids},
                   out / "map.pt")
    res = {"psnr": [], "ssim": [], "abs_rel": [], "med_rel": [], "delta1": [], "n_pts": [],
           "scan_rel": [], "scan_med": [], "scan_rel_flat": [], "scan_d1": [],
           "gt_sharp": [], "grad_ratio": []}

    def grad_energy(x):
        g = x[..., 0] if x.dim() == 3 else x
        return float(((g[:, 1:] - g[:, :-1]).abs().mean() + (g[1:] - g[:-1]).abs().mean()) / 2)
    scan_pts = load_scan(args.root)
    scan = ScanDepth(scan_pts) if scan_pts is not None else None
    panels = []
    with torch.no_grad():
        for n, i in enumerate(test_ids):
            fr = frames[i]
            gt = to_t(fr.image, dev)
            with timers("рендер (оценка)"):
                rgb, d, alpha, _ = gmap.render(K, to_t(fr.T_wc, dev), W, H)
            rgb = rgb.clamp(0, 1)
            res["psnr"].append(psnr(rgb, gt))
            res["ssim"].append(float(ssim(rgb, gt, win)))
            # резкость кадра (дисперсия лапласиана) и во сколько рендер резче/мягче кадра
            g8 = (fr.image[:, :, 0] * 255).astype(np.uint8)
            res["gt_sharp"].append(float(cv2.Laplacian(g8, cv2.CV_32F).var()))
            res["grad_ratio"].append(grad_energy(rgb) / max(grad_energy(gt), 1e-9))
            # глубину сравниваем с точками, триангулированными для этого отложенного кадра
            neigh = [(feats[j], frames[j].T_wc) for j in (i - args.kf_step // 2, i + args.kf_step // 2)
                     if j in frames]
            sd = tri.sparse_depth(feats[i], K_np, fr.T_wc, neigh)
            if len(sd.depth) >= 20:
                dn = d.cpu().numpy()
                u = np.clip(np.round(sd.uv[:, 0]).astype(int), 0, W - 1)
                v = np.clip(np.round(sd.uv[:, 1]).astype(int), 0, H - 1)
                dr = dn[v, u]
                good = dr > 0.05
                if good.sum() >= 20:
                    ratio = np.maximum(dr[good] / sd.depth[good], sd.depth[good] / dr[good])
                    res["abs_rel"].append(float(np.mean(np.abs(dr[good] - sd.depth[good]) / sd.depth[good])))
                    res["med_rel"].append(float(np.median(np.abs(dr[good] - sd.depth[good]) / sd.depth[good])))
                    res["delta1"].append(float(np.mean(ratio < 1.25)))
                    res["n_pts"].append(int(good.sum()))
            gt_d = None
            if scan is not None:
                # глубина по скану, отдельно считаем на областях без текстуры
                gt_d, gt_ok = scan.render(K, to_t(fr.T_wc, dev), W, H)
                m = gt_ok & (d > 0.05) & (alpha > 0.5)
                if m.sum() > 1000:
                    rel_map = (d - gt_d).abs() / gt_d.clamp(min=0.05)
                    g8 = (fr.image[:, :, 0] * 255).astype(np.uint8)
                    grad = cv2.GaussianBlur(np.abs(cv2.Laplacian(g8, cv2.CV_32F)), (0, 0), 3.0)
                    flat = torch.as_tensor(grad < np.percentile(grad, 30), device=dev) & m
                    ratio = torch.maximum(d / gt_d.clamp(min=0.05), gt_d / d.clamp(min=0.05))
                    res["scan_rel"].append(float(rel_map[m].mean()))
                    res["scan_med"].append(float(rel_map[m].median()))
                    res["scan_d1"].append(float((ratio[m] < 1.25).float().mean()))
                    if flat.sum() > 200:
                        res["scan_rel_flat"].append(float(rel_map[flat].mean()))
            if n in (len(test_ids) // 4, len(test_ids) // 2, 3 * len(test_ids) // 4):
                dn = d.cpu().numpy()
                lo, hi = np.percentile(dn[dn > 0], [2, 98]) if (dn > 0).any() else (0, 1)
                g = cv2.cvtColor((fr.image[:, :, 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                r = (rgb.cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
                row = [g, r, colorize(dn, lo, hi)]
                if gt_d is not None:
                    row.append(colorize(gt_d.cpu().numpy(), lo, hi))
                panels.append(np.hstack(row))
    if panels:
        cv2.imwrite(str(out / "test_frames.png"), np.vstack(panels))

    # делим отложенные кадры по резкости: самая резкая треть вместо резкого эталона
    order = np.argsort(res["gt_sharp"])
    k3 = max(1, len(order) // 3)
    ps = np.array(res["psnr"])
    sharp_split = {"psnr_sharpest_third": float(ps[order[-k3:]].mean()),
                   "psnr_blurriest_third": float(ps[order[:k3]].mean()),
                   "grad_ratio": float(np.mean(res["grad_ratio"]))}
    # PSNR на ключевых кадрах: обычный рендер и (если есть деблюр) через MLP
    with torch.no_grad():
        kf_sharp, kf_blur = [], []
        for kf in kfs:
            rgb, _, _, _ = gmap.render(K, kf["T_wc"], W, H)
            kf_sharp.append(psnr(rgb.clamp(0, 1), kf["image"]))
            if blur is not None:
                q_b, ls_b, _ = blur(gmap.params["means"], gmap.params["quats"], gmap.params["log_scales"], kf["n"])
                rgb, _, _, _ = gmap.render(K, kf["T_wc"], W, H, quats=q_b, log_scales=ls_b)
                kf_blur.append(psnr(rgb.clamp(0, 1), kf["image"]))
    if blur is not None:
        with torch.no_grad():
            st = [blur(gmap.params["means"], gmap.params["quats"], gmap.params["log_scales"], kf["n"])[2] for kf in kfs[::5]]
            sharp_split["dls_mean"] = float(torch.stack([x.mean() for x in st]).mean())
            sharp_split["dls_p95"] = float(torch.cat([x.flatten() for x in st]).quantile(0.95))
        print(f"поправка log-масштаба: средняя {sharp_split['dls_mean']:.3f}, p95 {sharp_split['dls_p95']:.3f}")
    sharp_split["psnr_kf_sharp_render"] = float(np.mean(kf_sharp))
    sharp_split["psnr_kf_blur_render"] = float(np.mean(kf_blur)) if kf_blur else None
    summary = {"deblur": sharp_split,
        "sequence": Path(args.root).name, "tag": args.tag, "args": vars(args),
        "n_keyframes": len(kfs), "n_test": len(test_ids), "n_gaussians": len(gmap),
        "aligned_keyframes": int(ok.sum()),
        "psnr": float(np.mean(res["psnr"])), "ssim": float(np.mean(res["ssim"])),
        "depth_abs_rel": float(np.mean(res["abs_rel"])) if res["abs_rel"] else None,
        "depth_delta1": float(np.mean(res["delta1"])) if res["delta1"] else None,
        "depth_med_rel": float(np.mean(res["med_rel"])) if res["med_rel"] else None,
        "depth_eval_frames": len(res["abs_rel"]),
        "scan_abs_rel": float(np.mean(res["scan_rel"])) if res["scan_rel"] else None,
        "scan_med_rel": float(np.mean(res["scan_med"])) if res["scan_med"] else None,
        "scan_abs_rel_flat": float(np.mean(res["scan_rel_flat"])) if res["scan_rel_flat"] else None,
        "scan_delta1": float(np.mean(res["scan_d1"])) if res["scan_d1"] else None,
        "timing_ms": {k: timers.stats(k, skip=3) for k in timers.samples},
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nотложенные кадры ({len(test_ids)}): PSNR {summary['psnr']:.2f} дБ, SSIM {summary['ssim']:.3f}")
    if res["abs_rel"]:
        print(f"глубина по опорным точкам ({len(res['abs_rel'])} кадров, в среднем {np.mean(res['n_pts']):.0f} точек): "
              f"отн. ошибка {summary['depth_abs_rel'] * 100:.1f}% (медианная {summary['depth_med_rel'] * 100:.1f}%), доля точек с ошибкой < 25%: "
              f"{summary['depth_delta1'] * 100:.1f}%")
    if res["scan_rel"]:
        print(f"глубина по лазерному скану ({len(res['scan_rel'])} кадров): отн. ошибка {summary['scan_abs_rel'] * 100:.1f}% "
              f"(медианная {summary['scan_med_rel'] * 100:.1f}%), на бестекстурных областях "
              f"{summary['scan_abs_rel_flat'] * 100:.1f}%, доля пикселей с ошибкой < 25%: {summary['scan_delta1'] * 100:.1f}%")
    print(f"резкие трети: PSNR на самых резких {sharp_split['psnr_sharpest_third']:.2f} дБ, на самых смазанных "
          f"{sharp_split['psnr_blurriest_third']:.2f} дБ; резкость рендера / кадра {sharp_split['grad_ratio']:.3f}; "
          f"ключевые кадры: резкий рендер {sharp_split['psnr_kf_sharp_render']:.2f} дБ"
          + (f", через MLP {sharp_split['psnr_kf_blur_render']:.2f} дБ" if kf_blur else ""))
    print(timers.table(skip=3))
    print(f"пик видеопамяти: {summary['vram_peak_mb']:.0f} МБ; результаты: {out}")


if __name__ == "__main__":
    main()

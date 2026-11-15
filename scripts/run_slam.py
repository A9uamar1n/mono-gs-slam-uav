# запуск SLAM на последовательности EuRoC. Эталон нужен только для оценки (ATE, RPE после Sim3)
#   python scripts/run_slam.py data/euroc/V1_01_easy --start 800 --n-frames 400 --tag v1_800
#   python scripts/run_slam.py data/euroc/MH_05_difficult --tag mh05_full
import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsslam.data.euroc import EurocSequence  # noqa: E402
from gsslam.eval.trajectory import align_trajectory, ate, rpe, save_tum  # noqa: E402
from gsslam.map.losses import psnr  # noqa: E402
from gsslam.slam.async_system import AsyncGsSlam  # noqa: E402
from gsslam.slam.system import GsSlam, SlamConfig  # noqa: E402
from gsslam.utils.timing import Timers  # noqa: E402


def parse_levels(s):
    return tuple((float(a), int(b)) for a, b in (x.split(":") for x in s.split(",")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--levels", default=None, help="по умолчанию 0.5:10 (render) или 0.5:4,1:4 (direct)")
    ap.add_argument("--tracker", choices=["render", "direct"], default="render")
    ap.add_argument("--track-points", type=int, default=0)
    ap.add_argument("--lr-trans", type=float, default=3e-3)
    ap.add_argument("--lr-rot", type=float, default=3e-3)
    ap.add_argument("--kf-max", type=int, default=10)
    ap.add_argument("--cov-min", type=float, default=0.85)
    ap.add_argument("--iters-per-kf", type=int, default=40)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--lambda-depth", type=float, default=0.5)
    ap.add_argument("--no-refine", action="store_true", help="не уточнять позы ключевых кадров при маппинге")
    ap.add_argument("--align", choices=["map", "triangulation"], default="triangulation")
    ap.add_argument("--depth", choices=["dav2", "scan_aligned", "scan_metric"], default="dav2",
                    help="диагностика: глубина ключевых кадров из лазерного скана (только V1/V2)")
    ap.add_argument("--align-field", choices=["affine", "linear", "grid3"], default="affine")
    ap.add_argument("--align-lam", type=float, default=0.3)
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--async", dest="async_", action="store_true",
                    help="асинхронный маппинг в отдельном потоке, кадры подаются в темпе камеры")
    ap.add_argument("--publish-every", type=int, default=10)
    ap.add_argument("--idle-duty", type=float, default=1.0)
    ap.add_argument("--switch-ms", type=float, default=0.5)
    ap.add_argument("--init-px", type=float, default=30.0)
    ap.add_argument("--lost-cm", type=float, default=30.0)
    ap.add_argument("--reject", choices=["auto", "on", "off"], default="auto",
                    help="отбраковка неправдоподобных поз: auto -- только в асинхронном режиме "
                         "(в синхронном она замораживала трекинг на сложных фрагментах, этап 5)")
    ap.add_argument("--max-corr-rot", type=float, default=5.0, help="порог отбраковки поправки, градусы")
    ap.add_argument("--max-corr-trans", type=float, default=0.05)
    ap.add_argument("--reject-loss-ratio", type=float, default=1.5,
                    help="0 -- отбраковка только по величине поправки (как до этапа 5)")
    ap.add_argument("--kf-motion-trans", type=float, default=0.02, help="0 и --kf-motion-rot 0 -- без условия движения")
    ap.add_argument("--kf-motion-rot", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-map", action="store_true", help="сохранить карту, позы и экспозицию в map.pt")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="outputs/slam")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    end = None if args.n_frames is None else args.start + args.n_frames
    seq = EurocSequence(args.root, scale=args.scale, start=args.start, end=end)
    if args.levels is None:
        args.levels = "0.5:4,1:4" if args.tracker == "direct" else "0.5:10"
    reject_on = args.reject == "on" or (args.reject == "auto" and args.async_)
    # без отбраковки просто ставим огромные пороги
    cfg = SlamConfig(tracker=args.tracker, track_points=args.track_points,
                     track_levels=parse_levels(args.levels), lr_trans=args.lr_trans, lr_rot=args.lr_rot,
                     kf_max=args.kf_max, cov_min=args.cov_min,
                     iters_per_kf=args.iters_per_kf, window=args.window, lambda_depth=args.lambda_depth,
                     refine_poses=not args.no_refine, init_min_px=args.init_px, align=args.align,
                     depth_mode=args.depth, align_field=args.align_field, align_lam=args.align_lam,
                     depth_model=args.depth_model, reject_loss_ratio=args.reject_loss_ratio,
                     max_corr_rot_deg=args.max_corr_rot if reject_on else 1e9,
                     max_corr_trans=args.max_corr_trans if reject_on else 1e9,
                     kf_motion_trans=args.kf_motion_trans, kf_motion_rot_deg=args.kf_motion_rot)
    timers = Timers()
    oracle = None
    if args.depth != "dav2":
        from gsslam.data.gt_depth import ScanDepth, load_scan
        scan = ScanDepth(load_scan(args.root))
        K_t = torch.as_tensor(seq.K, dtype=torch.float32, device="cuda")

        def oracle(i):
            d, ok = scan.render(K_t, torch.as_tensor(seq.gt_T_wc[i], dtype=torch.float32, device="cuda"),
                                seq.width, seq.height)
            return d.cpu().numpy(), ok.cpu().numpy()
    if args.async_:
        slam = AsyncGsSlam(seq.K, seq.width, seq.height, cfg, timers, depth_oracle=oracle,
                           publish_every=args.publish_every, idle_duty=args.idle_duty,
                           switch_interval_ms=args.switch_ms)
    else:
        slam = GsSlam(seq.K, seq.width, seq.height, cfg, timers, depth_oracle=oracle)
    name = Path(args.root).name
    print(f"{name}: кадры {args.start}..{args.start + len(seq) - 1}, {seq.width}x{seq.height}")

    t0 = time.perf_counter()
    lag = []   # отставание: конец обработки кадра минус время, когда он снят
    t_cam0 = None
    for i in range(len(seq)):
        if args.async_ and slam.initialized:
            # подаём кадры в темпе камеры, не раньше времени съёмки
            if t_cam0 is None:
                t_cam0 = time.perf_counter() - (seq.timestamps[i] - seq.timestamps[0])
            due = t_cam0 + (seq.timestamps[i] - seq.timestamps[0])
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        img = seq.read_image(i)
        slam.process(i, img)
        if t_cam0 is not None:
            lag.append((time.perf_counter() - (t_cam0 + (seq.timestamps[i] - seq.timestamps[0]))) * 1e3)
        if slam.initialized and i % 200 == 0:
            print(f"  кадр {i}: ключевых {len(slam.kf_ids)}, гауссиан {len(slam.mapper.gmap)}", flush=True)
    wall = time.perf_counter() - t0
    if args.async_:
        slam.finish()
    if not slam.initialized:
        print("инициализация не удалась")
        return

    ids = np.array([i for i in sorted(slam.poses) if seq.gt_T_wc[i] is not None])
    est = np.stack([slam.poses[i] for i in ids])
    gt = np.stack([seq.gt_T_wc[i] for i in ids])
    a = ate(est, gt, with_scale=True)
    r1 = rpe(est, gt, delta=1, with_scale=True)
    r10 = rpe(est, gt, delta=10, with_scale=True)
    al, s_, _, _ = align_trajectory(est, gt, with_scale=True)
    err = np.linalg.norm(al[:, :3, 3] - gt[:, :3, 3], axis=1)
    lost = np.nonzero(err * 100 > args.lost_cm)[0]
    path_len = float(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum())

    # PSNR рендера по оценённым позам на каждом 10-м неключевом кадре
    test = [i for i in ids if i not in set(slam.kf_ids)][::10]
    ps = []
    K_t = torch.as_tensor(seq.K, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for i in test:
            rgb, _, _, _ = slam.mapper.gmap.render(K_t, torch.as_tensor(slam.poses[i], dtype=torch.float32, device="cuda"),
                                                  seq.width, seq.height)
            gti = torch.as_tensor(seq.read_image(i), device="cuda")
            ex = slam.track_log[[l["frame"] for l in slam.track_log].index(i)]["exposure"] \
                if i in {l["frame"] for l in slam.track_log} else (0.0, 0.0)
            ps.append(psnr((np.exp(ex[0]) * rgb + ex[1]).clamp(0, 1), gti))

    n_kf = len(slam.kf_ids)
    tt = timers.stats("трекинг (кадр)", skip=5)
    tm = timers.stats("маппинг (ключевой кадр)", skip=1)
    kf_interval_ms = 1000.0 * (len(slam.poses) / max(n_kf, 1)) / seq.rate_hz
    align_src = [x["source"] for x in slam.mapper.align_log]
    summary = {
        "sequence": name, "tag": args.tag, "args": vars(args), "init": slam.init_info,
        "n_frames": len(seq), "n_tracked": len(slam.poses), "n_eval": len(ids), "n_keyframes": n_kf,
        "n_gaussians": len(slam.mapper.gmap), "path_length_m": path_len,
        "ate_rmse_cm": a.rmse * 100, "ate_median_cm": a.median * 100 if hasattr(a, "median") else None,
        "ate_percent_of_path": a.rmse / path_len * 100, "sim3_scale": float(s_),
        "rpe1_trans_cm": r1.trans_rmse * 100, "rpe1_rot_deg": r1.rot_rmse_deg,
        "rpe10_trans_cm": r10.trans_rmse * 100, "rpe10_rot_deg": r10.rot_rmse_deg,
        "frames_over_lost_threshold": int(len(lost)), "first_over_threshold": int(ids[lost[0]]) if len(lost) else None,
        "psnr_test": float(np.mean(ps)) if ps else None,
        "align_sources": {k: align_src.count(k) for k in set(align_src)},
        "track_ms": tt, "map_ms_per_kf": tm, "kf_interval_ms": kf_interval_ms,
        "track_rejected": getattr(slam, "n_rejected", 0),
        "wall_s": wall, "vram_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "async": None if not args.async_ else {
            "lag_ms_median": float(np.median(lag)), "lag_ms_p95": float(np.percentile(lag, 95)),
            "lag_ms_max": float(np.max(lag)), "lag_ms_last": float(lag[-1]),
            "kf_requested": slam.stats["kf_requested"], "kf_rejected": slam.stats["kf_rejected"],
            "idle_iters": slam.stats["idle_iters"], "publishes": slam.stats["publishes"],
            "insert_ms_median": float(np.median(slam.stats["insert_ms"])) if slam.stats["insert_ms"] else None,
            "snap_age_frames_median": float(np.median(slam.stats["snap_age_frames"])) if slam.stats["snap_age_frames"] else None,
            "map_iters_total": slam.stats["idle_iters"] + len(slam.mapper.kfs) * args.iters_per_kf,
        },
        "timing_ms": {k: timers.stats(k, skip=1) for k in timers.samples},
    }
    out = Path(args.out) / f"{name}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_tum(str(out / "est_tum.txt"), seq.timestamps[ids], est)
    save_tum(str(out / "gt_tum.txt"), seq.timestamps[ids], gt)
    (out / "track_log.json").write_text(json.dumps(slam.track_log), encoding="utf-8")
    if args.save_map:
        torch.save({"params": {k: v.detach().cpu() for k, v in slam.mapper.gmap.params.items()},
                    "K": seq.K, "width": seq.width, "height": seq.height, "scale": args.scale,
                    "start": args.start, "ids": ids, "poses": est, "gt_poses": gt,
                    "kf_ids": list(slam.kf_ids), "sim3": (float(s_), al),
                    "exposure": {l["frame"]: l["exposure"] for l in slam.track_log}}, out / "map.pt")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(gt[:, 0, 3], gt[:, 1, 3], "k-", lw=1, label="эталон")
    ax[0].plot(al[:, 0, 3], al[:, 1, 3], "C0-", lw=1, label="оценка (Sim3)")
    kfm = np.isin(ids, slam.kf_ids)
    ax[0].plot(al[kfm, 0, 3], al[kfm, 1, 3], "C1.", ms=3, label="ключевые кадры")
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_xlabel("x, м"); ax[0].set_ylabel("y, м")
    ax[0].set_title(f"{name}: ATE {a.rmse * 100:.1f} см")
    ax[1].plot(ids, err * 100, "C0-", lw=1)
    ax[1].set_xlabel("кадр"); ax[1].set_ylabel("ошибка позиции, см"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "trajectory.png", dpi=120); plt.close(fig)

    print(f"инициализация: кадры {slam.init_info['first']} и {slam.init_info['second']}, "
          f"смещение {slam.init_info['median_px']:.1f} px, точек {slam.init_info['points']}")
    print(f"оценено {len(ids)} кадров, путь {path_len:.1f} м, ключевых {n_kf}, гауссиан {len(slam.mapper.gmap)}")
    print(f"ATE {summary['ate_rmse_cm']:.2f} см ({summary['ate_percent_of_path']:.2f}% пути), "
          f"RPE(1) {summary['rpe1_trans_cm']:.2f} см / {summary['rpe1_rot_deg']:.3f}°, "
          f"RPE(10) {summary['rpe10_trans_cm']:.2f} см / {summary['rpe10_rot_deg']:.2f}°")
    print(f"кадров с ошибкой > {args.lost_cm:.0f} см: {len(lost)}"
          + (f" (первый {summary['first_over_threshold']})" if len(lost) else ""))
    print(f"PSNR по оценённым позам: {summary['psnr_test']:.2f} дБ; выравнивание глубины: {summary['align_sources']}")
    print(f"трекинг: медиана {tt['median']:.1f} мс, p95 {tt['p95']:.1f} мс; маппинг на ключевой кадр: "
          f"медиана {tm['median']:.0f} мс при интервале {kf_interval_ms:.0f} мс; всего {wall:.0f} с")
    if args.async_:
        a_ = {k: (v if v is not None else float("nan")) for k, v in summary["async"].items()}
        print(f"асинхронно: отставание от камеры медиана {a_['lag_ms_median']:.0f} мс, p95 {a_['lag_ms_p95']:.0f}, "
              f"макс {a_['lag_ms_max']:.0f}, в конце {a_['lag_ms_last']:.0f} мс; ключевых запрошено "
              f"{a_['kf_requested']}, отклонено {a_['kf_rejected']}; вставка {a_['insert_ms_median']:.0f} мс; "
              f"итераций маппинга всего {a_['map_iters_total']}; возраст снимка карты {a_['snap_age_frames_median']:.0f} кадр.")
    print(timers.table(skip=1))


if __name__ == "__main__":
    main()

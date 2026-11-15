# трекинг по готовой карте, построенной по эталонным позам (mapping_gt_poses.py --save-map).
# эталон даётся только для первого кадра, дальше постоянная скорость по своим оценкам.
# карта тут идеальная, так что это предел точности трекера: если здесь не сходится,
# в полной системе тем более не сойдётся
#   python scripts/mapping_gt_poses.py data/euroc/V1_01_easy --start 800 --tag map800 --save-map
#   python scripts/tracking_gt_map.py data/euroc/V1_01_easy outputs/mapping/V1_01_easy_map800/map.pt --levels 0.5:10,1:5
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsslam.data.euroc import EurocSequence  # noqa: E402
from gsslam.eval.trajectory import ate, rpe  # noqa: E402
from gsslam.map.gaussians import GaussianMap  # noqa: E402
from gsslam.track.direct_tracker import DirectTracker  # noqa: E402
from gsslam.track.render_tracker import RenderTracker  # noqa: E402
from gsslam.utils.timing import Timers  # noqa: E402


def pose_err(T, G):
    # ошибка сдвига (м) и поворота (град.); угол через scipy, arccos следа около нуля врёт
    dt = float(np.linalg.norm(T[:3, 3] - G[:3, 3]))
    return dt, float(np.degrees(Rotation.from_matrix(T[:3, :3].T @ G[:3, :3]).magnitude()))


def parse_levels(s):
    return [(float(a), int(b)) for a, b in (x.split(":") for x in s.split(","))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("map")
    ap.add_argument("--levels", default="0.5:10,1:5", help="масштаб:итерации через запятую, от грубого к точному")
    ap.add_argument("--lr-trans", type=float, default=3e-3)
    ap.add_argument("--lr-rot", type=float, default=3e-3)
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--n-frames", type=int, default=None, help="по умолчанию весь фрагмент карты")
    ap.add_argument("--lost-cm", type=float, default=30.0, help="порог ошибки, после которого трекинг считается сорванным")
    ap.add_argument("--init", choices=["cv", "gt"], default="cv",
                    help="cv: постоянная скорость по своим оценкам; gt: старт с эталонной позы каждого кадра "
                         "(диагностика: куда сходится оптимум карты)")
    ap.add_argument("--tracker", choices=["render", "direct"], default="render")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="outputs/tracking")
    args = ap.parse_args()

    dev = torch.device("cuda")
    m = torch.load(args.map, map_location="cpu", weights_only=False)
    gmap = GaussianMap.load(m["params"])
    n = args.n_frames or m["n_frames"]
    seq = EurocSequence(args.root, scale=m["scale"], start=m["start"], end=m["start"] + n)
    W, H = seq.width, seq.height
    assert (W, H) == (m["width"], m["height"])
    K = torch.as_tensor(seq.K, dtype=torch.float32, device=dev)
    cls = DirectTracker if args.tracker == "direct" else RenderTracker
    tracker = cls(gmap, K, W, H, levels=parse_levels(args.levels),
                            lr_trans=args.lr_trans, lr_rot=args.lr_rot, tol=args.tol)
    timers = Timers()

    ids = [i for i in range(len(seq)) if seq.gt_T_wc[i] is not None]
    images = {i: torch.as_tensor(seq.read_image(i), device=dev) for i in ids}   # читаем заранее, чтобы диск не попал в замер
    est, gt, init_err, err, iters, lost_at = [], [], [], [], [], None
    T_prev = T_prev2 = None
    for k, i in enumerate(ids):
        G = seq.gt_T_wc[i]
        if k == 0:
            est.append(G.copy()); gt.append(G)
            T_prev = G.copy()
            continue
        # позы в float64 на CPU, во float32 поворот перестаёт быть ортогональным
        if args.init == "gt":
            T_init = G.copy()
        else:
            T_init = T_prev if T_prev2 is None else T_prev @ np.linalg.inv(T_prev2) @ T_prev
        with timers("трекинг (кадр)"):
            r = tracker.track(images[i], T_init)
        Tn = r.T_wc
        init_err.append(pose_err(T_init, G))
        err.append(pose_err(Tn, G))
        iters.append(r.iters)
        est.append(Tn); gt.append(G)
        if lost_at is None and err[-1][0] * 100 > args.lost_cm:
            lost_at = k
        T_prev2, T_prev = T_prev, r.T_wc

    est, gt = np.stack(est), np.stack(gt)
    e, e0 = np.array(err), np.array(init_err)
    # карта в метрах, поэтому ATE и без масштаба (SE3), и с масштабом (Sim3)
    a3, a7 = ate(est, gt, with_scale=False), ate(est, gt, with_scale=True)
    rp = rpe(est, gt, delta=1, with_scale=False)
    t = timers.stats("трекинг (кадр)", skip=5)
    summary = {
        "sequence": Path(args.root).name, "tag": args.tag, "args": vars(args),
        "n_gaussians": len(gmap), "n_frames": len(ids),
        "ate_se3_cm": a3.rmse * 100, "ate_sim3_cm": a7.rmse * 100,
        "rpe_trans_cm": rp.trans_rmse * 100, "rpe_rot_deg": rp.rot_rmse_deg,
        "frame_err_cm_median": float(np.median(e[:, 0]) * 100), "frame_err_cm_max": float(e[:, 0].max() * 100),
        "frame_err_deg_median": float(np.median(e[:, 1])),
        "init_err_cm_median": float(np.median(e0[:, 0]) * 100), "init_err_deg_median": float(np.median(e0[:, 1])),
        "lost_at_frame": lost_at, "iters_mean": float(np.mean(iters)),
        "time_ms": t,
    }
    out = Path(args.out) / f"{Path(args.root).name}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out / "est.npy", est); np.save(out / "gt.npy", gt)
    print(f"{args.tag}: уровни {args.levels}, кадров {len(ids)}, итераций на кадр {summary['iters_mean']:.1f}")
    print(f"  ATE {summary['ate_se3_cm']:.2f} см (Sim3 {summary['ate_sim3_cm']:.2f}), RPE {summary['rpe_trans_cm']:.2f} см / {summary['rpe_rot_deg']:.3f}°")
    print(f"  ошибка кадра: медиана {summary['frame_err_cm_median']:.2f} см / {summary['frame_err_deg_median']:.3f}°, "
          f"макс {summary['frame_err_cm_max']:.1f} см; ошибка начального приближения {summary['init_err_cm_median']:.2f} см")
    print(f"  срыв: {'нет' if lost_at is None else f'на кадре {lost_at}'}; "
          f"время на кадр: медиана {t['median']:.1f} мс, p95 {t['p95']:.1f} мс")


if __name__ == "__main__":
    main()

# проверка загрузчика EuRoC: сводка по последовательности, кадр до/после undistort,
# и проверка, что эталонные позы сходятся с картинками (эпиполярная ошибка)
#   python scripts/check_euroc_loader.py data/euroc/V1_01_easy [--scale 0.5]
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsslam.data.euroc import EurocSequence  # noqa: E402
from gsslam.eval.trajectory import ate, rpe  # noqa: E402
from gsslam.utils.timing import Timers  # noqa: E402


def epipolar_check(seq, i, j):
    # медианное расстояние от ORB-точек до эпиполярных линий, пикс.
    # если позы, калибровка и undistort правильные, должно быть около пикселя
    a, b = seq[i], seq[j]
    ga = (a.image[:, :, 0] * 255).astype(np.uint8)
    gb = (b.image[:, :, 0] * 255).astype(np.uint8)
    orb = cv2.ORB_create(2000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    pa = np.array([ka[m.queryIdx].pt for m in matches])
    pb = np.array([kb[m.trainIdx].pt for m in matches])

    T_ba = np.linalg.inv(b.T_wc) @ a.T_wc   # из камеры a в камеру b
    R, t = T_ba[:3, :3], T_ba[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Kinv = np.linalg.inv(a.K)
    F = Kinv.T @ tx @ R @ Kinv
    ha = np.column_stack([pa, np.ones(len(pa))])
    hb = np.column_stack([pb, np.ones(len(pb))])
    lines = ha @ F.T   # эпиполярные линии в кадре b
    d = np.abs((hb * lines).sum(1)) / np.linalg.norm(lines[:, :2], axis=1)

    # ещё поворот по картинкам через RANSAC и насколько он отличается от эталона (град.)
    E, mask = cv2.findEssentialMat(pa, pb, a.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_est, _, _ = cv2.recoverPose(E, pa, pb, a.K, mask=mask)
    cos = np.clip((np.trace(R_est.T @ R) - 1) / 2, -1, 1)
    return float(np.median(d)), float(np.mean(d < 2.0)), float(np.degrees(np.arccos(cos)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/check_data")
    args = ap.parse_args()

    seq = EurocSequence(args.root, scale=args.scale)
    idx, ts, poses = seq.groundtruth()
    dur = seq.timestamps[-1] - seq.timestamps[0]
    path_len = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()
    print(f"последовательность: {Path(args.root).name}")
    print(f"кадров: {len(seq)}, длительность {dur:.1f} с, частота {(len(seq) - 1) / dur:.2f} Гц")
    print(f"размер кадра: {seq.width}x{seq.height} (исходный {seq.raw_size[0]}x{seq.raw_size[1]})")
    print(f"K:\n{np.round(seq.K, 2)}")
    print(f"кадров с эталонной позой: {len(idx)} ({idx[0]}..{idx[-1]}), длина пути {path_len:.1f} м")
    dt = np.diff(seq.timestamps)
    print(f"интервал между кадрами: медиана {np.median(dt) * 1e3:.1f} мс, макс {dt.max() * 1e3:.1f} мс")

    timers = Timers(sync_cuda=False)
    for i in range(0, min(len(seq), 200)):
        with timers("чтение + дисторсия"):
            fr = seq[i]
    print(timers.table())
    assert fr.image.shape == (seq.height, seq.width, 3) and fr.image.dtype == np.float32
    assert 0.0 <= fr.image.min() and fr.image.max() <= 1.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mid = int(idx[len(idx) // 2])
    raw = cv2.imread(str(seq._paths[mid]), cv2.IMREAD_GRAYSCALE)
    und = (seq[mid].image[:, :, 0] * 255).astype(np.uint8)
    und = cv2.resize(und, (raw.shape[1], raw.shape[0]))
    name = Path(args.root).name
    cv2.imwrite(str(out / f"{name}_raw_vs_undistorted.png"), np.hstack([raw, und]))
    print(f"пример кадра: {out / (name + '_raw_vs_undistorted.png')}")

    print(f"доля валидных пикселей после устранения дисторсии: {seq.valid_fraction:.4f}")
    print("\nпроверка эталонных поз по изображениям: медиана эпиполярной ошибки, пикс. / "
          "доля сопоставлений ближе 2 пикс. / расхождение поворота с RANSAC, град.")
    for gap in (5, 10, 20):
        vals = [epipolar_check(seq, int(idx[k]), int(idx[k + gap]))
                for k in range(len(idx) // 6, len(idx) - gap, len(idx) // 6)]
        print(f"  через {gap:>2} кадров: " + ", ".join(f"{d:.2f}/{p:.2f}/{r:.2f}" for d, p, r in vals))

    # проверка метрик: к эталону добавляем шум 2 см и меняем масштаб
    rng = np.random.default_rng(0)
    noisy = poses.copy()
    noisy[:, :3, 3] += rng.normal(0, 0.02, (len(poses), 3))
    noisy[:, :3, 3] *= 0.4   # как будто масштаб неизвестен
    a, r = ate(noisy, poses), rpe(noisy, poses, delta=20)
    print(f"\nсамопроверка метрик (шум 2 см, масштаб 0.4): ATE RMSE {a.rmse * 100:.2f} см "
          f"(ожидается ~{0.02 * np.sqrt(3) * 100:.2f}), найденный масштаб {a.scale:.3f} (ожидается 2.5), "
          f"RPE {r.trans_rmse * 100:.2f} см / {r.rot_rmse_deg:.3f}°")


if __name__ == "__main__":
    main()

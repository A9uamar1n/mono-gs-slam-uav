# метрики ATE и RPE. Позы T_wc (N, 4, 4).
# масштаб у монокулярной системы неизвестен, поэтому выравниваем по Sim(3) (Умеяма)
from dataclasses import dataclass

import numpy as np


def umeyama(src, dst, with_scale=True):
    # s, R, t такие, что dst ~ s R src + t (src, dst - (N, 3))
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_trajectory(est, gt, with_scale=True):
    s, R, t = umeyama(est[:, :3, 3], gt[:, :3, 3], with_scale)
    out = est.copy()
    out[:, :3, :3] = R @ est[:, :3, :3]
    out[:, :3, 3] = (s * (R @ est[:, :3, 3].T)).T + t
    return out, s, R, t


@dataclass
class AteResult:
    rmse: float
    mean: float
    median: float
    std: float
    max: float
    scale: float
    errors: np.ndarray   # ошибка по кадрам, м


def ate(est, gt, with_scale=True):
    aligned, s, _, _ = align_trajectory(est, gt, with_scale)
    e = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    return AteResult(float(np.sqrt((e ** 2).mean())), float(e.mean()), float(np.median(e)),
                     float(e.std()), float(e.max()), s, e)


@dataclass
class RpeResult:
    trans_rmse: float    # м на delta кадров
    rot_rmse_deg: float  # градусы на delta кадров
    delta: int
    trans_errors: np.ndarray
    rot_errors_deg: np.ndarray


def rpe(est, gt, delta=1, with_scale=True):
    # сначала приводим масштаб к эталону, иначе сдвиг сравнивать бессмысленно
    if with_scale:
        est, _, _, _ = align_trajectory(est, gt, True)
    n = len(est) - delta
    if n <= 0:
        raise ValueError("траектория короче интервала delta")
    te, re = np.empty(n), np.empty(n)
    for i in range(n):
        d_est = np.linalg.inv(est[i]) @ est[i + delta]
        d_gt = np.linalg.inv(gt[i]) @ gt[i + delta]
        err = np.linalg.inv(d_gt) @ d_est
        te[i] = np.linalg.norm(err[:3, 3])
        cos = np.clip((np.trace(err[:3, :3]) - 1) / 2, -1.0, 1.0)
        re[i] = np.degrees(np.arccos(cos))
    return RpeResult(float(np.sqrt((te ** 2).mean())), float(np.sqrt((re ** 2).mean())), delta, te, re)


def save_tum(path, timestamps, poses):
    # формат TUM: t tx ty tz qx qy qz qw
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    out = np.column_stack([timestamps, poses[:, :3, 3], q])
    np.savetxt(path, out, fmt="%.9f")

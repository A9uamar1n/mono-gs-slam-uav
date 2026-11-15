"""Проверка метрик траектории на синтетике. Запуск: python tests/test_trajectory.py или pytest."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsslam.eval.trajectory import ate, rpe, umeyama  # noqa: E402


def _random_traj(n=200, seed=0):
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, 3] = np.cumsum(rng.normal(0, 0.05, (n, 3)), axis=0)
    T[:, :3, :3] = Rotation.from_rotvec(np.cumsum(rng.normal(0, 0.02, (n, 3)), axis=0)).as_matrix()
    return T


def _similarity(T, s, R, t):
    out = T.copy()
    out[:, :3, :3] = R @ T[:, :3, :3]
    out[:, :3, 3] = (s * (R @ T[:, :3, 3].T)).T + t
    return out


def test_umeyama_recovers_similarity():
    gt = _random_traj()
    R = Rotation.from_euler("xyz", [20, -35, 70], degrees=True).as_matrix()
    s, t = 2.7, np.array([1.0, -2.0, 0.5])
    est = _similarity(gt, 1 / s, R.T, -(R.T @ t) / s)   # обратное преобразование
    s2, R2, t2 = umeyama(est[:, :3, 3], gt[:, :3, 3])
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R, atol=1e-9) and np.allclose(t2, t, atol=1e-9)


def test_ate_zero_under_similarity():
    gt = _random_traj()
    R = Rotation.from_euler("xyz", [10, 50, -30], degrees=True).as_matrix()
    est = _similarity(gt, 0.31, R, np.array([3.0, 1.0, -4.0]))
    r = ate(est, gt)
    assert r.rmse < 1e-9 and abs(r.scale - 1 / 0.31) < 1e-9
    assert rpe(est, gt, delta=5).trans_rmse < 1e-9
    assert rpe(est, gt, delta=5).rot_rmse_deg < 1e-5


def test_ate_matches_known_noise():
    gt = _random_traj(n=5000)
    rng = np.random.default_rng(1)
    est = gt.copy()
    sigma = 0.02
    est[:, :3, 3] += rng.normal(0, sigma, (len(gt), 3))
    r = ate(est, gt, with_scale=False)
    expected = sigma * np.sqrt(3)        # RMSE модуля трёхмерной гауссовой ошибки
    assert abs(r.rmse - expected) / expected < 0.05


def test_without_scale_detects_scale_error():
    gt = _random_traj()
    est = _similarity(gt, 0.5, np.eye(3), np.zeros(3))
    assert ate(est, gt, with_scale=False).rmse > 1e-3
    assert ate(est, gt, with_scale=True).rmse < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)

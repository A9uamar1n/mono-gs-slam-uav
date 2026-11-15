# инициализация по двум кадрам: ORB -> существенная матрица -> поза и триангуляция.
# масштаб произвольный, фиксируем медианную глубину точек первого кадра = target_depth
from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry.sparse import SparseDepth, SparseTriangulator


@dataclass
class InitResult:
    T_wc_b: np.ndarray        # поза второго кадра (первый в начале координат)
    sparse_a: SparseDepth
    sparse_b: SparseDepth
    n_inliers: int
    median_px: float


class TwoViewInit:
    def __init__(self, K, tri, min_px=12.0, min_inliers=100, min_points=80, target_depth=2.0):
        self.K = K
        self.tri = tri
        self.min_px = min_px
        self.min_inliers = min_inliers
        self.min_points = min_points
        self.target_depth = target_depth

    def try_init(self, feat_a, feat_b):
        (ka, da), (kb, db) = feat_a, feat_b
        if da is None or db is None or len(ka) < 50 or len(kb) < 50:
            return None
        m = self.tri.matcher.match(da, db)
        if len(m) < self.min_inliers:
            return None
        pa = np.array([ka[x.queryIdx].pt for x in m], dtype=np.float64)
        pb = np.array([kb[x.trainIdx].pt for x in m], dtype=np.float64)
        med = float(np.median(np.linalg.norm(pa - pb, axis=1)))
        if med < self.min_px:
            return None
        E, mask = cv2.findEssentialMat(pa, pb, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            return None
        n, R, t, _ = cv2.recoverPose(E, pa, pb, self.K, mask=mask)
        if n < self.min_inliers:
            return None
        # |t| = 1, берём базу 0.1, чтобы глубины прошли фильтры триангуляции,
        # нормальный масштаб ставим ниже
        T_ba = np.eye(4)
        T_ba[:3, :3] = R
        T_ba[:3, 3] = t[:, 0] * 0.1
        T_wc_b = np.linalg.inv(T_ba)
        eye = np.eye(4)
        sa = self.tri.triangulate(feat_a, feat_b, self.K, eye, T_wc_b)
        if len(sa.depth) < self.min_points:
            return None
        s = self.target_depth / float(np.median(sa.depth))
        T_wc_b[:3, 3] *= s
        sb = self.tri.triangulate(feat_b, feat_a, self.K, T_wc_b, eye)
        sa = SparseDepth(sa.uv, sa.depth * s, sa.reproj, sa.parallax)
        return InitResult(T_wc_b, sa, sb, int(n), med)

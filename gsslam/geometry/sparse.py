# разреженные точки глубины: триангуляция ORB-сопоставлений между кадрами с известными позами.
# по ним потом выравнивается масштаб и сдвиг глубины от сети
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SparseDepth:
    uv: np.ndarray        # (N, 2) пиксели в опорном кадре
    depth: np.ndarray     # (N,) глубина по Z, м
    reproj: np.ndarray    # (N,) ошибка репроекции, пикс (макс. по двум кадрам)
    parallax: np.ndarray  # (N,) параллакс, градусы


def _gray_u8(image):
    g = image[:, :, 0] if image.ndim == 3 else image
    return np.clip(g * 255.0, 0, 255).astype(np.uint8)


def _proj(K, T_wc):
    T_cw = np.linalg.inv(T_wc)
    return K @ T_cw[:3, :]


class SparseTriangulator:
    def __init__(self, n_features=3000, max_epipolar_px=1.5, max_reproj_px=1.5,
                 min_parallax_deg=1.0, max_depth=30.0):
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_epipolar_px = max_epipolar_px
        self.max_reproj_px = max_reproj_px
        self.min_parallax_deg = min_parallax_deg
        self.max_depth = max_depth

    def features(self, image):
        return self.orb.detectAndCompute(_gray_u8(image), None)

    def triangulate(self, feat_a, feat_b, K, T_wc_a, T_wc_b):
        # точки глубины для кадра a по паре (a, b)
        (ka, da), (kb, db) = feat_a, feat_b
        empty = SparseDepth(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0))
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return empty
        m = self.matcher.match(da, db)
        if len(m) < 8:
            return empty
        pa = np.array([ka[x.queryIdx].pt for x in m], dtype=np.float64)
        pb = np.array([kb[x.trainIdx].pt for x in m], dtype=np.float64)

        # выкидываем плохие сопоставления по расстоянию до эпиполярной линии
        T_ba = np.linalg.inv(T_wc_b) @ T_wc_a
        R, t = T_ba[:3, :3], T_ba[:3, 3]
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        Kinv = np.linalg.inv(K)
        F = Kinv.T @ tx @ R @ Kinv
        ha = np.column_stack([pa, np.ones(len(pa))])
        hb = np.column_stack([pb, np.ones(len(pb))])
        lb = ha @ F.T
        d = np.abs((hb * lb).sum(1)) / (np.linalg.norm(lb[:, :2], axis=1) + 1e-12)
        keep = d < self.max_epipolar_px
        pa, pb = pa[keep], pb[keep]
        if len(pa) < 8:
            return empty

        Pa, Pb = _proj(K, T_wc_a), _proj(K, T_wc_b)
        X = cv2.triangulatePoints(Pa, Pb, pa.T, pb.T)
        X = (X[:3] / X[3]).T   # (N, 3) в мире

        def project(P, X):
            x = (P @ np.column_stack([X, np.ones(len(X))]).T).T
            return x[:, :2] / x[:, 2:3], x[:, 2]

        ua, za = project(Pa, X)
        ub, zb = project(Pb, X)
        reproj = np.maximum(np.linalg.norm(ua - pa, axis=1), np.linalg.norm(ub - pb, axis=1))

        ca, cb = T_wc_a[:3, 3], T_wc_b[:3, 3]
        ra, rb = X - ca, X - cb
        cosang = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
        parallax = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

        ok = ((za > 0.05) & (zb > 0.05) & (za < self.max_depth) & np.isfinite(za) &
              (reproj < self.max_reproj_px) & (parallax > self.min_parallax_deg))
        return SparseDepth(pa[ok], za[ok], reproj[ok], parallax[ok])

    def sparse_depth(self, feat_ref, K, T_wc_ref, neighbours):
        # точки по нескольким соседям [(feat, T_wc), ...]; если точка нашлась несколько
        # раз, оставляем вариант с самым большим параллаксом
        parts = [self.triangulate(feat_ref, f, K, T_wc_ref, T) for f, T in neighbours]
        parts = [p for p in parts if len(p.depth)]
        if not parts:
            return SparseDepth(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0))
        uv = np.concatenate([p.uv for p in parts])
        depth = np.concatenate([p.depth for p in parts])
        reproj = np.concatenate([p.reproj for p in parts])
        par = np.concatenate([p.parallax for p in parts])
        order = np.argsort(-par)
        key = np.round(uv[order] * 4).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        sel = order[np.sort(first)]
        return SparseDepth(uv[sel], depth[sel], reproj[sel], par[sel])

# выравнивание глубины сети по опорным точкам и карта уверенности (из 6 семестра).
# Depth Anything выдаёт обратную глубину, поэтому подгоняем 1/D = s * p + t.
# вместо штрафа за отрицательную глубину просто маска: где s * p + t <= eps - невалидно
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Alignment:
    s: float
    t: float
    n_points: int
    n_inliers: int
    rel_err: float          # медиана относительной ошибки глубины на опорных точках
    ok: bool


def _bilinear(img, uv):
    h, w = img.shape
    u = np.clip(uv[:, 0], 0, w - 1.001)
    v = np.clip(uv[:, 1], 0, h - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = u - u0, v - v0
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv) +
            img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def fit_scale_shift(pred, uv, depth, weights=None, inverse=True, iters=8, min_points=30):
    # s, t взвешенным МНК с весами Хьюбера (IRLS)
    n = len(depth)
    if n < min_points:
        return Alignment(1.0, 0.0, n, 0, float("inf"), False)
    x = _bilinear(pred, uv).astype(np.float64)
    y = (1.0 / depth if inverse else depth).astype(np.float64)
    w0 = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64) + 1e-6
    w = w0.copy()
    s, t = 1.0, 0.0
    for _ in range(iters):
        sw = w.sum()
        mx, my = (w * x).sum() / sw, (w * y).sum() / sw
        var = (w * (x - mx) ** 2).sum()
        if var < 1e-12:
            return Alignment(1.0, 0.0, n, 0, float("inf"), False)
        s = (w * (x - mx) * (y - my)).sum() / var
        t = my - s * mx
        r = y - (s * x + t)
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        k = 1.345 * sigma
        w = w0 * np.where(np.abs(r) <= k, 1.0, k / (np.abs(r) + 1e-12))
    r = y - (s * x + t)
    sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
    inl = np.abs(r) < 2.5 * sigma
    fit = s * x + t
    d_fit = 1.0 / np.maximum(fit, 1e-6) if inverse else fit
    rel = np.abs(d_fit - depth) / depth
    ok = bool(s > 0 and inl.sum() >= min_points and np.median(rel) < 0.25) if inverse \
        else bool(s > 0 and inl.sum() >= min_points)
    return Alignment(float(s), float(t), n, int(inl.sum()), float(np.median(rel)), ok)


def apply_alignment(pred, a, inverse=True, max_depth=40.0):
    # метрическая глубина и маска
    v = a.s * pred + a.t
    if inverse:
        valid = v > 1.0 / max_depth
        depth = np.where(valid, 1.0 / np.maximum(v, 1e-6), 0.0)
    else:
        valid = (v > 0.05) & (v < max_depth)
        depth = np.where(valid, v, 0.0)
    return depth.astype(np.float32), valid


def confidence_map(gray, depth, valid, w_e=0.4, w_t=0.2, w_g=0.4):
    # C = w_e*C_e + w_t*C_t + w_g*C_g:
    # C_e - меньше на перепадах глубины (Кэнни), C_t - больше где есть текстура (Лаплас),
    # C_g - падает с ростом относительного градиента глубины
    g8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    inv = np.where(valid, 1.0 / np.maximum(depth, 1e-3), 0.0)
    lo, hi = np.percentile(inv[valid], [1, 99]) if valid.any() else (0.0, 1.0)
    inv8 = np.clip((inv - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)

    edges = cv2.Canny(inv8, 20, 60)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    c_e = 1.0 - cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 1.5)

    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g8, (0, 0), 1.0), cv2.CV_32F))
    lap = cv2.GaussianBlur(lap, (0, 0), 3.0)
    c_t = np.clip(lap / (np.percentile(lap, 90) + 1e-6), 0, 1)

    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    rel_grad = np.sqrt(gx ** 2 + gy ** 2) / np.maximum(depth, 1e-3)
    c_g = np.exp(-rel_grad / 0.05)

    c = w_e * c_e + w_t * c_t + w_g * c_g
    return np.where(valid, np.clip(c, 0, 1), 0.0).astype(np.float32)


# гибкое выравнивание: s и t не константы, а гладкие функции от (u, v)
#   1/D = s(u,v) * p + t(u,v),  s = sum a_k B_k,  t = sum b_k B_k
# базис: linear - 1, u, v; grid3 - билинейные "шапочки" на сетке 3x3.
# по скану V1_01_easy ошибка формы 12,6% -> 9,8% (linear) и 7,5% (grid3).
# точек мало и они шумные, поэтому тянем коэффициенты к обычному (s, t),
# а если точек не хватает - просто обычное выравнивание

@dataclass
class FieldAlignment:
    kind: str
    coef: np.ndarray          # (2K,): сначала масштаб, потом сдвиг
    n_points: int
    rel_err: float
    ok: bool


_MIN_POINTS = {"linear": 100, "grid3": 250}


def _basis(kind, u, v):
    # u, v в [0, 1] -> (N, K)
    if kind == "linear":
        return np.stack([np.ones_like(u), u - 0.5, v - 0.5], 1)
    if kind == "grid3":
        c = np.array([0.0, 0.5, 1.0])
        bu = np.maximum(0, 1 - np.abs(u[:, None] - c[None]) * 2)   # (N, 3)
        bv = np.maximum(0, 1 - np.abs(v[:, None] - c[None]) * 2)
        return (bv[:, :, None] * bu[:, None, :]).reshape(len(u), 9)
    raise ValueError(kind)


def _prior(kind, s, t):
    if kind == "linear":
        return np.array([s, 0, 0, t, 0, 0], dtype=np.float64)
    k = 9   # шапочки в сумме дают 1, поэтому во всех узлах (s, t)
    return np.concatenate([np.full(k, s), np.full(k, t)])


def fit_scale_shift_field(pred, uv, depth, kind="linear", lam=0.3, iters=6):
    a0 = fit_scale_shift(pred, uv, depth)
    if not a0.ok or kind == "affine" or len(depth) < _MIN_POINTS[kind]:
        return a0
    h, w = pred.shape
    x = _bilinear(pred, uv).astype(np.float64)
    y = 1.0 / depth.astype(np.float64)
    B = _basis(kind, uv[:, 0] / w, uv[:, 1] / h)
    A = np.concatenate([B * x[:, None], B], 1)
    c0 = _prior(kind, a0.s, a0.t)
    # штраф за отход от (s, t), каждый столбец нормируем на его СКЗ
    col = np.sqrt((A ** 2).mean(0)) + 1e-12
    n = len(y)
    wts = np.ones(n)
    c = c0.copy()
    for _ in range(iters):
        Aw = A * wts[:, None]
        M = A.T @ Aw + lam * n * np.diag(col ** 2)
        rhs = Aw.T @ y + lam * n * (col ** 2) * c0
        c = np.linalg.solve(M, rhs)
        r = y - A @ c
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        k = 1.345 * sigma
        wts = np.where(np.abs(r) <= k, 1.0, k / (np.abs(r) + 1e-12))
    fit = A @ c
    d_fit = 1.0 / np.maximum(fit, 1e-6)
    rel = float(np.median(np.abs(d_fit - depth) / depth))
    ok = bool(np.all(fit > 0) and rel < 0.25)
    if not ok:
        return a0
    return FieldAlignment(kind, c, n, rel, True)


def apply_field_alignment(pred, a, max_depth=40.0):
    if isinstance(a, Alignment):
        return apply_alignment(pred, a)
    h, w = pred.shape
    vv, uu = np.mgrid[0:h, 0:w]
    B = _basis(a.kind, (uu.ravel() + 0.0) / w, (vv.ravel() + 0.0) / h)
    k = B.shape[1]
    s = (B @ a.coef[:k]).reshape(h, w)
    t = (B @ a.coef[k:]).reshape(h, w)
    v = s * pred + t
    valid = v > 1.0 / max_depth
    depth = np.where(valid, 1.0 / np.maximum(v, 1e-6), 0.0)
    return depth.astype(np.float32), valid

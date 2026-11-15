# прямой трекер (Гаусс-Ньютон). Карту рендерим один раз на кадр из начальной позы,
# получаем опорную картинку R и глубину D, потом кадр I выравниваем по ним:
#   X = D(u) K^-1 [u, 1],  p = pi(K T_cr X),  r(u) = I(p) - (exp(a) R(u) + b)
# T_cr - текущая камера относительно опорной, a, b - экспозиция.
# якобиан аналитический, веса Хьюбера, немного демпфирования.
# backward через растеризатор не нужен, поэтому намного быстрее RenderTracker
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

from .render_tracker import se3_exp_np


@dataclass
class TrackResult:
    T_wc: np.ndarray
    iters: int
    loss: float
    coverage: float
    exposure: tuple[float, float]
    history: list[float] = field(default_factory=list)


def _grad(img):
    gx = torch.zeros_like(img)
    gy = torch.zeros_like(img)
    gx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    gy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    return gx, gy


class DirectTracker:
    def __init__(self, gmap, K, width, height, render_scale=0.5, levels=((0.5, 6), (1.0, 6)),
                 alpha_min=0.95, min_grad=0.01, damping=1e-4, huber_k=1.345, stop_step=1e-4,
                 cull_margin=0.2, max_points=3000, ab_prior=0.1, ab_max=(0.7, 0.3), **_ignored):
        self.gmap = gmap
        self.K = K
        self.W, self.H = width, height
        self.render_scale = render_scale
        self.levels = list(levels)    # (масштаб от разрешения рендера, итераций)
        self.alpha_min = alpha_min
        self.min_grad = min_grad
        self.damping = damping
        self.huber_k = huber_k
        self.stop_step = stop_step
        self.cull_margin = cull_margin
        self.max_points = max_points  # точек на уровень
        # ограничение на экспозицию: без него на слабой текстуре e^a -> 0, рендер
        # становится константой b, ошибка не зависит от позы и трекинг улетает
        self.ab_prior = ab_prior
        self.ab_max = np.asarray(ab_max, dtype=np.float64)
        self.exposure = np.zeros(2)
        self._frozen = None
        self._dev = K.device

    def refresh_map(self):
        p = self.gmap.params
        self._frozen = (p["means"].detach(), p["quats"].detach(), p["log_scales"].detach().exp(),
                        torch.sigmoid(p["opac"].detach()), torch.sigmoid(p["colors"].detach()))

    @torch.no_grad()
    def _render(self, T_wc, K, w, h):
        m, q, s, o, c = self._frozen
        V = torch.as_tensor(np.linalg.inv(T_wc), dtype=torch.float32, device=self._dev)
        out, alpha, _ = rasterization(m, q, s, o, c, V[None], K[None], w, h, near_plane=0.05,
                                      far_plane=100.0, render_mode="RGB+ED", packed=False)
        return out[0, ..., 0], out[0, ..., 3], alpha[0, ..., 0]

    @torch.no_grad()
    def track(self, image, T_init):
        if self._frozen is None:
            self.refresh_map()
        T0 = T_init.detach().cpu().double().numpy() if torch.is_tensor(T_init) else np.asarray(T_init, np.float64)
        u_, _, vt = np.linalg.svd(T0[:3, :3])
        T0 = T0.copy()
        T0[:3, :3] = u_ @ vt

        gray = image[..., 0] if image.dim() == 3 else image
        rw, rh = int(round(self.W * self.render_scale)), int(round(self.H * self.render_scale))
        Kr = self.K.clone()
        Kr[0, :] *= rw / self.W
        Kr[1, :] *= rh / self.H
        R0, D0, A0 = self._render(T0, Kr, rw, rh)
        cov_map = float((A0 > self.alpha_min).float().mean())   # нужно для выбора ключевых кадров
        I0 = F.interpolate(gray[None, None], size=(rh, rw), mode="area")[0, 0] if (rw, rh) != (self.W, self.H) else gray

        T_cr = np.eye(4)
        ab = self.exposure.copy()
        iters, history, coverage, loss_v = 0, [], cov_map, float("nan")
        # уровни считаем на GPU и разом копируем на CPU: сам цикл ГН по 5-20 тыс. точек
        # в numpy быстрее, чем куча маленьких ядер CUDA
        lv_data = []
        for scale, n_it in self.levels:
            if scale == 1.0:
                R, D, A, I, K = R0, D0, A0, I0, Kr
            else:
                w, h = max(8, int(round(rw * scale))), max(8, int(round(rh * scale)))
                pool = lambda x: F.interpolate(x[None, None], size=(h, w), mode="area")[0, 0]  # noqa: E731
                R, A, I = pool(R0), pool(A0), pool(I0)
                D = pool(D0 * A0) / pool(A0).clamp(min=1e-6)
                K = Kr.clone()
                K[0, :] *= w / rw
                K[1, :] *= h / rh
            lv_data.append((torch.stack([R, D, A, I]), K, n_it))
        flat = torch.cat([x[0].flatten() for x in lv_data]).cpu().numpy()
        off = 0
        for stack_t, K, n_it in lv_data:
            h, w = stack_t.shape[1:]
            R, D, A, I = flat[off:off + 4 * h * w].reshape(4, h, w)
            off += 4 * h * w
            fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
            T_cr, ab, it, loss_l = self._gn_level(R, D, A, I, fx, fy, cx, cy, n_it, T_cr, ab, history)
            iters += it
            if loss_l is not None:
                loss_v = loss_l
        if not (np.isfinite(T_cr).all() and np.isfinite(ab).all()):
            # разошлось: экспозицию не сохраняем, позу ставим NaN (её отбросит GsSlam._accept)
            return TrackResult(np.full((4, 4), np.nan), 0, float("nan"), coverage, (float(self.exposure[0]),
                               float(self.exposure[1])), history)
        self.exposure = ab
        T = T0 @ np.linalg.inv(T_cr)
        return TrackResult(T, iters, loss_v, coverage, (float(ab[0]), float(ab[1])), history)

    def _gn_level(self, R, D, A, I, fx, fy, cx, cy, n_it, T_cr, ab, history):
        h, w = R.shape
        gxR, gyR = _grad_np(R)
        gm = np.hypot(gxR, gyR)
        sel = (A > self.alpha_min) & (D > 0.05) & (gm > self.min_grad)
        vv, uu = np.nonzero(sel)
        if len(vv) < 50:
            return T_cr, ab, 0, None
        if self.max_points and len(vv) > self.max_points:
            # берём точки с большим градиентом, но равномерно по кадру:
            # в каждой клетке 8x8 сортируем по градиенту и берём лучшие из всех клеток
            g = gm[vv, uu]
            cell = (vv // 8) * ((w + 7) // 8) + (uu // 8)
            order = np.lexsort((-g, cell))
            rank = np.empty_like(order)
            first = np.r_[0, np.nonzero(np.diff(cell[order]))[0] + 1]
            counts = np.diff(np.r_[first, len(order)])
            rank[order] = np.arange(len(order)) - np.repeat(first, counts)
            keep = np.argsort(rank, kind="stable")[:self.max_points]
            vv, uu = vv[keep], uu[keep]
        d = D[vv, uu]
        Xr = np.stack([(uu - cx) / fx * d, (vv - cy) / fy * d, d], 1).astype(np.float32)
        Rv = R[vv, uu]
        gxI, gyI = _grad_np(I)
        chans = np.stack([I, gxI, gyI], -1)   # (h, w, 3)
        it, loss_v = 0, None
        for _ in range(n_it):
            Rm = T_cr[:3, :3].astype(np.float32)
            Xc = Xr @ Rm.T + T_cr[:3, 3].astype(np.float32)
            x, y, z = Xc[:, 0], Xc[:, 1], np.maximum(Xc[:, 2], 1e-3)
            pu = fx * x / z + cx
            pv = fy * y / z + cy
            inb = (pu > 1) & (pu < w - 2) & (pv > 1) & (pv < h - 2) & (Xc[:, 2] > 0.05)
            if inb.sum() < 50:
                break
            x, y, z, pu, pv, Rq = x[inb], y[inb], z[inb], pu[inb], pv[inb], Rv[inb]
            u0 = pu.astype(np.int32)
            v0 = pv.astype(np.int32)
            du, dv = (pu - u0)[:, None], (pv - v0)[:, None]
            smp = (chans[v0, u0] * (1 - du) * (1 - dv) + chans[v0, u0 + 1] * du * (1 - dv) +
                   chans[v0 + 1, u0] * (1 - du) * dv + chans[v0 + 1, u0 + 1] * du * dv)
            Iw, gu, gv = smp[:, 0], smp[:, 1], smp[:, 2]
            ea = np.float32(np.exp(ab[0]))
            r = Iw - (ea * Rq + np.float32(ab[1]))
            iz = 1.0 / z
            xz, yz = x * iz, y * iz
            # якобиан по (rho, phi, a, b), возмущение слева в системе текущей камеры
            J = np.empty((len(r), 8), np.float32)
            J[:, 0] = gu * fx * iz
            J[:, 1] = gv * fy * iz
            J[:, 2] = -(gu * fx * xz + gv * fy * yz) * iz
            J[:, 3] = -gu * fx * xz * yz - gv * fy * (1 + yz * yz)
            J[:, 4] = gu * fx * (1 + xz * xz) + gv * fy * xz * yz
            J[:, 5] = -gu * fx * yz + gv * fy * xz
            J[:, 6] = -ea * Rq
            J[:, 7] = -1.0
            ar = np.abs(r)
            kk = self.huber_k * max(1.4826 * float(np.median(ar)), 1e-4)
            wts = np.where(ar <= kk, 1.0, kk / np.maximum(ar, 1e-12)).astype(np.float32)
            Jw = J * wts[:, None]
            Hn = (Jw.T @ J).astype(np.float64)
            gn = (Jw.T @ r).astype(np.float64)
            loss_v = float((wts * r * r).sum() / len(r))
            if self.ab_prior > 0:   # штраф N * lambda * (a^2 + b^2)
                lam = self.ab_prior * float(wts.sum())
                Hn[6, 6] += lam
                Hn[7, 7] += lam
                gn[6:] += lam * ab
            Hn += self.damping * np.diag(np.diag(Hn) + 1e-9)
            try:
                step = -np.linalg.solve(Hn, gn)
            except np.linalg.LinAlgError:
                break
            T_cr = se3_exp_np(step[:6]) @ T_cr
            ab = np.clip(ab + step[6:], -self.ab_max, self.ab_max)
            it += 1
            history.append(loss_v)
            if np.abs(step[:6]).max() < self.stop_step:
                break
        return T_cr, ab, it, loss_v


def _grad_np(img):
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    gy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    return gx, gy

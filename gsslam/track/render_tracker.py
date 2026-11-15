# трекер через дифференцируемый рендеринг (первый вариант, потом заменён прямым).
# оптимизируем V = T_cw по замороженной карте:
#   1) рендер из V, L1 по яркости на пикселях с alpha > alpha_min
#   2) backward, gsplat отдаёт G = dL/dV (4x4)
#   3) переводим G в градиент по (rho, phi): M = G V^T,
#      dL/drho = M[:3, 3], dL/dphi = (M21 - M12, M02 - M20, M10 - M01)
#   4) Adam и exp SE(3) на CPU, там всего 6 чисел
# через autograd сборка позы стоила 2 мс на итерацию (много мелких ядер), так быстрее.
# экспозиция: exp(a) * I + b. пирамида levels = (масштаб, итераций), от грубого к точному
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def se3_exp_np(xi):
    # xi = (rho, phi), формула Родрига, float64
    rho, phi = xi[:3], xi[3:]
    th = np.linalg.norm(phi)
    P = _skew(phi)
    P2 = P @ P
    if th < 1e-6:
        a, b, c = 1.0, 0.5, 1.0 / 6
    else:
        a, b, c = np.sin(th) / th, (1 - np.cos(th)) / th ** 2, (th - np.sin(th)) / th ** 3
    R = np.eye(3) + a * P + b * P2
    Vm = np.eye(3) + b * P + c * P2
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = Vm @ rho
    return T


@dataclass
class TrackResult:
    T_wc: np.ndarray
    iters: int
    loss: float
    coverage: float                     # доля пикселей под картой на последнем уровне
    exposure: tuple[float, float]
    history: list[float] = field(default_factory=list)


class _Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros_like(lr)
        self.v = np.zeros_like(lr)
        self.t = 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return -self.lr * mh / (np.sqrt(vh) + self.eps)


class RenderTracker:
    def __init__(self, gmap, K, width, height, levels=((0.5, 10), (1.0, 5)), lr_trans=3e-3,
                 lr_rot=3e-3, lr_exposure=1e-2, alpha_min=0.95, tol=0.0, cull_margin=0.2,
                 **_ignored):
        self.gmap = gmap
        self.K = K
        self.W, self.H = width, height
        self.levels = list(levels)
        self.lr = np.array([lr_trans] * 3 + [lr_rot] * 3 + [lr_exposure] * 2)
        self.alpha_min = alpha_min
        self.tol = tol                  # выход с уровня, если потери почти не падают
        self.exposure = np.zeros(2)     # переходит на следующий кадр
        self.cull_margin = cull_margin  # запас в долях кадра, None - не отсекать
        self._frozen = None
        self._dev = K.device

    def refresh_map(self):
        # копия параметров карты без градиентов, звать после изменения карты
        p = self.gmap.params
        self._frozen = (p["means"].detach(), p["quats"].detach(), p["log_scales"].detach().exp(),
                        torch.sigmoid(p["opac"].detach()), torch.sigmoid(p["colors"].detach()))

    def _cull(self, V):
        # оставляем только гауссианы, которые видно из начальной позы (с запасом).
        # в кадр попадает меньше половины карты, а gsplat тратит время на все
        m, q, s, o, c = self._frozen
        if self.cull_margin is None:
            return m, q, s, o, c
        Vt = torch.as_tensor(V, dtype=torch.float32, device=self._dev)
        pc = m @ Vt[:3, :3].T + Vt[:3, 3]
        z = pc[:, 2]
        r = s.max(dim=1).values * 3   # радиус 3 сигмы
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        zc = z.clamp(min=1e-3)
        mw, mh = self.cull_margin * self.W, self.cull_margin * self.H
        u = pc[:, 0] / zc * fx + cx
        v = pc[:, 1] / zc * fy + cy
        ru, rv = r / zc * fx, r / zc * fy
        keep = ((z + r > 0.05) & (u + ru > -mw) & (u - ru < self.W + mw) &
                (v + rv > -mh) & (v - rv < self.H + mh))
        idx = keep.nonzero()[:, 0]
        return m[idx], q[idx], s[idx], o[idx], c[idx]

    def _pyramid(self, gray):
        out = []
        for scale, n_it in self.levels:
            w, h = max(8, int(round(self.W * scale))), max(8, int(round(self.H * scale)))
            tgt = gray if (w, h) == (self.W, self.H) else \
                F.interpolate(gray[None, None], size=(h, w), mode="area")[0, 0]
            K = self.K.clone()
            K[0, :] *= w / self.W
            K[1, :] *= h / self.H
            out.append((tgt, K, w, h, n_it))
        return out

    def track(self, image, T_init):
        # image (H, W, 3) или (H, W) на GPU, T_init - начальная T_wc
        if self._frozen is None:
            self.refresh_map()
        gray = image[..., 0] if image.dim() == 3 else image
        T0 = T_init.detach().cpu().double().numpy() if torch.is_tensor(T_init) else np.asarray(T_init, np.float64)
        u, _, vt = np.linalg.svd(T0[:3, :3])   # ортогонализуем поворот
        T0 = T0.copy()
        T0[:3, :3] = u @ vt
        V = np.linalg.inv(T0)
        m, q, s, o, c = self._cull(V)
        expo = self.exposure.copy()
        adam = _Adam(self.lr)

        history = []
        iters, coverage, loss_v = 0, 0.0, float("nan")
        for tgt, K, w, h, n_it in self._pyramid(gray):
            prev = None
            for _ in range(n_it):
                vm = torch.tensor(V, dtype=torch.float32, device=self._dev).requires_grad_(True)
                ex = torch.tensor(expo, dtype=torch.float32, device=self._dev).requires_grad_(True)
                out, alpha, _ = rasterization(m, q, s, o, c, vm[None], K[None], w, h,
                                              near_plane=0.05, far_plane=100.0, packed=False)
                mask = (alpha[0, ..., 0].detach() > self.alpha_min).float()
                n = mask.sum().clamp(min=1)
                pred = torch.exp(ex[0]) * out[0, ..., 0] + ex[1]
                loss = ((pred - tgt).abs() * mask).sum() / n
                loss.backward()
                pack = torch.cat([vm.grad.flatten(), ex.grad, loss.detach()[None], n[None]]).double().cpu().numpy()
                G, g_ex, loss_v, n_pix = pack[:16].reshape(4, 4), pack[16:18], float(pack[18]), pack[19]
                if n_pix < 100:
                    break
                M = G @ V.T
                g = np.concatenate([M[:3, 3], [M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]], g_ex])
                d = adam.step(g)
                V = se3_exp_np(d[:6]) @ V
                expo = expo + d[6:]
                iters += 1
                history.append(loss_v)
                coverage = n_pix / (w * h)
                if self.tol > 0 and prev is not None and (prev - loss_v) < self.tol * prev:
                    break
                prev = loss_v
        self.exposure = expo
        return TrackResult(np.linalg.inv(V), iters, loss_v, coverage, (float(expo[0]), float(expo[1])), history)

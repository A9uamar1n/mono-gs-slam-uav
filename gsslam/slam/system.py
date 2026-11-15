# монокулярный SLAM на 3DGS, синхронный вариант (маппинг в том же потоке).
# кадры подаются по одному в process():
#  - инициализация по двум кадрам (init.py)
#  - потом каждый кадр трекается по карте, начальная поза из модели постоянной скорости
#  - ключевой кадр: прошло kf_max кадров или покрытие < cov_min (но не раньше kf_min),
#    и камера при этом сдвинулась (kf_motion_*), иначе на стоянке сотни одинаковых кадров
#  - ключевой кадр отдаём в mapper.py
from dataclasses import dataclass

import numpy as np
import torch

from ..geometry.sparse import SparseTriangulator
from ..track.direct_tracker import DirectTracker
from ..track.render_tracker import RenderTracker
from .init import TwoViewInit
from .mapper import Mapper


def _interp(Ta, Tb, a):
    # slerp для поворота, линейно для положения
    from scipy.spatial.transform import Rotation, Slerp
    R = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ta[:3, :3], Tb[:3, :3]])))([a]).as_matrix()[0]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (1 - a) * Ta[:3, 3] + a * Tb[:3, 3]
    return T


@dataclass
class SlamConfig:
    tracker: str = "render"              # render - через растеризатор, direct - Гаусс-Ньютон
    track_levels: tuple = ((0.5, 10),)
    track_points: int = 0
    lr_trans: float = 3e-3
    lr_rot: float = 3e-3
    kf_max: int = 10
    kf_min: int = 3
    cov_min: float = 0.85
    iters_per_kf: int = 40
    window: int = 8
    lambda_depth: float = 0.5
    refine_poses: bool = True
    align: str = "triangulation"
    init_min_px: float = 30.0
    init_max_frames: int = 60
    init_depth: float = 2.0
    # ограничения для трекинга (сдвиг в долях init_depth, повороты в градусах)
    max_vel_rot_deg: float = 6.0
    max_vel_trans: float = 0.075
    max_corr_rot_deg: float = 5.0
    max_corr_trans: float = 0.05
    # большую поправку отбрасываем, только если и ошибка трекера выше обычной
    # (больше reject_loss_ratio * медиана последних). Иначе после одной отбраковки
    # прогноз отстаёт и дальше отбрасывается всё подряд
    reject_loss_ratio: float = 1.5
    # насколько камера должна сдвинуться с прошлого ключевого кадра
    kf_motion_trans: float = 0.02
    kf_motion_rot_deg: float = 2.0
    depth_mode: str = "dav2"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    align_field: str = "affine"
    align_lam: float = 0.3


class GsSlam:
    def __init__(self, K, width, height, cfg, timers, depth_oracle=None):
        self.cfg = cfg
        self.K = K
        self.W, self.H = width, height
        self.timers = timers
        self.tri = SparseTriangulator()
        self.initializer = TwoViewInit(K, self.tri, min_px=cfg.init_min_px, target_depth=cfg.init_depth)
        self.mapper = Mapper(K, width, height, timers, self.tri, iters_per_kf=cfg.iters_per_kf,
                             window=cfg.window, lambda_depth=cfg.lambda_depth,
                             refine_poses=cfg.refine_poses, align=cfg.align,
                             depth_mode=cfg.depth_mode, depth_oracle=depth_oracle,
                             align_field=cfg.align_field, align_lam=cfg.align_lam,
                             depth_model=cfg.depth_model)
        self.depth_oracle = depth_oracle
        tracker_cls = DirectTracker if cfg.tracker == "direct" else RenderTracker
        self.tracker = tracker_cls(self.mapper.gmap, torch.as_tensor(K, dtype=torch.float32, device="cuda"),
                                     width, height, levels=cfg.track_levels,
                                     lr_trans=cfg.lr_trans, lr_rot=cfg.lr_rot, max_points=cfg.track_points)
        self.poses = {}
        self.kf_ids = []
        self.track_log = []
        self._pending = []   # кадры, пришедшие до инициализации
        self._first = None
        self.initialized = False
        self.init_failed = False

    def process(self, idx, image):
        if not self.initialized:
            return self._initialize(idx, image)
        return self._track(idx, image)

    def _initialize(self, idx, image):
        with self.timers("инициализация: ORB"):
            feats = self.tri.features(image)
        if self._first is None:
            self._first = (idx, image, feats)
            return None
        self._pending.append((idx, image))
        with self.timers("инициализация: две позы"):
            r = self.initializer.try_init(self._first[2], feats)
        if r is None:
            if len(self._pending) >= self.cfg.init_max_frames:
                # долго не получается - начинаем с более позднего кадра (в начале может не быть движения)
                self._first = (idx, image, feats)
                self._pending = []
            return None
        i0, img0, f0 = self._first
        eye = np.eye(4)
        if self.cfg.depth_mode == "scan_metric":
            # для проверки: сразу метрический масштаб по глубине скана в опорных точках
            from ..geometry.sparse import SparseDepth
            gd, gok = self.depth_oracle(i0)
            ui = np.clip(np.round(r.sparse_a.uv[:, 0]).astype(int), 0, self.W - 1)
            vi = np.clip(np.round(r.sparse_a.uv[:, 1]).astype(int), 0, self.H - 1)
            k = gok[vi, ui] & (gd[vi, ui] > 0.2)
            s = float(np.median(gd[vi, ui][k] / r.sparse_a.depth[k]))
            r.T_wc_b[:3, 3] *= s
            r.sparse_a = SparseDepth(r.sparse_a.uv, r.sparse_a.depth * s, r.sparse_a.reproj, r.sparse_a.parallax)
            r.sparse_b = SparseDepth(r.sparse_b.uv, r.sparse_b.depth * s, r.sparse_b.reproj, r.sparse_b.parallax)
        self.mapper.add_keyframe(i0, img0, eye, sparse=r.sparse_a, feats=f0)
        self.mapper.add_keyframe(idx, image, r.T_wc_b, sparse=r.sparse_b, feats=feats)
        if len(self.mapper.gmap) == 0:
            # глубина не выровнялась ни на одном кадре, карта пустая - пробуем заново с этого кадра
            self.mapper.kfs.clear()
            self.init_retries = getattr(self, "init_retries", 0) + 1
            self._first = (idx, image, feats)
            self._pending = []
            return None
        self.mapper.optimize(100, global_=True)
        self.tracker.refresh_map()
        self.poses[i0] = eye
        self.poses[idx] = self.mapper.pose_np(self.mapper.kfs[-1])
        self.kf_ids += [i0, idx]
        self.initialized = True
        self.init_info = {"first": i0, "second": idx, "inliers": r.n_inliers,
                          "median_px": r.median_px, "points": len(r.sparse_a.depth)}
        # кадры между двумя первыми трекаем от интерполированной позы.
        # с постоянной скоростью от единичной позы трекинг уезжал и был скачок на втором кадре
        T_b = self.poses[idx]
        n = idx - i0
        for j, img in self._pending[:-1]:
            res = self.tracker.track(torch.as_tensor(img, device="cuda"), _interp(eye, T_b, (j - i0) / n))
            self.poses[j] = res.T_wc
        self._pending = []
        self._last_kf = idx
        self._last = idx
        self._vel = np.linalg.inv(_interp(eye, T_b, (n - 1) / n)) @ T_b   # средняя скорость за кадр
        return self.poses[idx]

    def _accept(self, T_ref, T_init, res):
        # проверяем результат трекинга и обновляем скорость.
        # если поправка к прогнозу слишком большая - берём прогноз и уменьшаем скорость.
        # скорость тоже ограничена, иначе одна плохая поза её разгоняет
        from scipy.spatial.transform import Rotation
        c = self.cfg
        L = c.init_depth
        if not np.isfinite(res.T_wc).all():   # трекер разошёлся
            self._last_check = {"corr_rot": float("nan"), "corr_trans": float("nan"), "reject": "nan"}
            self.n_rejected = getattr(self, "n_rejected", 0) + 1
            self._vel = _interp(np.eye(4), np.linalg.inv(T_ref) @ T_init, 0.5)
            return T_init
        corr = np.linalg.inv(T_init) @ res.T_wc
        c_rot = float(np.degrees(Rotation.from_matrix(corr[:3, :3]).magnitude()))
        c_tr = float(np.linalg.norm(corr[:3, 3]) / L)
        reason = ("rot" if c_rot > c.max_corr_rot_deg else "trans" if c_tr > c.max_corr_trans
                  else "iters" if res.iters == 0 else "")
        hist = getattr(self, "_loss_hist", None)
        if hist is None:
            hist = self._loss_hist = []
        if (reason in ("rot", "trans") and c.reject_loss_ratio > 0 and len(hist) >= 5
                and res.loss <= c.reject_loss_ratio * float(np.median(hist[-30:]))):
            reason = ""   # ошибка нормальная, значит поправка настоящая
        bad = bool(reason)
        if not bad:
            hist.append(float(res.loss))
        self._last_check = {"corr_rot": c_rot, "corr_trans": c_tr, "reject": reason}
        T_new = T_init if bad else res.T_wc
        vel = np.linalg.inv(T_ref) @ T_new
        if bad:
            vel = _interp(np.eye(4), vel, 0.5)
            self.n_rejected = getattr(self, "n_rejected", 0) + 1
        ang = np.degrees(Rotation.from_matrix(vel[:3, :3]).magnitude())
        tr = np.linalg.norm(vel[:3, 3])
        k = min(1.0, c.max_vel_rot_deg / max(ang, 1e-9), c.max_vel_trans * L / max(tr, 1e-12))
        self._vel = vel if k >= 1.0 else _interp(np.eye(4), vel, k)
        return T_new

    def _want_kf(self, since, coverage, T_new):
        from scipy.spatial.transform import Rotation
        c = self.cfg
        if not (since >= c.kf_max or (since >= c.kf_min and coverage < c.cov_min)):
            return False
        rel = np.linalg.inv(self.poses[self._last_kf]) @ T_new
        return bool(np.linalg.norm(rel[:3, 3]) > c.kf_motion_trans * c.init_depth
                    or np.degrees(Rotation.from_matrix(rel[:3, :3]).magnitude()) > c.kf_motion_rot_deg)

    def _track(self, idx, image):
        # скорость считаем только по трекингу: уточнение поз в маппинге сдвигает позу,
        # и эта разница - не движение камеры (раньше из-за этого был перелёт)
        T_ref = self.poses[self._last]
        T_init = T_ref @ self._vel
        img_t = torch.as_tensor(image, device="cuda")
        with self.timers("трекинг (кадр)"):
            res = self.tracker.track(img_t, T_init)
        T_new = self._accept(T_ref, T_init, res)
        self.poses[idx] = T_new
        self._last = idx
        since = idx - self._last_kf
        is_kf = self._want_kf(since, res.coverage, T_new)
        self.track_log.append({"frame": idx, "loss": res.loss, "coverage": res.coverage,
                               "exposure": res.exposure, "kf": bool(is_kf), **self._last_check})
        if is_kf:
            with self.timers("маппинг (ключевой кадр)"):
                self.mapper.add_keyframe(idx, image, T_new)
                self.tracker.refresh_map()
                # уточнённые позы ключевых кадров вместо поз от трекера
                for kf in self.mapper.kfs[-self.cfg.window:]:
                    self.poses[kf["id"]] = self.mapper.pose_np(kf)
            self.kf_ids.append(idx)
            self._last_kf = idx
        return self.poses[idx]

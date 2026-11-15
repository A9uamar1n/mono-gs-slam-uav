# маппинг по ключевым кадрам. На каждый новый ключевой кадр:
#  1) глубина от Depth Anything V2 (обратная, без масштаба)
#  2) выравнивание масштаба и сдвига: align="map" - по глубине, отрендеренной из
#     текущей карты (если карта покрывает кадр), иначе по триангулированным точкам;
#     align="triangulation" - наоборот (база между ключевыми кадрами короткая, масштаб шумит)
#  3) добавляем гауссианы там, где карты нет или она дальше новой глубины
#  4) оптимизируем карту по окну ключевых кадров (цвет + глубина с уверенностью)
# для проверок depth_mode может брать глубину из скана: "scan_aligned" - форма идеальная,
# выравнивание как у сети; "scan_metric" - сразу метрическая.
# refine_poses: позы ключевых кадров в окне тоже уточняются (T_wc * Exp(xi), свой Adam),
# первый ключевой кадр закреплён
import random

import cv2
import numpy as np
import torch

from ..depth.align import (apply_alignment, apply_field_alignment, confidence_map, fit_scale_shift,
                           fit_scale_shift_field)
from ..depth.mono import MonoDepth
from ..geometry.sparse import SparseDepth, SparseTriangulator
from ..map.gaussians import GaussianMap
from ..map.losses import _gauss_window, color_loss, depth_loss, smooth_loss
from ..track.se3 import se3_exp


def _t(x, dev):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


class Mapper:
    def __init__(self, K, width, height, timers, tri, iters_per_kf=40, window=8, stride=2,
                 lambda_ssim=0.2, lambda_depth=0.5, lambda_smooth=0.0, n_neighbours=3,
                 prune_every=5, refine_poses=True, lr_pose=5e-4, align="triangulation",
                 depth_mode="dav2", depth_oracle=None, align_field="affine", align_lam=0.3,
                 depth_model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda"):
        self.dev = torch.device(device)
        self.K_np = K
        self.K = _t(K, self.dev)
        self.W, self.H = width, height
        self.timers = timers
        self.tri = tri
        self.iters_per_kf = iters_per_kf
        self.window = window
        self.stride = stride
        self.lambda_ssim, self.lambda_depth, self.lambda_smooth = lambda_ssim, lambda_depth, lambda_smooth
        self.n_neighbours = n_neighbours
        self.prune_every = prune_every
        self.refine_poses = refine_poses
        self.lr_pose = lr_pose
        self.align = align
        self.depth_mode = depth_mode
        self.align_field = align_field
        self.align_lam = align_lam
        self.depth_oracle = depth_oracle
        self.gmap = GaussianMap(device)
        self.mono = MonoDepth(depth_model, device=device)
        self.win = _gauss_window(device=self.dev)
        self.kfs = []
        self.align_log = []

    def _align_to_map(self, pred_np, T_wc, min_cov=0.0):
        # опорные точки - глубина из текущей карты там, где alpha > 0.95
        if len(self.gmap) == 0:
            return None
        with torch.no_grad():
            _, d, alpha, _ = self.gmap.render(self.K, T_wc, self.W, self.H)
        d, alpha = d.cpu().numpy(), alpha.cpu().numpy()
        v, u = np.mgrid[2:self.H:4, 2:self.W:4]
        ok = (alpha[v, u] > 0.95) & (d[v, u] > 0.05)
        if ok.sum() < max(100, min_cov * ok.size):
            return None
        uv = np.stack([u[ok], v[ok]], 1).astype(np.float64)
        return self._fit(pred_np, uv, d[v, u][ok].astype(np.float64))

    def _fit(self, pred_np, uv, depth):
        if self.align_field == "affine":
            return fit_scale_shift(pred_np, uv, depth)
        return fit_scale_shift_field(pred_np, uv, depth, self.align_field, self.align_lam)

    def add_keyframe(self, frame_id, image, T_wc, sparse=None, feats=None):
        tm = self.timers
        img_t = _t(image, self.dev)
        T_t = _t(T_wc, self.dev)
        if feats is None:
            with tm("маппинг: ORB"):
                feats = self.tri.features(image)
        pred_ok = None
        if self.depth_mode == "dav2":
            with tm("маппинг: глубина (сеть)"):
                pred = self.mono(image)
            pred_np = pred.cpu().numpy()
        else:
            gd, pred_ok = self.depth_oracle(frame_id)
            pred_ok = pred_ok & (gd > 0.2)
            pred_np = np.where(pred_ok, 1.0 / np.maximum(gd, 0.2), 0.0).astype(np.float32)
        with tm("маппинг: опорные точки"):
            if sparse is None:
                neigh = [(k["feats"], k["T_np"]) for k in self.kfs[-self.n_neighbours:]]
                sparse = self.tri.sparse_depth(feats, self.K_np, T_wc, neigh) if neigh else None
            if sparse is not None and pred_ok is not None:   # только где есть скан
                ui = np.clip(np.round(sparse.uv[:, 0]).astype(int), 0, self.W - 1)
                vi = np.clip(np.round(sparse.uv[:, 1]).astype(int), 0, self.H - 1)
                k = pred_ok[vi, ui]
                sparse = SparseDepth(sparse.uv[k], sparse.depth[k], sparse.reproj[k], sparse.parallax[k])
        with tm("маппинг: выравнивание"):
            tri_fit = lambda: self._fit(pred_np, sparse.uv, sparse.depth) if sparse is not None else None  # noqa: E731
            if self.align == "map" and len(self.kfs) > 0:
                source, a = "map", self._align_to_map(pred_np, T_t, min_cov=0.3)
                if a is None or not a.ok:
                    source, a = "triangulation", tri_fit()
            else:
                source, a = "triangulation", tri_fit()
                if a is None or not a.ok:
                    source, a = "map", self._align_to_map(pred_np, T_t)
            if self.depth_mode == "scan_metric":
                source, a = "scan", None
                d_star = np.where(pred_ok, gd, 0.0).astype(np.float32)
                valid = pred_ok
                conf = confidence_map(image[:, :, 0], d_star, valid)
            elif a is not None and a.ok:
                d_star, valid = apply_field_alignment(pred_np, a)
                if pred_ok is not None:
                    valid = valid & pred_ok
                    d_star = np.where(valid, d_star, 0.0).astype(np.float32)
                conf = confidence_map(image[:, :, 0], d_star, valid)
            else:
                source = "none"
                d_star = np.zeros((self.H, self.W), np.float32)
                valid = np.zeros((self.H, self.W), bool)
                conf = np.zeros((self.H, self.W), np.float32)
        self.align_log.append({"frame": frame_id, "source": source,
                               "n_points": 0 if sparse is None else len(sparse.depth),
                               "rel_err": None if a is None else a.rel_err})
        g8 = (image[:, :, 0] * 255).astype(np.uint8)
        edges = cv2.dilate(cv2.Canny(g8, 50, 150), np.ones((3, 3), np.uint8))
        kf = {"id": frame_id, "image": img_t, "T_wc": T_t, "T_np": T_wc.copy(), "feats": feats,
              "depth": _t(d_star, self.dev), "valid": torch.as_tensor(valid, device=self.dev),
              "conf": _t(conf, self.dev), "non_edge": _t((edges == 0).astype(np.float32), self.dev),
              "ok": source != "none"}
        if self.refine_poses and self.kfs:   # у первого кадра позу не трогаем
            kf["xi"] = torch.zeros(6, device=self.dev, requires_grad=True)
            kf["popt"] = torch.optim.Adam([kf["xi"]], lr=self.lr_pose)
        self.kfs.append(kf)

        with tm("маппинг: добавление гауссиан"):
            if kf["ok"]:
                if len(self.gmap) == 0:
                    mask = kf["valid"]
                else:
                    with torch.no_grad():
                        _, d, alpha, _ = self.gmap.render(self.K, T_t, self.W, self.H)
                    mask = kf["valid"] & ((alpha < 0.5) | (d > 1.2 * kf["depth"]))
                self.gmap.add_from_depth(img_t, kf["depth"], mask, self.K, T_t, self.stride)
        with tm("маппинг: оптимизация"):
            self.optimize(self.iters_per_kf)
        if len(self.kfs) % self.prune_every == 0:
            self.gmap.prune()
        return kf

    def pose(self, kf):
        if "xi" not in kf:
            return kf["T_wc"]
        return kf["T_wc"] @ se3_exp(kf["xi"])

    def pose_np(self, kf):
        with torch.no_grad():
            return self.pose(kf).double().cpu().numpy()

    def _step(self, kf, refine):
        refine = refine and "xi" in kf
        if refine:
            T = self.pose(kf)
        else:
            with torch.no_grad():
                T = self.pose(kf)
        rgb, d, alpha, _ = self.gmap.render(self.K, T, self.W, self.H)
        # каналы одинаковые (серые картинки), считаем по одному
        loss = color_loss(rgb[..., :1], kf["image"][..., :1], self.lambda_ssim, self.win)
        if self.lambda_depth > 0 and kf["ok"]:
            loss = loss + self.lambda_depth * depth_loss(
                d, kf["depth"], kf["conf"], kf["valid"] & (alpha.detach() > 0.3))
        if self.lambda_smooth > 0:
            loss = loss + self.lambda_smooth * smooth_loss(d, kf["non_edge"])
        self.gmap.opt.zero_grad(set_to_none=True)
        if refine:
            kf["popt"].zero_grad(set_to_none=True)
        loss.backward()
        self.gmap.opt.step()
        if refine:
            kf["popt"].step()

    def optimize(self, iters, global_=False):
        if len(self.gmap) == 0 or not self.kfs:
            return
        n = len(self.kfs) - 1
        lo = 0 if global_ else max(0, n - self.window + 1)
        for _ in range(iters):
            # 70% итераций по окну, остальное по старым кадрам
            if global_ or lo == 0 or random.random() < 0.7:
                j = random.randint(lo, n)
            else:
                j = random.randint(0, lo - 1)
            self._step(self.kfs[j], refine=self.refine_poses and j >= lo)

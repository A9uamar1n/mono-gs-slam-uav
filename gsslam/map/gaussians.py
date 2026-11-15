# карта 3D-гауссиан: параметры, добавление по глубине, рендер, удаление.
# у гауссианы центр, log масштабов, кватернион, логит прозрачности и логит цвета.
# сферических гармоник нет, картинки EuRoC серые
import torch
from gsplat import rasterization

_LR = {"means": 1.0e-3, "log_scales": 5.0e-3, "quats": 1.0e-3, "opac": 5.0e-2, "colors": 5.0e-3}


class GaussianMap:
    def __init__(self, device="cuda", lr_scale=1.0):
        self.device = torch.device(device)
        z = lambda *s: torch.zeros(*s, device=self.device)  # noqa: E731
        self.params = {
            "means": torch.nn.Parameter(z(0, 3)),
            "log_scales": torch.nn.Parameter(z(0, 3)),
            "quats": torch.nn.Parameter(z(0, 4)),
            "opac": torch.nn.Parameter(z(0)),
            "colors": torch.nn.Parameter(z(0, 3)),
        }
        self.opt = torch.optim.Adam(
            [{"params": [p], "lr": _LR[k] * lr_scale, "name": k} for k, p in self.params.items()],
            eps=1e-15)

    @classmethod
    def load(cls, params, device="cuda"):
        g = cls(device)
        g._replace({k: params[k].to(g.device).float() for k in g.params})
        return g

    def __len__(self):
        return self.params["means"].shape[0]

    def _replace(self, new, keep=None):
        # подменяем тензоры параметров и не теряем состояние Adam.
        # keep - маска оставшихся (при удалении), None - значит добавили в конец
        for group in self.opt.param_groups:
            name = group["name"]
            old = group["params"][0]
            state = self.opt.state.pop(old, None)
            p = torch.nn.Parameter(new[name])
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    buf = state[key]
                    if keep is not None:
                        buf = buf[keep]
                    else:
                        extra = p.shape[0] - buf.shape[0]
                        buf = torch.cat([buf, torch.zeros(extra, *buf.shape[1:], device=buf.device)])
                    state[key] = buf
                self.opt.state[p] = state
            group["params"] = [p]
            self.params[name] = p

    def add(self, means, scales, colors, opacity=0.5):
        # изотропные гауссианы: means (N,3), scales (N,), colors (N,3)
        n = means.shape[0]
        if n == 0:
            return 0
        quats = torch.zeros(n, 4, device=self.device)
        quats[:, 0] = 1.0
        col = colors.clamp(1e-3, 1 - 1e-3)
        fresh = {
            "means": means,
            "log_scales": torch.log(scales.clamp(min=1e-5))[:, None].repeat(1, 3),
            "quats": quats,
            "opac": torch.full((n,), float(torch.logit(torch.tensor(opacity))), device=self.device),
            "colors": torch.logit(col),
        }
        self._replace({k: torch.cat([self.params[k].detach(), fresh[k].to(self.device)])
                       for k in self.params})
        return n

    @torch.no_grad()
    def add_from_depth(self, image, depth, mask, K, T_wc, stride=2, scale_factor=1.0):
        # обратная проекция каждого stride-го пикселя из mask по карте глубины.
        # размер гауссианы = след пикселя на этой глубине (depth * stride / f)
        h, w = depth.shape
        vs = torch.arange(stride // 2, h, stride, device=self.device)
        us = torch.arange(stride // 2, w, stride, device=self.device)
        v, u = torch.meshgrid(vs, us, indexing="ij")
        sel = mask[v, u] & (depth[v, u] > 0)
        u, v = u[sel].float(), v[sel].float()
        d = depth[v.long(), u.long()]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = torch.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], dim=1)
        pw = pc @ T_wc[:3, :3].T + T_wc[:3, 3]
        scales = d * stride / (0.5 * (fx + fy)) * scale_factor
        return self.add(pw, scales, image[v.long(), u.long()])

    @torch.no_grad()
    def prune(self, min_opacity=0.05, max_scale=None):
        keep = torch.sigmoid(self.params["opac"]) > min_opacity
        if max_scale is not None:
            keep &= self.params["log_scales"].exp().max(dim=1).values < max_scale
        removed = int((~keep).sum())
        if removed:
            self._replace({k: p.detach()[keep] for k, p in self.params.items()}, keep=keep)
        return removed

    def render(self, K, T_wc, width, height, near=0.05, far=100.0, quats=None, log_scales=None):
        # рендер из позы T_wc -> rgb (H,W,3), глубина (H,W), alpha (H,W), meta.
        # глубина ожидаемая (RGB+ED в gsplat), т.е. уже поделена на накопленную alpha
        p = self.params
        q = p["quats"] if quats is None else quats   # quats/log_scales подменяет Deblur MLP
        ls = p["log_scales"] if log_scales is None else log_scales
        viewmat = torch.linalg.inv(T_wc)[None]
        out, alpha, meta = rasterization(
            p["means"], q, ls.exp(), torch.sigmoid(p["opac"]),
            torch.sigmoid(p["colors"]), viewmat, K[None], width, height,
            near_plane=near, far_plane=far, render_mode="RGB+ED", packed=False)
        return out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0], meta

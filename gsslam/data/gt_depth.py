# эталонная глубина из лазерного скана (есть только у V1/V2), нужна для оценки геометрии
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_scan(seq_root):
    # точки скана (N, 3) или None; после первого чтения кладём в .npy, ply читается долго
    ply = Path(seq_root) / "mav0" / "pointcloud0" / "data.ply"
    if not ply.is_file():
        return None
    cache = ply.with_suffix(".xyz.npy")
    if cache.is_file():
        return np.load(cache)
    import pandas as pd

    with open(ply, "rb") as f:
        skip = 0
        for line in f:
            skip += 1
            if line.strip() == b"end_header":
                break
    df = pd.read_csv(ply, sep=r"\s+", header=None, skiprows=skip, usecols=[0, 1, 2], dtype=np.float32)
    pts = df.to_numpy(dtype=np.float32)
    np.save(cache, pts)
    return pts


class ScanDepth:
    def __init__(self, points, device="cuda"):
        self.pts = torch.as_tensor(points, device=device)

    @torch.no_grad()
    def render(self, K, T_wc, width, height, near=0.2, far=15.0):
        # проекция скана в кадр с z-буфером, возвращает глубину и маску
        T_cw = torch.linalg.inv(T_wc)
        pc = self.pts @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = pc[:, 2]
        u = (pc[:, 0] / z * K[0, 0] + K[0, 2]).round().long()
        v = (pc[:, 1] / z * K[1, 1] + K[1, 2]).round().long()
        ok = (z > near) & (z < far) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        idx = v[ok] * width + u[ok]
        depth = torch.full((height * width,), float("inf"), device=self.pts.device)
        depth.scatter_reduce_(0, idx, z[ok], reduce="amin")
        depth = depth.view(height, width)
        # дырки в один пиксель закрываем минимумом по окну 3x3
        filled = -F.max_pool2d(-depth[None, None], 3, stride=1, padding=1)[0, 0]
        valid = torch.isfinite(filled)
        return torch.where(valid, filled, torch.zeros_like(filled)), valid

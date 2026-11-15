# компенсация смаза (идея из Deblurring 3DGS): MLP по гауссиане и коду кадра выдаёт
# поправки к повороту и масштабу, смазанный кадр рендерится из изменённых гауссиан.
# для трекинга и оценки MLP выключается. В итоговую систему не вошло (стало хуже).
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurMLP(nn.Module):
    def __init__(self, n_frames, n_freq=4, embed_dim=16, hidden=64,
                 max_log_scale=0.7, max_dq=0.2):
        super().__init__()
        self.n_freq = n_freq
        # смаз в EuRoC от движения камеры, у каждого кадра свой, поэтому код кадра
        self.embed = nn.Embedding(n_frames, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        d_in = 3 + 3 * 2 * n_freq + 4 + 3 + embed_dim
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 7))
        # нули на выходе, чтобы сначала поправок не было
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_log_scale = max_log_scale   # exp(0.7) ~ в 2 раза максимум
        self.max_dq = max_dq
        self.center = None
        self.radius = None

    def set_bounds(self, means):
        with torch.no_grad():
            self.center = means.median(dim=0).values
            self.radius = (means - self.center).abs().quantile(0.95).clamp(min=1e-3)

    def forward(self, means, quats, log_scales, frame):
        x = ((means.detach() - self.center) / self.radius).clamp(-2, 2)
        freqs = (2.0 ** torch.arange(self.n_freq, device=x.device)) * math.pi
        xf = x[:, :, None] * freqs
        enc = torch.cat([x, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], 1)
        q = F.normalize(quats.detach(), dim=1)
        e = self.embed.weight[frame].expand(len(x), -1)
        o = self.net(torch.cat([enc, q, log_scales.detach(), e], 1))
        # масштаб только увеличиваем: если разрешить сжатие, гауссианы раздуваются,
        # MLP их сжимает обратно, и рендер без MLP портится (было 20,8 -> 15,8 дБ).
        # -4 чтобы в начале поправка была почти ноль, но градиент был
        dls = torch.sigmoid(o[:, :3] - 4.0) * self.max_log_scale
        dq = torch.tanh(o[:, 3:]) * self.max_dq
        return F.normalize(quats + dq, dim=1), log_scales + dls, dls

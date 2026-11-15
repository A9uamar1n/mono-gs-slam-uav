# экспонента SE(3) на torch, нужна для уточнения поз ключевых кадров в маппинге.
# ряды Тейлора по theta^2, чтобы градиент в нуле был нормальный (у нормы его нет)
import torch


def skew(v):
    z = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack([
        torch.stack([z, -v[2], v[1]]),
        torch.stack([v[2], z, -v[0]]),
        torch.stack([-v[1], v[0], z]),
    ])


def se3_exp(xi):
    # xi = (rho, phi): сдвиг и вектор поворота -> матрица 4x4
    rho, phi = xi[:3], xi[3:]
    t2 = (phi * phi).sum()
    t4 = t2 * t2
    a = 1 - t2 / 6 + t4 / 120
    b = 0.5 - t2 / 24 + t4 / 720
    c = 1.0 / 6 - t2 / 120 + t4 / 5040
    P = skew(phi)
    P2 = P @ P
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    R = eye + a * P + b * P2
    V = eye + b * P + c * P2
    top = torch.cat([R, (V @ rho)[:, None]], dim=1)
    bottom = torch.tensor([[0, 0, 0, 1]], dtype=xi.dtype, device=xi.device)
    return torch.cat([top, bottom], dim=0)

# функция потерь маппинга (из отчёта за 6 семестр):
# L = L_color + lambda_depth * L_depth + lambda_smooth * L_smooth
# L_color = (1 - lambda_ssim) * L1 + lambda_ssim * (1 - SSIM)
import torch
import torch.nn.functional as F


def _gauss_window(size=11, sigma=1.5, device=None):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b, window=None):
    # SSIM для (H, W, C), среднее по картинке.
    # окно гауссово и раздельное, поэтому две свёртки 11x1 и 1x11 (так быстрее).
    # window остался от старой версии, берётся только его размер
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    c = a.shape[1]
    size = 11 if window is None else window.shape[-1]
    x = torch.arange(size, dtype=a.dtype, device=a.device) - size // 2
    g = torch.exp(-x ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    gh = g.view(1, 1, 1, size).expand(c, 1, 1, size)
    gv = g.view(1, 1, size, 1).expand(c, 1, size, 1)
    p = size // 2

    def blur(t):
        return F.conv2d(F.conv2d(t, gh, padding=(0, p), groups=c), gv, padding=(p, 0), groups=c)

    mu_a, mu_b = blur(a), blur(b)
    va = blur(a * a) - mu_a ** 2
    vb = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean()


def color_loss(render, target, lambda_ssim=0.2, window=None):
    l1 = (render - target).abs().mean()
    if lambda_ssim <= 0:
        return l1
    return (1 - lambda_ssim) * l1 + lambda_ssim * (1 - ssim(render, target, window))


def depth_loss(d_render, d_target, conf, valid):
    # L1 по глубине с весами уверенности, только по валидным пикселям
    n = valid.sum().clamp(min=1)
    return (conf * (d_render - d_target).abs() * valid).sum() / n


def smooth_loss(d_render, non_edge):
    # штраф за разность глубин соседей там, где нет границы на картинке.
    # глубину делим на среднее, чтобы не зависело от масштаба сцены
    d = d_render / d_render.detach().mean().clamp(min=1e-6)
    dx = (d[:, 1:] - d[:, :-1]) ** 2 * non_edge[:, 1:] * non_edge[:, :-1]
    dy = (d[1:, :] - d[:-1, :]) ** 2 * non_edge[1:, :] * non_edge[:-1, :]
    return dx.mean() + dy.mean()


def psnr(a, b):
    mse = ((a - b) ** 2).mean().clamp(min=1e-12)
    return float(-10 * torch.log10(mse))

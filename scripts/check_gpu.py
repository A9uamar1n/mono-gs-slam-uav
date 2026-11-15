# проверка, что CUDA и gsplat работают: рендер и backward на случайных гауссианах + время
#   python scripts/check_gpu.py [--n 200000] [--width 376] [--height 240]
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsslam.utils.timing import Timers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="число гауссиан")
    ap.add_argument("--width", type=int, default=376)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()

    import gsplat
    from gsplat import rasterization

    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, gsplat {gsplat.__version__}")
    if not torch.cuda.is_available():
        sys.exit("CUDA недоступна")
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} ГБ")

    g = torch.Generator(device="cpu").manual_seed(0)
    n = args.n
    means = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([8.0, 6.0, 6.0]) + torch.tensor([0, 0, 5.0])
    means = means.to(dev).requires_grad_(True)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1).to(dev).requires_grad_(True)
    scales = (torch.rand(n, 3, generator=g) * 0.03 + 0.005).to(dev).requires_grad_(True)
    opac = torch.rand(n, generator=g).to(dev).requires_grad_(True)
    colors = torch.rand(n, 3, generator=g).to(dev).requires_grad_(True)

    W, H = args.width, args.height
    f = 0.6 * W
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev)[None]
    view = torch.eye(4, device=dev)[None]
    target = torch.rand(1, H, W, 3, device=dev)

    timers = Timers()
    for _ in range(args.iters):
        with timers("рендер (вперёд)"):
            rgbd, alpha, _ = rasterization(means, quats, scales, opac, colors, view, K, W, H,
                                           render_mode="RGB+ED")
        with timers("обратный проход"):
            loss = (rgbd[..., :3] - target).abs().mean() + 0.1 * rgbd[..., 3].mean()
            loss.backward()
        for p in (means, quats, scales, opac, colors):
            p.grad = None

    assert rgbd.shape == (1, H, W, 4), rgbd.shape
    assert torch.isfinite(rgbd).all()
    print(f"\n{n} гауссиан, {W}x{H}, RGB + глубина")
    print(timers.table(skip=10))
    print(f"пик видеопамяти: {torch.cuda.max_memory_allocated() / 2**20:.0f} МБ")


if __name__ == "__main__":
    main()

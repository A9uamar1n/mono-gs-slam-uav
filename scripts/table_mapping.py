# таблица по запускам маппинга (PSNR, SSIM, ошибки глубины)
#   python scripts/table_mapping.py outputs/mapping [префикс]
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else ""
rows = []
for f in sorted(root.glob(f"{prefix}*/metrics.json")):
    m = json.loads(f.read_text(encoding="utf-8"))
    rows.append((f.parent.name, m))
print(f"{'запуск':<44}{'PSNR':>7}{'SSIM':>7}{'глуб.ср%':>10}{'глуб.мед%':>11}{'<25%':>7}{'гаусс.':>9}{'скан%':>8}{'скан.мед%':>11}{'скан.гладк%':>13}{'скан<25%':>10}")
for name, m in rows:
    g = lambda k, mul=100: (f"{m[k] * mul:.1f}" if m.get(k) is not None else "-")  # noqa: E731
    print(f"{name:<44}{m['psnr']:>7.2f}{m['ssim']:>7.3f}{g('depth_abs_rel'):>10}{g('depth_med_rel'):>11}"
          f"{g('depth_delta1'):>7}{m['n_gaussians']:>9}{g('scan_abs_rel'):>8}{g('scan_med_rel'):>11}"
          f"{g('scan_abs_rel_flat'):>13}{g('scan_delta1'):>10}")

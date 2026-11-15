# таймеры по модулям для бюджета времени
#   timers = Timers()
#   with timers("tracking"): ...
#   print(timers.table())
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _sync():
    # синхронизируем только свой CUDA-поток: в асинхронном режиме у маппинга свой поток,
    # и если ждать всё устройство, время маппинга попадёт в трекинг
    if torch is not None and torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


class Timers:
    def __init__(self, sync_cuda=True):
        self.sync_cuda = sync_cuda
        self.samples = defaultdict(list)

    @contextmanager
    def __call__(self, name):
        if self.sync_cuda:
            _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                _sync()
            self.samples[name].append((time.perf_counter() - t0) * 1e3)

    def stats(self, name, skip=0):
        # в мс, skip - сколько первых замеров выкинуть (прогрев)
        x = np.asarray(self.samples[name][skip:], dtype=np.float64)
        if x.size == 0:
            return {"n": 0, "mean": float("nan"), "median": float("nan"),
                    "p95": float("nan"), "max": float("nan")}
        return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
                "p95": float(np.percentile(x, 95)), "max": float(x.max())}

    def table(self, skip=0, budget_ms=None):
        head = f"{'модуль':<22}{'n':>7}{'среднее':>10}{'медиана':>10}{'p95':>10}{'макс':>10}  мс"
        lines = [head, "-" * len(head)]
        for name in self.samples:
            s = self.stats(name, skip)
            mark = ""
            if budget_ms is not None and s["n"]:
                mark = "  в бюджете" if s["p95"] <= budget_ms else "  ПРЕВЫШЕН"
            lines.append(f"{name:<22}{s['n']:>7}{s['mean']:>10.2f}{s['median']:>10.2f}"
                         f"{s['p95']:>10.2f}{s['max']:>10.2f}{mark}")
        return "\n".join(lines)

# асинхронный вариант: трекинг в основном потоке, маппинг в фоновом.
#  - у трекера и маппинга свои CUDA-потоки
#  - маппинг вставляет ключевой кадр, а в остальное время крутит оптимизацию окна
#  - после каждой порции итераций маппинг выкладывает копию карты и поз окна,
#    трекер забирает её перед следующим кадром (с живой картой трекер не работает)
#  - если маппинг занят, ключевой кадр не принимается
# GPU один, но трекер много времени ждёт запуска ядер (WDDM), маппинг занимает эти паузы
import queue
import sys
import threading
import time

import numpy as np
import torch

from .system import GsSlam


class AsyncGsSlam(GsSlam):
    def __init__(self, *args, publish_every=10, idle_sleep_ms=0.0,
                 switch_interval_ms=0.5, idle_duty=1.0, **kw):
        super().__init__(*args, **kw)
        # по умолчанию поток держит GIL до 5 мс, и трекер после каждой синхронизации
        # с GPU ждёт поток маппинга. Уменьшаем интервал переключения
        self._old_switch = sys.getswitchinterval()
        sys.setswitchinterval(switch_interval_ms / 1000.0)
        self.publish_every = publish_every
        # какую долю времени маппинг оптимизирует между ключевыми кадрами (1 - без пауз)
        self.idle_duty = idle_duty
        self.idle_sleep = idle_sleep_ms / 1000.0
        self.lock = threading.Lock()
        self.kf_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.inserting = False
        self.trk_stream = torch.cuda.Stream(priority=-1)   # у трекера приоритет выше
        self.map_stream = torch.cuda.Stream()
        self._snap = None
        self._snap_poses = {}
        self._snap_new = False
        self.thread = None
        self.stats = {"kf_requested": 0, "kf_rejected": 0, "idle_iters": 0, "publishes": 0,
                      "insert_ms": [], "snap_age_frames": []}
        self._snap_frame = 0
        self._frame = 0
        self.error = None

    def process(self, idx, image):
        self._frame = idx
        if not self.initialized:
            T = self._initialize(idx, image)   # инициализация синхронно, поток ещё не запущен
            if self.initialized:
                torch.cuda.synchronize()
                self._publish()
                self.thread = threading.Thread(target=self._map_loop, name="mapping", daemon=True)
                self.thread.start()
            return T
        if self.error is not None:
            raise RuntimeError("поток маппинга упал") from self.error
        with torch.cuda.stream(self.trk_stream):
            return self._track_async(idx, image)

    def finish(self):
        # остановить маппинг и забрать итоговые позы ключевых кадров
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        sys.setswitchinterval(self._old_switch)
        torch.cuda.synchronize()
        for kf in self.mapper.kfs:
            self.poses[kf["id"]] = self.mapper.pose_np(kf)
        if self.error is not None:
            raise RuntimeError("поток маппинга упал") from self.error

    def _swap_snapshot(self):
        with self.lock:
            if not self._snap_new:
                return
            snap, poses, frame = self._snap, self._snap_poses, self._snap_frame
            self._snap_new = False
        stream = torch.cuda.current_stream()
        for x in snap:
            x.record_stream(stream)   # чтобы память не освободилась, пока трекер читает
        self.tracker._frozen = snap
        self.poses.update(poses)
        self.stats["snap_age_frames"].append(self._frame - frame)

    def _track_async(self, idx, image):
        self._swap_snapshot()
        T_ref = self.poses[self._last]
        T_init = T_ref @ self._vel
        img_t = torch.as_tensor(image, device="cuda")
        with self.timers("трекинг (кадр)"):
            res = self.tracker.track(img_t, T_init)
        T_new = self._accept(T_ref, T_init, res)
        self.poses[idx] = T_new
        self._last = idx
        since = idx - self._last_kf
        want = self._want_kf(since, res.coverage, T_new)
        accepted = False
        if want:
            self.stats["kf_requested"] += 1
            if not self.inserting and self.kf_queue.empty():
                self.inserting = True
                self.kf_queue.put((idx, image, T_new.copy()))
                self.kf_ids.append(idx)
                self._last_kf = idx
                accepted = True
            else:
                self.stats["kf_rejected"] += 1
        self.track_log.append({"frame": idx, "loss": res.loss, "coverage": res.coverage,
                               "exposure": res.exposure, "kf": accepted, **self._last_check})
        return T_new

    def _publish(self):
        p = self.mapper.gmap.params
        snap = (p["means"].detach().clone(), p["quats"].detach().clone(), p["log_scales"].detach().exp(),
                torch.sigmoid(p["opac"].detach()), torch.sigmoid(p["colors"].detach()))
        poses = {kf["id"]: self.mapper.pose_np(kf) for kf in self.mapper.kfs[-self.cfg.window:]}
        torch.cuda.current_stream().synchronize()   # копия должна быть готова до трекера
        with self.lock:
            self._snap, self._snap_poses, self._snap_frame = snap, poses, self._frame
            self._snap_new = True
        self.stats["publishes"] += 1

    def _map_loop(self):
        try:
            with torch.cuda.stream(self.map_stream):
                while not self.stop_event.is_set():
                    try:
                        idx, image, T = self.kf_queue.get_nowait()
                    except queue.Empty:
                        idx = None
                    if idx is not None:
                        t0 = time.perf_counter()
                        self.mapper.add_keyframe(idx, image, T)
                        self._publish()
                        self.stats["insert_ms"].append((time.perf_counter() - t0) * 1e3)
                        self.inserting = False
                    elif self.idle_duty <= 0:
                        time.sleep(0.002)
                    else:
                        t0 = time.perf_counter()
                        self.mapper.optimize(self.publish_every)
                        self.stats["idle_iters"] += self.publish_every
                        self._publish()
                        busy = time.perf_counter() - t0
                        pause = busy * (1.0 / self.idle_duty - 1.0) + self.idle_sleep
                        if pause > 0:
                            time.sleep(pause)
        except BaseException as e:   # noqa: BLE001  ошибку покажем в основном потоке
            self.error = e

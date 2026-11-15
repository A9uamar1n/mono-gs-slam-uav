# загрузка последовательностей EuRoC MAV (формат ASL), левая камера cam0.
# кадры без дисторсии, матрица K и эталонная поза T_wc на момент кадра
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Frame:
    index: int
    timestamp: float          # с
    image: np.ndarray         # (H, W, 3) в [0, 1], EuRoC чёрно-белый, каналы одинаковые
    K: np.ndarray             # (3, 3) уже после undistort и масштабирования
    T_wc: np.ndarray | None   # эталонная поза или None, если эталона нет


def _read_sensor_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class EurocSequence:
    # root - папка с mav0/, scale - уменьшение кадра (0.5 -> 376x240), stride - шаг по кадрам
    def __init__(self, root, scale=1.0, stride=1, start=0, end=None):
        self.root = Path(root)
        mav = self.root / "mav0"
        if not mav.is_dir():
            raise FileNotFoundError(f"не найдена папка {mav}")
        cam_dir = mav / "cam0"

        sensor = _read_sensor_yaml(cam_dir / "sensor.yaml")
        fu, fv, cu, cv_ = sensor["intrinsics"]
        self.raw_size = tuple(sensor["resolution"])   # (W, H)
        self.K_raw = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(sensor["distortion_coefficients"], dtype=np.float64)
        self.T_bc = np.array(sensor["T_BS"]["data"], dtype=np.float64).reshape(4, 4)  # камера -> тело
        self.rate_hz = float(sensor.get("rate_hz", 20))

        # undistort с той же K. Дисторсия бочкообразная, так что после исправления
        # весь кадр валидный (проверка в valid_fraction)
        w, h = self.raw_size
        K_new = self.K_raw.copy()
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K_raw, self.dist, None, K_new, (w, h), cv2.CV_32FC1)
        inside = ((self._map1 >= 0) & (self._map1 <= w - 1) &
                  (self._map2 >= 0) & (self._map2 <= h - 1))
        self.valid_fraction = float(inside.mean())

        self.scale = float(scale)
        self.width = int(round(w * self.scale))
        self.height = int(round(h * self.scale))
        sx, sy = self.width / w, self.height / h
        self.K = K_new.copy()
        self.K[0, :] *= sx
        self.K[1, :] *= sy

        # список кадров
        stamps, names = [], []
        with open(cam_dir / "data.csv", "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                stamps.append(int(row[0]))
                names.append(row[1].strip())
        sel = slice(start, end, stride)
        self._stamps_ns = np.array(stamps, dtype=np.int64)[sel]
        self._paths = [cam_dir / "data" / n for n in names][sel]
        self.timestamps = self._stamps_ns.astype(np.float64) * 1e-9

        self._load_groundtruth(mav / "state_groundtruth_estimate0" / "data.csv")

    def _load_groundtruth(self, path):
        self.gt_T_wc = [None] * len(self._paths)
        if not path.is_file():
            return
        data = np.loadtxt(path, delimiter=",", comments="#", usecols=range(8))
        t = data[:, 0] * 1e-9
        pos = data[:, 1:4]
        quat_wxyz = data[:, 4:8]
        rot = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])   # scipy хочет xyzw
        slerp = Slerp(t, rot)

        for i, ts in enumerate(self.timestamps):
            if ts < t[0] or ts > t[-1]:
                continue
            p = np.array([np.interp(ts, t, pos[:, k]) for k in range(3)])
            T_wb = np.eye(4)
            T_wb[:3, :3] = slerp([ts]).as_matrix()[0]
            T_wb[:3, 3] = p
            self.gt_T_wc[i] = T_wb @ self.T_bc

    def __len__(self):
        return len(self._paths)

    def read_image(self, i):
        img = cv2.imread(str(self._paths[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self._paths[i])
        img = cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)
        if self.scale != 1.0:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return np.repeat(img[:, :, None], 3, axis=2)

    def __getitem__(self, i):
        return Frame(index=i, timestamp=float(self.timestamps[i]), image=self.read_image(i),
                     K=self.K.copy(), T_wc=self.gt_T_wc[i])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def groundtruth(self):
        # индексы, времена и позы (N, 4, 4) только для кадров с эталоном
        idx = np.array([i for i, T in enumerate(self.gt_T_wc) if T is not None], dtype=np.int64)
        poses = np.stack([self.gt_T_wc[i] for i in idx]) if len(idx) else np.zeros((0, 4, 4))
        return idx, self.timestamps[idx], poses

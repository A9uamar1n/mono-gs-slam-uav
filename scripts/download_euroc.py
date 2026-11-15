# скачивание отдельных последовательностей EuRoC MAV из ETH Research Collection.
# там большие zip (machine_hall.zip 12.7 ГБ, vicon_room1.zip 6 ГБ), поэтому читаем
# оглавление zip через HTTP Range и качаем только нужный файл
#   python scripts/download_euroc.py --list vicon_room1
#   python scripts/download_euroc.py --get vicon_room1 vicon_room1/V1_01_easy/V1_01_easy.zip --out data/euroc
import argparse
import io
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib

BASE = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/{}/content"
ARCHIVES = {
    "machine_hall": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
    "vicon_room1": "02ecda9a-298f-498b-970c-b7c44334d880",
    "vicon_room2": "ea12bc01-3677-4b4c-853d-87c7870b8c44",
}
UA = {"User-Agent": "Mozilla/5.0 (dataset-fetcher)"}


class HttpRangeFile(io.IOBase):
    # файл только для чтения, данные берутся Range-запросами

    def __init__(self, url, block=4 << 20):
        self.url = url
        self.block = block
        self.pos = 0
        self._buf_start = 0
        self._buf = b""
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        for attempt in range(10):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status != 206:
                        raise RuntimeError(f"сервер не поддерживает Range (HTTP {r.status})")
                    self.size = int(r.headers["Content-Range"].split("/")[1])
                break
            except urllib.error.HTTPError as e:   # 429 - слишком часто, ждём
                if e.code != 429 or attempt == 9:
                    raise
                wait = int(e.headers.get("Retry-After", 0) or 0) or 30 * (attempt + 1)
                print(f"  HTTP 429, пауза {wait} с", file=sys.stderr, flush=True)
                time.sleep(wait)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={start}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    raise
                wait = 20 * (attempt + 1)
                if isinstance(e, urllib.error.HTTPError):
                    wait = int(e.headers.get("Retry-After", 0) or 0) or wait
                print(f"  повтор после ошибки: {e}, пауза {wait} с", file=sys.stderr)
                time.sleep(wait)
        return b""

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        bs, be = self._buf_start, self._buf_start + len(self._buf)
        if not (bs <= self.pos and self.pos + n <= be):
            length = max(n, self.block)
            end = min(self.pos + length, self.size) - 1
            self._buf = self._fetch(self.pos, end)
            self._buf_start = self.pos
        o = self.pos - self._buf_start
        data = self._buf[o:o + n]
        self.pos += len(data)
        return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", metavar="ARCHIVE")
    ap.add_argument("--get", nargs=2, metavar=("ARCHIVE", "MEMBER"))
    ap.add_argument("--out", default="data/euroc")
    args = ap.parse_args()

    name = args.list or args.get[0]
    f = HttpRangeFile(BASE.format(ARCHIVES[name]))
    zf = zipfile.ZipFile(f)

    if args.list:
        for i in zf.infolist():
            print(f"{i.file_size / 1e9:8.2f} ГБ  {i.filename}")
        return

    member = args.get[1]
    info = zf.getinfo(member)
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, os.path.basename(member))
    print(f"{member}: {info.file_size / 1e9:.2f} ГБ -> {dst}")
    # сам файл качаем одним запросом, а не блоками (иначе сервер отвечает 429).
    # если оборвалось - докачиваем с места обрыва
    f.seek(info.header_offset)
    hdr = f.read(30)
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    data_start = info.header_offset + 30 + name_len + extra_len
    stored = info.compress_type == zipfile.ZIP_STORED
    part = dst + ".part"
    done = os.path.getsize(part) if (stored and os.path.exists(part)) else 0
    total = info.compress_size
    decomp = None if stored else zlib.decompressobj(-15)
    last_print = 0.0

    with open(part, "ab" if done else "wb") as out:
        fails = 0
        while done < total:
            if not stored and done:
                raise RuntimeError("докачка сжатого файла не поддерживается, удалите .part")
            req = urllib.request.Request(f.url, headers={
                **UA, "Range": f"bytes={data_start + done}-{data_start + total - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk if stored else decomp.decompress(chunk))
                        done += len(chunk)
                        fails = 0
                        if time.time() - last_print > 5:
                            last_print = time.time()
                            print(f"  {done / 1e6:8.0f} / {total / 1e6:.0f} МБ", flush=True)
            except urllib.error.HTTPError as e:
                fails += 1
                wait = int(e.headers.get("Retry-After", 0) or 0) or min(30 * fails, 300)
                print(f"  HTTP {e.code}, пауза {wait} с", flush=True)
                if fails > 10:
                    raise
                time.sleep(wait)
            except Exception as e:  # noqa: BLE001  оборвалось соединение
                fails += 1
                print(f"  обрыв ({e}), повтор", flush=True)
                if fails > 10 or not stored:
                    raise
                time.sleep(5)
        if decomp is not None:
            out.write(decomp.flush())

    if os.path.getsize(part) != info.file_size:
        raise RuntimeError(f"размер {os.path.getsize(part)} не совпал с ожидаемым {info.file_size}")
    with zipfile.ZipFile(part) as inner:   # проверяем, что архив открывается
        print(f"  готово, файлов внутри: {len(inner.namelist())}")
    shutil.move(part, dst)


if __name__ == "__main__":
    main()

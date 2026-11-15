# патч исходников gsplat 1.4.0, чтобы собрался и работал на GTX 10xx (compute capability 6.1)
#   python scripts/patch_gsplat_gtx10.py ../gsplat_src
# 1) cg::labeled_partition есть только с архитектуры 7.0. В backward проекции она нужна
#    только чтобы сначала сложить градиенты внутри варпа. Убираем это сложение, каждый
#    поток сам делает atomicAdd - сумма та же, меняется только порядок сложения.
# 2) cudaFuncSetAttribute(MaxDynamicSharedMemorySize) падает на старых картах; если
#    ядру хватает обычных 48 КБ, ошибку не считаем фатальной.
# можно запускать повторно, второй раз ничего не поменяет
import re
import sys
from pathlib import Path

csrc = Path(sys.argv[1]) / "gsplat" / "cuda" / "csrc"

helpers = csrc / "helpers.cuh"
text = helpers.read_text(encoding="utf-8")
if "struct SoloGroup" not in text:
    shim = ("// Заглушка группы из одного потока для архитектур без cg::labeled_partition.\n"
            "struct SoloGroup {\n"
            "    inline __device__ unsigned int thread_rank() const { return 0; }\n"
            "};\n\n")
    marker = "template <class WarpT, class ScalarT>"
    assert marker in text
    text = text.replace(marker, shim + marker, 1)
    helpers.write_text(text, encoding="utf-8")
    print("helpers.cuh: добавлена SoloGroup")

n_part = n_sum = n_shm = 0
for f in sorted(csrc.glob("*.cu")):
    src = f.read_text(encoding="utf-8")
    new, a = re.subn(r"cg::labeled_partition\(warp,\s*\w+\)", "SoloGroup()", src)
    new, b = re.subn(r"^[ \t]*warpSum\([^;\n]*,\s*warp_group_[gc]\);[ \t]*\r?\n", "", new, flags=re.M)
    new, c = re.subn(r"\)\s*!=\s*cudaSuccess\)\s*\{", ") != cudaSuccess && shared_mem > 48 * 1024) {", new)
    if new != src:
        f.write_text(new, encoding="utf-8")
        print(f"{f.name}: labeled_partition {a}, warpSum {b}, shared_mem {c}")
    n_part += a; n_sum += b; n_shm += c
print(f"итого: labeled_partition {n_part}, warpSum {n_sum}, проверок shared_mem {n_shm}")

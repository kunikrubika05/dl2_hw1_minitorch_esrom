import csv
import statistics
import time
from pathlib import Path

import numpy as np
from numba import cuda

import minitorch


GPU_BACKEND = minitorch.TensorBackend(minitorch.CudaOps)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "module3_matmul.csv"


def naive_matmul(a, b):
    size = len(a)
    return [
        [sum(a[i][k] * b[k][j] for k in range(size)) for j in range(size)]
        for i in range(size)
    ]


def cuda_matmul(left, right):
    result = left @ right
    cuda.synchronize()
    return result


def main():
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    rng = np.random.default_rng(20260919)
    sizes = (8, 16, 32, 64, 128, 256)
    naive_times = []
    cuda_times = []

    warmup_left = minitorch.tensor([[1.0, 2.0], [3.0, 4.0]], backend=GPU_BACKEND)
    warmup_right = minitorch.tensor([[1.0, 0.0], [0.0, 1.0]], backend=GPU_BACKEND)
    cuda_matmul(warmup_left, warmup_right)

    for size in sizes:
        a = rng.random((size, size)).tolist()
        b = rng.random((size, size)).tolist()
        left = minitorch.tensor(a, backend=GPU_BACKEND)
        right = minitorch.tensor(b, backend=GPU_BACKEND)

        start = time.perf_counter()
        expected = naive_matmul(a, b)
        naive_time = time.perf_counter() - start

        samples = []
        result = None
        for _ in range(7):
            start = time.perf_counter()
            result = cuda_matmul(left, right)
            samples.append(time.perf_counter() - start)

        storage = result._tensor._storage
        if hasattr(storage, "copy_to_host"):
            storage = storage.copy_to_host()
        actual = np.asarray(storage).reshape(size, size)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        cuda_time = statistics.median(samples)
        naive_times.append(naive_time)
        cuda_times.append(cuda_time)
        print(
            f"size={size}, naive={naive_time:.6f}s, "
            f"cuda={cuda_time:.6f}s, speedup={naive_time / cuda_time:.1f}x",
            flush=True,
        )

    with OUTPUT_PATH.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("size", "naive_seconds", "cuda_seconds"))
        writer.writerows(zip(sizes, naive_times, cuda_times))


if __name__ == "__main__":
    main()

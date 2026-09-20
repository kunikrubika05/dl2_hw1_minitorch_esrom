import argparse
import random
import time

import minitorch

from run_fast_tensor import FastTensorBackend, FastTrain, GPUBackend


def module3_result(name, hidden, seed, backend):
    random.seed(seed)
    data = minitorch.datasets[name](50)
    trainer = FastTrain(hidden, backend=backend)
    result = {"loss": 0.0, "correct": 0}
    label = name if hidden == 10 else f"{name} (100 hidden)"

    def log(epoch, loss, correct, losses):
        if epoch % 50 == 0 or epoch == 499:
            print(
                f"{label}: epoch={epoch}, loss={float(loss):.6f}, "
                f"correct={correct}/50",
                flush=True,
            )
        if epoch == 499:
            result["loss"] = loss
            result["correct"] = correct

    print(f"Starting {label}", flush=True)
    start = time.perf_counter()
    trainer.train(data, 0.05, 500, log)
    elapsed = time.perf_counter() - start
    return float(result["loss"]), int(result["correct"]), elapsed / 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    backend = FastTensorBackend
    if args.backend == "gpu":
        if GPUBackend is None:
            raise RuntimeError("CUDA is not available")
        backend = GPUBackend

    for name, hidden, seed in (
        ("Simple", 10, 20260919),
        ("Diag", 10, 20260919),
        ("Split", 10, 20260919),
        ("Xor", 10, 20260919),
    ):
        loss, correct, epoch_time = module3_result(name, hidden, seed, backend)
        print(
            f"{name}: epoch=499, loss={loss:.6f}, correct={correct}/50, "
            f"seconds_per_epoch={epoch_time:.4f}",
            flush=True,
        )

    loss, correct, epoch_time = module3_result("Split", 100, 20260919, backend)
    print(
        f"Split (100 hidden): epoch=499, loss={loss:.6f}, "
        f"correct={correct}/50, seconds_per_epoch={epoch_time:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

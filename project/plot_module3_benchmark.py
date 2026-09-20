import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"

plt.rcParams.update(
    {
        "figure.figsize": (11, 7),
        "font.family": "DejaVu Serif",
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.titlesize": 24,
        "axes.labelsize": 16,
    }
)


def main():
    with (OUTPUT_DIR / "module3_matmul.csv").open() as source:
        rows = list(csv.DictReader(source))

    sizes = [int(row["size"]) for row in rows]
    naive_times = [float(row["naive_seconds"]) for row in rows]
    cuda_times = [float(row["cuda_seconds"]) for row in rows]

    fig, ax = plt.subplots()
    ax.plot(sizes, naive_times, marker="o", label="Naive Python")
    ax.plot(sizes, cuda_times, marker="o", label="CUDA")
    ax.set(
        xlabel="Square matrix size",
        ylabel="Seconds",
        title="MiniTorch matrix multiplication",
        yscale="log",
    )
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "module3_matmul.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

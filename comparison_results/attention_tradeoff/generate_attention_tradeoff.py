"""Generate a clean performance-speed trade-off plot from local JSON data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "attention_tradeoff_data.json"
OUTPUT_FILE = BASE_DIR / "attention_tradeoff.png"


def configure_plot() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def plot_tradeoff(data: dict) -> None:
    models = data["models"]
    x = [item["fps"] for item in models]
    y = [item["f1_score"] for item in models]
    labels = [item["name"] for item in models]
    colors = ["#4C78A8", "#E45756"]

    fig, ax = plt.subplots(figsize=(10.5, 6.8), facecolor="white")

    ax.plot(x, y, linestyle="--", linewidth=2.2, color="#B8C1CC", zorder=1)

    for idx, (xv, yv, label) in enumerate(zip(x, y, labels)):
        ax.scatter(
            xv,
            yv,
            s=220,
            color=colors[idx],
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(xv, yv),
            xytext=(8, 10 if idx == 0 else 12),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=colors[idx],
        )

    ax.set_title(data["title"], fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel(data["x_axis"]["label"], fontsize=15, fontweight="bold")
    ax.set_ylabel(data["y_axis"]["label"], fontsize=15, fontweight="bold")
    ax.set_xlim(min(x) - 0.5, max(x) + 0.6)
    ax.set_ylim(min(y) - 0.02, max(y) + 0.02)
    ax.grid(True, linestyle="--", alpha=0.28)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#BFC7D5")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plot()
    data = load_data()
    plot_tradeoff(data)
    print(f"Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

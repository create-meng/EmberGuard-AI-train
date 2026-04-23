"""Generate boxplot comparison charts from local JSON data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "attention_boxplot_data.json"
OUTPUT_FILE = BASE_DIR / "attention_boxplot.png"


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


def style_boxplot(boxplot, colors: list[str]) -> None:
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    for median in boxplot["medians"]:
        median.set_color("#1F2937")
        median.set_linewidth(2)

    for whisker in boxplot["whiskers"]:
        whisker.set_color("#6B7280")
        whisker.set_linewidth(1.3)

    for cap in boxplot["caps"]:
        cap.set_color("#6B7280")
        cap.set_linewidth(1.3)


def plot_boxplots(data: dict) -> None:
    from matplotlib.patches import Rectangle
    
    # 淡色系配色方案 - 与detection_stats统一，带渐变
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), facecolor="#FAFBFC")

    for ax, metric in zip(axes, data["metrics"]):
        labels = [model["name"] for model in metric["models"]]
        values = [model["values"] for model in metric["models"]]

        # 先绘制箱线图获取位置信息
        boxplot = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.55,
            tick_labels=labels,
        )
        
        # 为每个箱体添加垂直渐变效果
        for i, (patch, model_values) in enumerate(zip(boxplot["boxes"], values)):
            # 获取箱体的位置和大小
            path = patch.get_path()
            vertices = path.vertices
            x_min = vertices[:, 0].min()
            x_max = vertices[:, 0].max()
            y_min = vertices[:, 1].min()
            y_max = vertices[:, 1].max()
            
            # 清除原始箱体
            patch.set_facecolor('none')
            patch.set_edgecolor('none')
            
            # 绘制渐变箱体
            n_segments = 50
            box_height = y_max - y_min
            box_width = x_max - x_min
            
            if i == 0:  # YOLO - 淡蓝渐变
                for j in range(n_segments):
                    y_start = y_min + j * box_height / n_segments
                    segment_height = box_height / n_segments
                    ratio = j / n_segments
                    # 从 #81C4E8 (淡蓝) 到 #E3F2FD (极淡蓝)
                    r = int(129 + (227 - 129) * ratio)
                    g = int(196 + (242 - 196) * ratio)
                    b = int(232 + (253 - 232) * ratio)
                    color = f'#{r:02x}{g:02x}{b:02x}'
                    
                    segment = Rectangle((x_min, y_start), box_width, segment_height,
                                      facecolor=color, edgecolor='none', zorder=1)
                    ax.add_patch(segment)
                
                # 添加边框
                rect = Rectangle((x_min, y_min), box_width, box_height,
                               linewidth=1.5, edgecolor='#64B5F6',
                               facecolor='none', zorder=2)
                ax.add_patch(rect)
                
            else:  # YOLO+LSTM - 淡橙渐变
                for j in range(n_segments):
                    y_start = y_min + j * box_height / n_segments
                    segment_height = box_height / n_segments
                    ratio = j / n_segments
                    # 从 #FFB74D (淡橙) 到 #FFF3E0 (极淡橙)
                    r = 255
                    g = int(183 + (243 - 183) * ratio)
                    b = int(77 + (224 - 77) * ratio)
                    color = f'#{r:02x}{g:02x}{b:02x}'
                    
                    segment = Rectangle((x_min, y_start), box_width, segment_height,
                                      facecolor=color, edgecolor='none', zorder=1)
                    ax.add_patch(segment)
                
                # 添加边框
                rect = Rectangle((x_min, y_min), box_width, box_height,
                               linewidth=1.5, edgecolor='#FFA726',
                               facecolor='none', zorder=2)
                ax.add_patch(rect)
        
        # 中位线使用深色
        for median in boxplot["medians"]:
            median.set_color("#424242")
            median.set_linewidth(2.5)
            median.set_zorder(3)
        
        # 须线使用灰色
        for whisker in boxplot["whiskers"]:
            whisker.set_color("#9E9E9E")
            whisker.set_linewidth(1.3)
            whisker.set_linestyle("--")
        
        # 端点使用灰色
        for cap in boxplot["caps"]:
            cap.set_color("#9E9E9E")
            cap.set_linewidth(1.3)

        ax.set_facecolor("#FFFFFF")
        ax.set_title(metric["label"], fontsize=16, fontweight="bold", pad=12, color="#212121")
        ax.set_ylabel(metric["label"], fontsize=13, fontweight="bold", color="#424242")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.8, color="#BDBDBD", zorder=0)
        ax.tick_params(axis="x", labelrotation=10, labelsize=11, colors="#616161")
        ax.tick_params(axis="y", labelsize=10, colors="#616161")

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#E0E0E0")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(data["title"], fontsize=20, fontweight="bold", y=0.98, color="#212121")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close(fig)


def main() -> None:
    configure_plot()
    data = load_data()
    plot_boxplots(data)
    print(f"Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

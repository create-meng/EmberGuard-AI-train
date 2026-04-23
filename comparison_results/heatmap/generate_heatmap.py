"""生成热力图 - 展示YOLO和LSTM模型的性能对比"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 配置中文字体
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "comparison_report.json"
OUTPUT_FILE = BASE_DIR / "performance_heatmap.png"


def load_and_process_data():
    """加载并处理数据"""
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    
    yolo_data = {
        'avg_confidence': [],
        'volatility': [],
        'fire_ratio': [],
        'smoke_ratio': [],
        'accuracy': [],
        'video_names': []
    }
    
    lstm_data = {
        'avg_confidence': [],
        'volatility': [],
        'fire_ratio': [],
        'smoke_ratio': [],
        'accuracy': [],
        'video_names': []
    }
    
    for video in data['details']:
        ground_truth = video['ground_truth']
        video_name = video['video_name'][:20]  # 截断视频名称
        
        # YOLO数据
        yolo = video['yolo']
        yolo_pred = yolo['prediction']
        yolo_data['avg_confidence'].append(yolo['avg_confidence'])
        yolo_data['volatility'].append(yolo['volatility'])
        yolo_data['fire_ratio'].append(yolo['detection_ratios']['fire'])
        yolo_data['smoke_ratio'].append(yolo['detection_ratios']['smoke'])
        yolo_data['accuracy'].append(1 if yolo_pred == ground_truth else 0)
        yolo_data['video_names'].append(video_name)
        
        # LSTM数据
        lstm = video['lstm']
        lstm_pred = lstm['prediction']
        lstm_data['avg_confidence'].append(lstm['avg_confidence'])
        lstm_data['volatility'].append(lstm['volatility'])
        lstm_data['fire_ratio'].append(lstm['lstm_ratios']['fire'])
        lstm_data['smoke_ratio'].append(lstm['lstm_ratios']['smoke'])
        lstm_data['accuracy'].append(1 if lstm_pred == ground_truth else 0)
        lstm_data['video_names'].append(video_name)
    
    return yolo_data, lstm_data


def plot_performance_heatmap():
    """生成性能指标热力图"""
    yolo_data, lstm_data = load_and_process_data()
    
    # 准备数据矩阵
    metrics = ['平均置信度', '波动性', '火焰检测比例', '烟雾检测比例', '准确率']
    
    # 计算每个指标的平均值
    yolo_values = [
        np.mean(yolo_data['avg_confidence']),
        np.mean(yolo_data['volatility']),
        np.mean(yolo_data['fire_ratio']),
        np.mean(yolo_data['smoke_ratio']),
        np.mean(yolo_data['accuracy'])
    ]
    
    lstm_values = [
        np.mean(lstm_data['avg_confidence']),
        np.mean(lstm_data['volatility']),
        np.mean(lstm_data['fire_ratio']),
        np.mean(lstm_data['smoke_ratio']),
        np.mean(lstm_data['accuracy'])
    ]
    
    # 创建数据矩阵
    data_matrix = np.array([yolo_values, lstm_values]).T
    
    # 创建自定义渐变色 - 淡绿到淡橙
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#E8F5E9', '#81C784', '#FFB74D', '#FFF3E0']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#FAFBFC")
    
    # 绘制热力图
    im = ax.imshow(data_matrix, cmap=cmap_custom, aspect='auto', vmin=0, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(['YOLO', 'YOLO+LSTM'], fontsize=14, fontweight='bold', color='#1A1A1A')
    ax.set_yticklabels(metrics, fontsize=13, color='#1A1A1A')
    ax.set_facecolor('#FFFFFF')
    
    # 在每个单元格中显示数值
    for i in range(len(metrics)):
        for j in range(2):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                          ha="center", va="center", color="#1A1A1A", 
                          fontsize=13, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('数值', fontsize=13, fontweight='bold', color='#1A1A1A')
    cbar.ax.tick_params(labelsize=11, colors='#1A1A1A')
    
    # 设置标题
    ax.set_title('YOLO vs YOLO+LSTM 性能指标热力图', 
                fontsize=17, fontweight='bold', pad=20, color='#1A1A1A')
    
    # 设置网格
    ax.set_xticks(np.arange(2 + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#E0E0E0", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close()
    print(f"✓ 已生成: performance_heatmap.png")


def plot_category_heatmap():
    """生成分类别性能热力图"""
    yolo_data, lstm_data = load_and_process_data()
    
    # 按ground truth分类统计
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    
    categories = {'fire': [], 'smoke': [], 'normal': []}
    
    for video in data['details']:
        gt = video['ground_truth']
        yolo_correct = 1 if video['yolo']['prediction'] == gt else 0
        lstm_correct = 1 if video['lstm']['prediction'] == gt else 0
        
        if gt in categories:
            categories[gt].append([yolo_correct, lstm_correct])
    
    # 计算每个类别的准确率
    category_names = []
    yolo_acc = []
    lstm_acc = []
    
    for cat, results in categories.items():
        if results:
            category_names.append(cat.capitalize())
            results_array = np.array(results)
            yolo_acc.append(np.mean(results_array[:, 0]))
            lstm_acc.append(np.mean(results_array[:, 1]))
    
    # 创建数据矩阵
    data_matrix = np.array([yolo_acc, lstm_acc]).T
    
    # 创建自定义渐变色 - 淡粉到淡橙
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#FCE4EC', '#F48FB1', '#FFB74D', '#FFF3E0']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#FAFBFC")
    
    # 绘制热力图
    im = ax.imshow(data_matrix, cmap=cmap_custom, aspect='auto', vmin=0, vmax=1)
    
    # 设置刻度
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(len(category_names)))
    ax.set_xticklabels(['YOLO', 'YOLO+LSTM'], fontsize=14, fontweight='bold', color='#1A1A1A')
    ax.set_yticklabels(category_names, fontsize=13, color='#1A1A1A')
    ax.set_facecolor('#FFFFFF')
    
    # 在每个单元格中显示数值
    for i in range(len(category_names)):
        for j in range(2):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2%}',
                          ha="center", va="center", color="#1A1A1A", 
                          fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('准确率', fontsize=13, fontweight='bold', color='#1A1A1A')
    cbar.ax.tick_params(labelsize=11, colors='#1A1A1A')
    
    # 设置标题
    ax.set_title('各类别检测准确率热力图', 
                fontsize=17, fontweight='bold', pad=20, color='#1A1A1A')
    
    # 设置网格
    ax.set_xticks(np.arange(2 + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(category_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#E0E0E0", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE.parent / "category_heatmap.png", 
                dpi=300, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close()
    print(f"✓ 已生成: category_heatmap.png")


def plot_correlation_heatmap():
    """生成指标相关性热力图"""
    yolo_data, lstm_data = load_and_process_data()
    
    # 创建2x1子图 - 调整间距
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#FAFBFC")
    fig.subplots_adjust(wspace=0.3)  # 增加子图之间的间距
    
    # YOLO相关性矩阵
    yolo_matrix = np.array([
        yolo_data['avg_confidence'],
        yolo_data['volatility'],
        yolo_data['fire_ratio'],
        yolo_data['smoke_ratio'],
        yolo_data['accuracy']
    ])
    yolo_corr = np.corrcoef(yolo_matrix)
    
    # LSTM相关性矩阵
    lstm_matrix = np.array([
        lstm_data['avg_confidence'],
        lstm_data['volatility'],
        lstm_data['fire_ratio'],
        lstm_data['smoke_ratio'],
        lstm_data['accuracy']
    ])
    lstm_corr = np.corrcoef(lstm_matrix)
    
    metrics_short = ['置信度', '波动性', '火焰', '烟雾', '准确率']
    
    # 创建自定义渐变色 - 从淡蓝到淡橙
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#E3F2FD', '#81C4E8', '#FFFFFF', '#FFB74D', '#FFF3E0']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)
    
    # 绘制YOLO相关性热力图
    im1 = axes[0].imshow(yolo_corr, cmap=cmap_custom, aspect='auto', vmin=-1, vmax=1)
    axes[0].set_xticks(np.arange(len(metrics_short)))
    axes[0].set_yticks(np.arange(len(metrics_short)))
    axes[0].set_xticklabels(metrics_short, fontsize=12, color='#1A1A1A', rotation=45, ha='right')
    axes[0].set_yticklabels(metrics_short, fontsize=12, color='#1A1A1A')
    axes[0].set_title('YOLO 指标相关性', fontsize=15, fontweight='bold', pad=15, color='#1A1A1A')
    axes[0].set_facecolor('#FFFFFF')
    
    # 添加网格
    axes[0].set_xticks(np.arange(len(metrics_short) + 1) - 0.5, minor=True)
    axes[0].set_yticks(np.arange(len(metrics_short) + 1) - 0.5, minor=True)
    axes[0].grid(which="minor", color="#E0E0E0", linestyle='-', linewidth=2)
    axes[0].tick_params(which="minor", size=0)
    
    for i in range(len(metrics_short)):
        for j in range(len(metrics_short)):
            text = axes[0].text(j, i, f'{yolo_corr[i, j]:.2f}',
                              ha="center", va="center", 
                              color="#1A1A1A",
                              fontsize=11, fontweight='bold')
    
    # 绘制LSTM相关性热力图
    im2 = axes[1].imshow(lstm_corr, cmap=cmap_custom, aspect='auto', vmin=-1, vmax=1)
    axes[1].set_xticks(np.arange(len(metrics_short)))
    axes[1].set_yticks(np.arange(len(metrics_short)))
    axes[1].set_xticklabels(metrics_short, fontsize=12, color='#1A1A1A', rotation=45, ha='right')
    axes[1].set_yticklabels(metrics_short, fontsize=12, color='#1A1A1A')
    axes[1].set_title('YOLO+LSTM 指标相关性', fontsize=15, fontweight='bold', pad=15, color='#1A1A1A')
    axes[1].set_facecolor('#FFFFFF')
    
    # 添加网格
    axes[1].set_xticks(np.arange(len(metrics_short) + 1) - 0.5, minor=True)
    axes[1].set_yticks(np.arange(len(metrics_short) + 1) - 0.5, minor=True)
    axes[1].grid(which="minor", color="#E0E0E0", linestyle='-', linewidth=2)
    axes[1].tick_params(which="minor", size=0)
    
    for i in range(len(metrics_short)):
        for j in range(len(metrics_short)):
            text = axes[1].text(j, i, f'{lstm_corr[i, j]:.2f}',
                              ha="center", va="center",
                              color="#1A1A1A",
                              fontsize=11, fontweight='bold')
    
    # 添加颜色条 - 放在右侧，不遮挡图表
    # 创建一个新的axes用于colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('相关系数', fontsize=13, fontweight='bold', color='#1A1A1A', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=11, colors='#1A1A1A')
    
    plt.suptitle('模型指标相关性分析', fontsize=18, fontweight='bold', y=0.98, color='#1A1A1A')
    plt.savefig(OUTPUT_FILE.parent / "correlation_heatmap.png", 
                dpi=300, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close()
    print(f"✓ 已生成: correlation_heatmap.png")


def main():
    """生成所有热力图"""
    print("开始生成热力图...")
    print("-" * 50)
    
    plot_performance_heatmap()
    plot_category_heatmap()
    plot_correlation_heatmap()
    
    print("-" * 50)
    print("✓ 所有热力图生成完成！")
    print(f"\n输出目录: {BASE_DIR}")
    print("\n生成的图表:")
    print("1. performance_heatmap.png - 性能指标热力图")
    print("2. category_heatmap.png - 各类别检测准确率热力图")
    print("3. correlation_heatmap.png - 指标相关性分析热力图")


if __name__ == "__main__":
    main()

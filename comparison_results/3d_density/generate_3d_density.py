"""生成3D核密度图 - 展示YOLO和LSTM模型的多维度性能分布"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import stats

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
OUTPUT_FILE = BASE_DIR / "3d_density.png"


def load_and_process_data():
    """加载并处理数据"""
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    
    yolo_data = {
        'avg_confidence': [],
        'volatility': [],
        'fire_ratio': [],
        'smoke_ratio': [],
        'correct': []
    }
    
    lstm_data = {
        'avg_confidence': [],
        'volatility': [],
        'fire_ratio': [],
        'smoke_ratio': [],
        'correct': []
    }
    
    for video in data['details']:
        ground_truth = video['ground_truth']
        
        # YOLO数据
        yolo = video['yolo']
        yolo_pred = yolo['prediction']
        yolo_data['avg_confidence'].append(yolo['avg_confidence'])
        yolo_data['volatility'].append(yolo['volatility'])
        yolo_data['fire_ratio'].append(yolo['detection_ratios']['fire'])
        yolo_data['smoke_ratio'].append(yolo['detection_ratios']['smoke'])
        yolo_data['correct'].append(1 if yolo_pred == ground_truth else 0)
        
        # LSTM数据
        lstm = video['lstm']
        lstm_pred = lstm['prediction']
        lstm_data['avg_confidence'].append(lstm['avg_confidence'])
        lstm_data['volatility'].append(lstm['volatility'])
        lstm_data['fire_ratio'].append(lstm['lstm_ratios']['fire'])
        lstm_data['smoke_ratio'].append(lstm['lstm_ratios']['smoke'])
        lstm_data['correct'].append(1 if lstm_pred == ground_truth else 0)
    
    return yolo_data, lstm_data


def create_3d_wireframe_plot(ax, x, y, z, cmap_name, label, linewidth=1.5, existing_xlim=None, existing_ylim=None):
    """创建3D线框图 - 只有线条，带渐变色"""
    
    # 过滤掉无效值
    valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z) | 
                   np.isinf(x) | np.isinf(y) | np.isinf(z))
    x = np.array(x)[valid_mask]
    y = np.array(y)[valid_mask]
    z = np.array(z)[valid_mask]
    
    if len(x) < 3:
        return None, None
    
    # 保存原始数据范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # 创建网格 - 使用原始数据范围
    xi = np.linspace(x_min, x_max, 60)
    yi = np.linspace(y_min, y_max, 60)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 对每个网格点计算核密度估计
    positions = np.vstack([Xi.ravel(), Yi.ravel()])
    values = np.vstack([x, y])
    
    try:
        kernel = stats.gaussian_kde(values)
        Zi = kernel(positions).reshape(Xi.shape)
        
        # 归一化Z值
        Zi = Zi / Zi.max() * 1.8
        
        # 获取colormap
        cmap = plt.get_cmap(cmap_name)
        
        # 归一化Z值用于颜色映射
        norm_z = (Zi - Zi.min()) / (Zi.max() - Zi.min() + 1e-10)
        
        # 绘制沿X方向的线条（每一行）
        for i in range(0, Xi.shape[0], 2):  # 每隔2行画一条线
            points = np.array([Xi[i, :], Yi[i, :], Zi[i, :]]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # 为每条线段设置颜色（根据Z值）
            colors = [cmap(norm_z[i, j]) for j in range(len(Xi[i, :]) - 1)]
            
            lc = Line3DCollection(segments, colors=colors, linewidths=linewidth, alpha=0.8)
            ax.add_collection3d(lc)
        
        # 绘制沿Y方向的线条（每一列）
        for j in range(0, Xi.shape[1], 2):  # 每隔2列画一条线
            points = np.array([Xi[:, j], Yi[:, j], Zi[:, j]]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # 为每条线段设置颜色（根据Z值）
            colors = [cmap(norm_z[i, j]) for i in range(len(Xi[:, j]) - 1)]
            
            lc = Line3DCollection(segments, colors=colors, linewidths=linewidth, alpha=0.8)
            ax.add_collection3d(lc)
        
        # 返回数据范围，用于统一设置坐标轴
        return (x_min, x_max), (y_min, y_max)
        
    except Exception as e:
        print(f"警告: {label} 核密度估计失败: {e}")
        ax.scatter(x, y, z, c=z, cmap=cmap_name, marker='o', s=30, alpha=0.6, label=label)
        return None, None


def plot_confidence_volatility_accuracy():
    """图1: 置信度 vs 波动性 vs 准确率分布"""
    yolo_data, lstm_data = load_and_process_data()
    
    # 过滤波动性数据，只保留 0-0.1 范围内的
    yolo_mask = np.array(yolo_data['volatility']) <= 0.1
    yolo_conf_filtered = np.array(yolo_data['avg_confidence'])[yolo_mask]
    yolo_vol_filtered = np.array(yolo_data['volatility'])[yolo_mask]
    yolo_correct_filtered = np.array(yolo_data['correct'])[yolo_mask]
    
    lstm_mask = np.array(lstm_data['volatility']) <= 0.1
    lstm_conf_filtered = np.array(lstm_data['avg_confidence'])[lstm_mask]
    lstm_vol_filtered = np.array(lstm_data['volatility'])[lstm_mask]
    lstm_correct_filtered = np.array(lstm_data['correct'])[lstm_mask]
    
    fig = plt.figure(figsize=(16, 11), facecolor="#FAFBFC")
    ax = fig.add_subplot(111, projection='3d', facecolor="#FFFFFF")
    
    # YOLO - 使用cool色系 (蓝紫渐变)
    xlim_yolo, ylim_yolo = create_3d_wireframe_plot(
        ax,
        yolo_conf_filtered,
        yolo_vol_filtered,
        yolo_correct_filtered,
        'cool',  # 蓝紫渐变
        'YOLO',
        linewidth=0.8
    )
    
    # LSTM - 使用autumn色系 (红橙黄渐变)
    xlim_lstm, ylim_lstm = create_3d_wireframe_plot(
        ax,
        lstm_conf_filtered,
        lstm_vol_filtered,
        lstm_correct_filtered,
        'autumn',  # 红橙黄渐变
        'YOLO+可学习权重特征融合',
        linewidth=0.8
    )
    
    # 设置坐标轴范围
    if xlim_yolo and xlim_lstm:
        x_min = min(xlim_yolo[0], xlim_lstm[0])
        x_max = max(xlim_yolo[1], xlim_lstm[1])
        
        # 添加一点边距
        x_margin = (x_max - x_min) * 0.05
        
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(0, 0.1)  # 固定波动性范围为 0-0.1
        ax.set_zlim(0, 2.0)
    
    # 设置轴标签 - 使用深色
    ax.set_xlabel('平均置信度', fontsize=14, fontweight='bold', labelpad=15, color='#1A1A1A')
    ax.set_ylabel('波动性', fontsize=14, fontweight='bold', labelpad=15, color='#1A1A1A')
    ax.set_zlabel('核密度分布', fontsize=14, fontweight='bold', labelpad=15, color='#1A1A1A')
    ax.set_title('置信度-波动性 3D核密度分布', fontsize=18, fontweight='bold', pad=25, color='#1A1A1A')
    
    # 设置刻度标签颜色
    ax.tick_params(axis='x', colors='#1A1A1A', labelsize=11)
    ax.tick_params(axis='y', colors='#1A1A1A', labelsize=11)
    ax.tick_params(axis='z', colors='#1A1A1A', labelsize=11)
    
    # 设置背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#E0E0E0')
    ax.yaxis.pane.set_edgecolor('#E0E0E0')
    ax.zaxis.pane.set_edgecolor('#E0E0E0')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='#BDBDBD')
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 添加图例 - 不要括号说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8B4789', edgecolor='#8B4789', label='YOLO'),
        Patch(facecolor='#FF8C42', edgecolor='#FF8C42', label='YOLO+可学习权重特征融合')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=13, 
             framealpha=0.95, edgecolor='#E0E0E0', fancybox=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE.parent / "3d_confidence_volatility.png", 
                dpi=300, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close()
    print(f"✓ 已生成: 3d_confidence_volatility.png")
    print(f"  YOLO: {len(yolo_conf_filtered)}/{len(yolo_data['volatility'])} 个数据点 (波动性 ≤ 0.1)")
    print(f"  LSTM: {len(lstm_conf_filtered)}/{len(lstm_data['volatility'])} 个数据点 (波动性 ≤ 0.1)")


def main():
    """生成所有3D图表"""
    print("开始生成3D核密度图...")
    print("-" * 50)
    
    plot_confidence_volatility_accuracy()
    
    print("-" * 50)
    print("✓ 所有3D图表生成完成！")
    print(f"\n输出目录: {BASE_DIR}")


if __name__ == "__main__":
    main()

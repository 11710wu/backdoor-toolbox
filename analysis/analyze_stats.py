#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 CSV 文件，计算攻击率/隐蔽率/迁移率的两两相关矩阵，并做PCA主成分分析。

用法：
  python /workspace/backdoor-toolbox-new/analysis/analyze_stats.py \
  --data-csv /workspace/backdoor-toolbox-new/analysis/data_cifar10.csv \
  --fit-plane \
  --plane-model z \
  --plane-color orange \
  --static-plot \
  --output-dir /workspace/backdoor-toolbox-new/analysis/analysis_outputs \
  --static-out scatter3d.png \
  --interactive-out scatter3d_interactive.html \
  --correlation-analysis \
  --corr-heatmap-out correlation_heatmap.png \
  --corr-scatter-prefix corr_scatter \
  --pareto-front \
  --pareto-out pareto_front.png \
  --pareto-front-attack-stealth \
  --pareto-attack-stealth-out pareto_front_attack_stealth.png \
  --pareto-front-attack-transfer \
  --pareto-attack-transfer-out pareto_front_attack_transfer.png \
  --pareto-front-3d \
  --pareto3d-out pareto_front_3d.png \
  --violin-plot \
  --violin-out violin_by_attack_type.png

说明：
  - 只支持 CSV 文件输入（不再支持 JSON）
  - 默认将所有组的数据合并后计算。
  - 若加 --standardize，会在做PCA前对三列做标准化(均值0/方差1)。
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go
import plotly.express as px


def load_all_points_from_csv(csv_path: str) -> pd.DataFrame:
    """从 CSV 文件加载数据点"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    
    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"CSV 文件为空: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        raise ValueError(f"读取 CSV 文件时出错: {csv_path}\n错误信息: {e}")
    
    # 检查必需的列
    required_cols = ['attack_type', 'attack_rate', 'stealth_rate', 'transfer_rate']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 文件缺少必需的列: {missing_cols}\n可用列: {list(df.columns)}")
    
    # 确保数值列为浮点数类型
    for col in ['attack_rate', 'stealth_rate', 'transfer_rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 移除包含 NaN 的行
    df = df.dropna(subset=['attack_rate', 'stealth_rate', 'transfer_rate'])
    
    # 添加 group_id（如果不存在）
    if 'group_id' not in df.columns:
        df['group_id'] = range(1, len(df) + 1)
    
    print(f"成功从 CSV 加载 {len(df)} 个数据点")
    return df







def run_pca(df: pd.DataFrame, standardize: bool = False) -> tuple[PCA, np.ndarray, pd.DataFrame]:
    cols = ['attack_rate', 'stealth_rate', 'transfer_rate']
    X = df[cols].to_numpy(dtype=float)
    if standardize:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    comps = pd.DataFrame(pca.components_, columns=cols, index=[f'PC{i+1}' for i in range(3)])
    return pca, X_pca, comps


def calculate_point_to_plane_distance(points: np.ndarray, plane_coeffs: tuple) -> np.ndarray:
    """计算各点到平面的距离"""
    a, b, c, d = plane_coeffs
    # 点到平面距离公式: |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
    numerator = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    return numerator / denominator


def plot_violin_by_attack_type(df: pd.DataFrame, output_path: str) -> None:
    """生成三个小提琴图子图，分别显示 attack_rate, stealth_rate, transfer_rate 在不同攻击类型上的分布"""
    if df.empty:
        print("小提琴图绘制跳过：数据为空")
        return
    
    # 检查是否有 attack_type 列
    if 'attack_type' not in df.columns:
        print("警告: 数据中没有 'attack_type' 列，无法按攻击类型分组")
        return
    
    # 获取所有攻击类型并排序
    attack_types = sorted(df['attack_type'].unique())
    
    # 定义三个指标
    metrics = [
        ('attack_rate', 'Attack Rate'),
        ('stealth_rate', 'Stealth Rate'),
        ('transfer_rate', 'Transfer Rate')
    ]
    
    # 使用 1x3 布局，更符合参考图片的样式
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribution of Metrics by Attack Type', fontsize=18, fontweight='bold', y=1.02)
    
    # 使用统一的配色方案（所有小提琴使用相同颜色，更简洁）
    # 也可以使用不同颜色：colors = plt.cm.Set3(np.linspace(0, 1, len(attack_types)))
    base_color = '#4A90E2'  # 蓝色系，更专业
    
    # 绘制三个指标的小提琴图
    for idx, (metric_col, metric_name) in enumerate(metrics):
        ax = axes[idx]
        
        # 准备数据：按攻击类型分组
        data_by_type = [df[df['attack_type'] == at][metric_col].dropna().values 
                       for at in attack_types]
        
        # 绘制小提琴图
        # showmeans=True: 显示均值线（细线或点），表示数据的平均值
        # showmedians=True: 显示中位数线（粗线），表示数据的中位数（50%分位数）
        # 中位数线将数据分成两半，均值线显示数据的平均水平
        parts = ax.violinplot(data_by_type, positions=range(len(attack_types)), 
                              showmeans=True, showmedians=True,
                              widths=0.7)  # 设置小提琴图的宽度
        
        # 为每个攻击类型设置颜色（统一颜色或不同颜色）
        for i, pc in enumerate(parts['bodies']):
            # 选项1：统一颜色（更简洁）
            pc.set_facecolor(base_color)
            # 选项2：不同颜色（取消注释下面这行，注释掉上面这行）
            # pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # 优化中位数线和均值线的样式
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(3.0)  # 中位数线更粗，更明显
        parts['cmeans'].set_color('darkred')
        parts['cmeans'].set_linewidth(2.5)
        parts['cmeans'].set_linestyle('--')
        parts['cbars'].set_color('black')
        parts['cbars'].set_linewidth(1.5)
        parts['cmins'].set_color('black')
        parts['cmins'].set_linewidth(1.5)
        parts['cmaxes'].set_color('black')
        parts['cmaxes'].set_linewidth(1.5)
        
        # 设置标签和标题
        ax.set_xticks(range(len(attack_types)))
        ax.set_xticklabels(attack_types, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_title(f'{metric_name} by Attack Type', fontsize=14, fontweight='bold', pad=15)
        
        # 优化网格线
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)  # 将网格线放在图形下方
        
        # 设置y轴范围，留出一些边距
        y_min = min([np.min(vals) for vals in data_by_type if len(vals) > 0])
        y_max = max([np.max(vals) for vals in data_by_type if len(vals) > 0])
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.15 * y_range)
        
        # 可选：添加统计信息（中位数和均值）
        # 如果数据点较多，可以注释掉这部分以避免图表过于拥挤
        show_stats = True  # 设置为 False 可以隐藏统计信息
        if show_stats:
            for i, at in enumerate(attack_types):
                values = df[df['attack_type'] == at][metric_col].dropna().values
                if len(values) > 0:
                    median_val = np.median(values)
                    mean_val = np.mean(values)
                    # 在顶部显示中位数（更重要的统计量）
                    y_top = ax.get_ylim()[1]
                    ax.text(i, y_top * 0.95, f'{median_val:.3f}', 
                           ha='center', va='top', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='black', alpha=0.85, linewidth=1))
        
        # 设置边框样式
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('gray')
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Violin plots saved to: {os.path.abspath(output_path)}")


def run_correlation_analysis(
    df: pd.DataFrame,
    heatmap_path: str,
    scatter_prefix: str,
) -> None:
    """生成相关性热力图以及成对散点 + 拟合直线"""
    cols = ['attack_rate', 'stealth_rate', 'transfer_rate']
    if df.empty:
        print("相关性分析跳过：数据为空")
        return

    corr = df[cols].corr()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_labels = [label.replace('_', ' ').title() for label in cols]
    plt.xticks(range(len(cols)), tick_labels, rotation=45, ha='right')
    plt.yticks(range(len(cols)), tick_labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(
                j,
                i,
                f"{corr.iloc[i, j]:.3f}",
                ha='center',
                va='center',
                color='black',
            )
    plt.title('Attack / Stealth / Transfer Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to: {os.path.abspath(heatmap_path)}")

    scatter_pairs = [
        ('attack_rate', 'stealth_rate', 'Attack vs Stealth', 'transfer_rate', 'Transfer Rate'),
        ('attack_rate', 'transfer_rate', 'Attack vs Transfer', 'stealth_rate', 'Stealth Rate'),
        ('stealth_rate', 'transfer_rate', 'Stealth vs Transfer', 'attack_rate', 'Attack Rate'),
    ]
    
    for x_col, y_col, title, color_col, color_label in scatter_pairs:
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        color_values = df[color_col].to_numpy()
        
        if len(x) < 2:
            print(f"散点拟合跳过 {title}：数据点不足")
            continue

        try:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
        except np.linalg.LinAlgError:
            print(f"散点拟合失败（奇异矩阵）：{title}")
            continue

        # 创建图形，稍微增大以容纳colorbar
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # 使用第三个指标的颜色映射（颜色越深，值越高）
        scatter = ax.scatter(x, y, c=color_values, cmap='viridis', 
                           alpha=0.7, edgecolors='black', linewidths=0.5, 
                           s=50, vmin=color_values.min(), vmax=color_values.max())
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(color_label, rotation=270, labelpad=15, fontsize=11, fontweight='bold')
        
        # 绘制拟合线
        ax.plot(line_x, line_y, color='red', linewidth=2, 
               label=f'Fit: y={slope:.3f}x+{intercept:.3f}')
        
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()

        output_name = f"{scatter_prefix}_{x_col}_vs_{y_col}.png"
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter with regression saved to: {os.path.abspath(output_name)}")


def compute_pareto_front_2d(df: pd.DataFrame, x_col: str, y_col: str, tol: float = 1e-9) -> pd.DataFrame:
    """
    计算二维帕累托前沿（默认值越大越优）。
    x_col 作为主排序列，y_col 作为次排序列。
    """
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    working = df.copy()
    working = working.sort_values(
        by=[x_col, y_col],
        ascending=[False, False],
        ignore_index=True,
    )

    front_rows = []
    best_secondary = -np.inf
    for _, row in working.iterrows():
        secondary_val = row[y_col]
        if secondary_val >= best_secondary - tol:
            if secondary_val > best_secondary + tol:
                best_secondary = secondary_val
            front_rows.append(row)

    if not front_rows:
        return pd.DataFrame(columns=df.columns)

    front_df = pd.DataFrame(front_rows).drop_duplicates(
        subset=[x_col, y_col], keep='first'
    )
    front_df = front_df.sort_values(by=x_col, ascending=True).reset_index(drop=True)
    return front_df


def plot_pareto_front_2d(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str,
    title: str,
) -> None:
    """二维帕累托前沿可视化"""
    if df.empty or pareto_df.empty:
        print(f"{title} 绘制跳过：数据为空或无最优点")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df[x_col],
        df[y_col],
        alpha=0.4,
        label='All Points',
        edgecolors='black',
        linewidths=0.3,
    )
    plt.scatter(
        pareto_df[x_col],
        pareto_df[y_col],
        color='red',
        label='Pareto Front Points',
        s=60,
    )
    plt.plot(
        pareto_df[x_col],
        pareto_df[y_col],
        color='red',
        linewidth=2,
    )
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"{title} saved to: {os.path.abspath(output_path)}")


def compute_pareto_front_multi(df: pd.DataFrame, columns: List[str], tol: float = 1e-9) -> pd.DataFrame:
    """通用多目标帕累托前沿计算，columns 中的值越大越优"""
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    values = df[columns].to_numpy()
    keep_indices = []

    for i in range(len(values)):
        dominated = False
        for j in range(len(values)):
            if i == j:
                continue
            better_or_equal = np.all(values[j] >= values[i] - tol)
            strictly_better = np.any(values[j] > values[i] + tol)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            keep_indices.append(i)

    if not keep_indices:
        return pd.DataFrame(columns=df.columns)

    front_df = df.iloc[keep_indices].drop_duplicates(subset=columns, keep='first')
    front_df = front_df.sort_values(by=columns, ascending=[False] * len(columns)).reset_index(drop=True)
    return front_df


def plot_pareto_front_3d(df: pd.DataFrame, pareto_df: pd.DataFrame, output_path: str) -> None:
    """三目标帕累托前沿可视化：Attack / Stealth / Transfer"""
    if df.empty or pareto_df.empty:
        print("三目标帕累托前沿绘制跳过：数据为空或无最优点")
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        df['attack_rate'],
        df['stealth_rate'],
        df['transfer_rate'],
        alpha=0.3,
        label='All Points',
        edgecolors='black',
        linewidths=0.3,
    )

    ax.scatter(
        pareto_df['attack_rate'],
        pareto_df['stealth_rate'],
        pareto_df['transfer_rate'],
        color='red',
        s=80,
        label='Pareto Front (3D)',
    )

    ax.set_xlabel('Attack Rate')
    ax.set_ylabel('Stealth Rate')
    ax.set_zlabel('Transfer Rate')
    ax.set_title('3D Pareto Front (Attack, Stealth, Transfer)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D Pareto front plot saved to: {os.path.abspath(output_path)}")


def create_static_3d_plot(df: pd.DataFrame, plane_coeffs: tuple = None, 
                         plane_model: str = 'z', plane_color: str = 'orange',
                         output_path: str = 'scatter3d.png',
                         figsize: tuple = (15, 12), elev: int = 20, azim: int = 45) -> None:
    """创建静态3D散点图并保存为PNG"""
    fig = plt.figure(figsize=figsize)  # 使用传入的图片尺寸
    ax = fig.add_subplot(projection='3d')
    
    # 为不同组分配不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['group_id'].unique())))
    
    # 绘制散点
    for i, group_id in enumerate(df['group_id'].unique()):
        group_data = df[df['group_id'] == group_id]
        ax.scatter(group_data['attack_rate'], 
                  group_data['stealth_rate'], 
                  group_data['transfer_rate'],
                  c=[colors[i]], 
                  label=f'Group {group_id}',
                  s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # 添加拟合平面
    if plane_coeffs is not None:
        a, b, _, c = plane_coeffs
        xs = df['attack_rate'].to_numpy()
        ys = df['stealth_rate'].to_numpy()
        zs = df['transfer_rate'].to_numpy()
        
        # 创建网格用于绘制平面
        x_range = np.linspace(xs.min(), xs.max(), 20)
        y_range = np.linspace(ys.min(), ys.max(), 20)
        Xg, Yg = np.meshgrid(x_range, y_range)
        Zg = a * Xg + b * Yg + c
        
        ax.plot_surface(Xg, Yg, Zg, alpha=0.3, color=plane_color)
    
    
    
    ax.set_xlabel('Attack Rate', fontsize=12)
    ax.set_ylabel('Stealth Rate', fontsize=12)
    ax.set_zlabel('Transfer Rate', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('3D Scatter Plot (Attack, Stealth, Transfer)', fontsize=14, pad=20)
    
    # 设置观察视角
    ax.view_init(elev=elev, azim=azim)  # elev=仰角(0-90度), azim=方位角(0-360度)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved static 3D scatter plot to: {output_path}")
    plt.close()


def create_interactive_3d_plot(df: pd.DataFrame, plane_coeffs: tuple = None, 
                              plane_model: str = 'z', plane_color: str = 'orange') -> go.Figure:
    """创建交互式3D散点图"""
    fig = go.Figure()
    
    # 添加散点
    for group_id in df['group_id'].unique():
        group_data = df[df['group_id'] == group_id]
        fig.add_trace(go.Scatter3d(
            x=group_data['attack_rate'],
            y=group_data['stealth_rate'],
            z=group_data['transfer_rate'],
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            name=f'Group {group_id}',
            text=[f'Group: {group_id}<br>Attack: {x:.3f}<br>Stealth: {y:.3f}<br>Transfer: {z:.3f}' 
                  for x, y, z in zip(group_data['attack_rate'], group_data['stealth_rate'], group_data['transfer_rate'])],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # 添加拟合平面
    if plane_coeffs is not None:
        # plane_coeffs 是 (a, b, -1, c) 格式，需要转换为 (a, b, c) 格式用于绘制
        a, b, _, c = plane_coeffs  # 忽略第三个参数（-1），使用第四个参数作为常数项
        xs = df['attack_rate'].to_numpy()
        ys = df['stealth_rate'].to_numpy()
        zs = df['transfer_rate'].to_numpy()
        
        # 创建网格用于绘制平面
        x_range = np.linspace(xs.min(), xs.max(), 20)
        y_range = np.linspace(ys.min(), ys.max(), 20)
        Xg, Yg = np.meshgrid(x_range, y_range)
        
        # 最小二乘拟合：z = ax + by + c
        Zg = a * Xg + b * Yg + c
        
        fig.add_trace(go.Surface(
            x=Xg, y=Yg, z=Zg,
            colorscale=[[0, plane_color], [1, plane_color]],
            opacity=0.6,
            name='Linear Fitted Plane',
            showscale=False
        ))
    
    fig.update_layout(
        title='3D Interactive Scatter Plot (Attack, Stealth, Transfer)',
        scene=dict(
            xaxis_title='Attack Rate',
            yaxis_title='Stealth Rate',
            zaxis_title='Transfer Rate',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )
    
    return fig




def main() -> None:
    parser = argparse.ArgumentParser(description='相关性与PCA分析')
    parser.add_argument('--data-csv', type=str, required=True, help='CSV 数据文件路径（如 data_cifar10.csv）')
    parser.add_argument('--standardize', action='store_true', help='在PCA前进行标准化')
    parser.add_argument('--min-attack', type=float, default=0.3, help='筛选阈值：中毒率(attack_rate)低于此值将被剔除')
    parser.add_argument('--group-avg', action='store_true', help='按组平均：先对每组内三列求均值，再基于组均值进行分析')
    parser.add_argument('--output-dir', type=str, default='analysis_outputs', help='所有图表输出目录')
    parser.add_argument('--interactive-out', type=str, default='scatter3d_interactive.html', help='交互式3D散点图输出路径（默认自动生成）')
    parser.add_argument('--static-out', type=str, default='scatter3d.png', help='静态3D散点图输出路径（PNG格式）')
    parser.add_argument('--static-plot', action='store_true', help='生成静态PNG格式的3D散点图')
    parser.add_argument('--fit-plane', action='store_true', help='拟合平面并画在散点图上')
    parser.add_argument('--plane-model', type=str, choices=['z'], default='z',
                        help="平面模型：'z' 表示 z=ax+by+c (最小二乘拟合)")
    parser.add_argument('--plane-color', type=str, default='orange', help='拟合平面颜色')
    parser.add_argument('--plane-alpha', type=float, default=0.35, help='拟合平面透明度')
    parser.add_argument('--correlation-analysis', action='store_true',
                        help='生成相关性热力图与成对散点图（含拟合直线）')
    parser.add_argument('--corr-heatmap-out', type=str, default='correlation_heatmap.png',
                        help='相关性热力图输出路径')
    parser.add_argument('--corr-scatter-prefix', type=str, default='corr_scatter',
                        help='成对散点图文件名前缀')
    parser.add_argument('--pareto-front', action='store_true',
                        help='执行隐蔽性-迁移性帕累托前沿分析')
    parser.add_argument('--pareto-out', type=str, default='pareto_front.png',
                        help='帕累托前沿图输出路径')
    parser.add_argument('--pareto-front-3d', action='store_true',
                        help='同时在攻击/隐蔽/迁移三目标上计算帕累托前沿')
    parser.add_argument('--pareto3d-out', type=str, default='pareto_front_3d.png',
                        help='三目标帕累托前沿图输出路径')
    parser.add_argument('--pareto-front-attack-stealth', action='store_true',
                        help='执行攻击性 vs 隐蔽性的帕累托前沿分析')
    parser.add_argument('--pareto-attack-stealth-out', type=str, default='pareto_front_attack_stealth.png',
                        help='攻击性-隐蔽性帕累托前沿图输出路径')
    parser.add_argument('--pareto-front-attack-transfer', action='store_true',
                        help='执行攻击性 vs 迁移性的帕累托前沿分析')
    parser.add_argument('--pareto-attack-transfer-out', type=str, default='pareto_front_attack_transfer.png',
                        help='攻击性-迁移性帕累托前沿图输出路径')
    parser.add_argument('--violin-plot', action='store_true',
                        help='生成按攻击类型分组的小提琴图（三个子图：attack_rate, stealth_rate, transfer_rate）')
    parser.add_argument('--violin-out', type=str, default='violin_by_attack_type.png',
                        help='小提琴图输出路径')
    args = parser.parse_args()

    # 获取脚本所在目录（analysis文件夹）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从 CSV 文件加载数据
    data_path = args.data_csv
    # 如果路径是相对路径，尝试相对于脚本目录
    if not os.path.isabs(data_path) and not os.path.exists(data_path):
        data_path = os.path.join(script_dir, data_path)
    df = load_all_points_from_csv(data_path)

    # 按中毒率(attack_rate)筛选，剔除低于阈值的数据
    before_n = len(df)
    df = df[df['attack_rate'] >= args.min_attack].reset_index(drop=True)
    after_n = len(df)
    removed_n = before_n - after_n
    print(f"Filter: attack_rate < {args.min_attack:.3f} 被剔除 {removed_n} 条，保留 {after_n}/{before_n} 条。")

    # 对迁移率进行处理：迁移率 = 迁移率 / 攻击率
    # df['transfer_rate'] = df['transfer_rate'] / df['attack_rate']

    # 可选：按组平均
    if args.group_avg:
        print('Note: 按组平均模式，每组内三列取均值后进行分析')
        df = df.groupby('group_id', as_index=False).agg({
            'attack_rate': 'mean',
            'stealth_rate': 'mean',
            'transfer_rate': 'mean',
        }).reset_index(drop=True)
        print(f'Group averages: {len(df)} groups')

    # 打印用于分析的数据
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('Data used for analysis (filtered): group_id, attack_rate, stealth_rate, transfer_rate')
        print(df[['group_id', 'attack_rate', 'stealth_rate', 'transfer_rate']]
              .to_string(index=False, float_format=lambda x: f'{x:.6f}'))


    # PCA
    pca, X_pca, comps = run_pca(df, standardize=args.standardize)
    explained = [f'{v:.6f}' for v in pca.explained_variance_ratio_]
    print('\nPCA explained_variance_ratio_ (PC1, PC2, PC3):')
    print(', '.join(explained))
    print('\nPCA components (rows=PC, cols=features):')
    print(comps.to_string(float_format=lambda x: f'{x:.6f}'))
    # 获取脚本所在目录（analysis文件夹）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 如果输出目录是相对路径，则相对于脚本目录
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def resolve_output_path(path: str) -> str:
        return os.path.join(output_dir, os.path.basename(path))

    heatmap_path = resolve_output_path(args.corr_heatmap_out)
    scatter_prefix = os.path.join(output_dir, os.path.basename(args.corr_scatter_prefix))
    interactive_out_path = resolve_output_path(args.interactive_out)
    static_out_path = resolve_output_path(args.static_out)
    pareto_out_path = resolve_output_path(args.pareto_out)
    pareto3d_out_path = resolve_output_path(args.pareto3d_out)
    pareto_attack_stealth_out_path = resolve_output_path(args.pareto_attack_stealth_out)
    pareto_attack_transfer_out_path = resolve_output_path(args.pareto_attack_transfer_out)
    violin_out_path = resolve_output_path(args.violin_out)

    # 平面拟合参数存储
    plane_coeffs = None
    
    # 最小二乘拟合平面：z = a*x + b*y + c
    if args.fit_plane:
        xs = df['attack_rate'].to_numpy()
        ys = df['stealth_rate'].to_numpy()
        zs = df['transfer_rate'].to_numpy()
        
        # 普通最小二乘：z = a*x + b*y + c
        A = np.c_[xs, ys, np.ones_like(xs)]
        coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
        a, b, c = [float(v) for v in coeffs]
        z_pred = A @ coeffs
        ss_res = float(np.sum((zs - z_pred) ** 2))
        ss_tot = float(np.sum((zs - np.mean(zs)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        print(f"\nFitted plane (z=ax+by+c): z = {a:.6f} * x + {b:.6f} * y + {c:.6f} (R^2={r2:.6f})")
        # 存储平面系数用于距离计算
        plane_coeffs = (a, b, -1, c)  # 转换为 ax + by + cz + d = 0 形式

        # 计算各点到拟合平面的距离
        points = np.c_[xs, ys, zs]
        distances = calculate_point_to_plane_distance(points, plane_coeffs)
        avg_distance = np.mean(distances)
        print(f"\n各点到拟合平面的距离:")
        print(f"平均距离: {avg_distance:.6f}")
        print(f"最大距离: {np.max(distances):.6f}")
        print(f"最小距离: {np.min(distances):.6f}")
        print(f"距离标准差: {np.std(distances):.6f}")
        
        # 将距离信息添加到DataFrame
        df['distance_to_plane'] = distances
    
    # 生成交互式3D散点图
    interactive_fig = create_interactive_3d_plot(df, plane_coeffs, args.plane_model, args.plane_color)
    interactive_fig.write_html(interactive_out_path)
    print(f"Saved interactive 3D scatter to: {interactive_out_path}")
    print(f"💡 提示：在VSCode中右键点击HTML文件，选择'Open with Live Server'或'在浏览器中打开'来查看交互式图表")
    
    # 生成静态3D散点图
    if args.static_plot:
        print(f"\n生成静态3D散点图...")
        create_static_3d_plot(df, plane_coeffs, args.plane_model, args.plane_color, static_out_path)
        print(f"静态图片已保存到: {static_out_path}")

    if args.correlation_analysis:
        print(f"\n{'='*60}")
        print("开始相关性分析")
        print(f"{'='*60}")
        run_correlation_analysis(df, heatmap_path, scatter_prefix)

    if args.pareto_front:
        print(f"\n{'='*60}")
        print("开始帕累托前沿分析（隐蔽性 vs 迁移性）")
        print(f"{'='*60}")
        pareto_df = compute_pareto_front_2d(df, 'stealth_rate', 'transfer_rate')
        if pareto_df.empty:
            print("未找到任何帕累托最优点。")
        else:
            print(f"检测到 {len(pareto_df)} 个帕累托最优点：")
            print(pareto_df[['group_id', 'attack_rate', 'stealth_rate', 'transfer_rate']]
                  .to_string(index=False, float_format=lambda x: f'{x:.6f}'))
            plot_pareto_front_2d(
                df,
                pareto_df,
                'stealth_rate',
                'transfer_rate',
                pareto_out_path,
                'Pareto Front (Stealth vs Transfer)',
            )

    if args.pareto_front_3d:
        print(f"\n{'='*60}")
        print("开始三目标帕累托前沿分析（Attack / Stealth / Transfer）")
        print(f"{'='*60}")
        columns = ['attack_rate', 'stealth_rate', 'transfer_rate']
        pareto3d_df = compute_pareto_front_multi(df, columns)
        if pareto3d_df.empty:
            print("未找到任何三目标帕累托最优点。")
        else:
            print(f"检测到 {len(pareto3d_df)} 个三目标帕累托最优点：")
            print(pareto3d_df[['group_id', 'attack_rate', 'stealth_rate', 'transfer_rate']]
                  .to_string(index=False, float_format=lambda x: f'{x:.6f}'))
            plot_pareto_front_3d(df, pareto3d_df, pareto3d_out_path)

    if args.pareto_front_attack_stealth:
        print(f"\n{'='*60}")
        print("开始帕累托前沿分析（攻击性 vs 隐蔽性）")
        print(f"{'='*60}")
        pareto_as_df = compute_pareto_front_2d(df, 'attack_rate', 'stealth_rate')
        if pareto_as_df.empty:
            print("未找到任何攻击性-隐蔽性帕累托最优点。")
        else:
            print(f"检测到 {len(pareto_as_df)} 个攻击性-隐蔽性帕累托最优点：")
            print(pareto_as_df[['group_id', 'attack_rate', 'stealth_rate', 'transfer_rate']]
                  .to_string(index=False, float_format=lambda x: f'{x:.6f}'))
            plot_pareto_front_2d(
                df,
                pareto_as_df,
                'attack_rate',
                'stealth_rate',
                pareto_attack_stealth_out_path,
                'Pareto Front (Attack vs Stealth)',
            )

    if args.pareto_front_attack_transfer:
        print(f"\n{'='*60}")
        print("开始帕累托前沿分析（攻击性 vs 迁移性）")
        print(f"{'='*60}")
        pareto_at_df = compute_pareto_front_2d(df, 'attack_rate', 'transfer_rate')
        if pareto_at_df.empty:
            print("未找到任何攻击性-迁移性帕累托最优点。")
        else:
            print(f"检测到 {len(pareto_at_df)} 个攻击性-迁移性帕累托最优点：")
            print(pareto_at_df[['group_id', 'attack_rate', 'stealth_rate', 'transfer_rate']]
                  .to_string(index=False, float_format=lambda x: f'{x:.6f}'))
            plot_pareto_front_2d(
                df,
                pareto_at_df,
                'attack_rate',
                'transfer_rate',
                pareto_attack_transfer_out_path,
                'Pareto Front (Attack vs Transfer)',
            )

    if args.violin_plot:
        print(f"\n{'='*60}")
        print("开始生成按攻击类型分组的小提琴图")
        print(f"{'='*60}")
        # 需要确保数据包含 attack_type 列
        if 'attack_type' not in df.columns:
            print("警告: 数据中没有 'attack_type' 列，无法生成小提琴图")
        else:
            plot_violin_by_attack_type(df, violin_out_path)


if __name__ == '__main__':
    main()


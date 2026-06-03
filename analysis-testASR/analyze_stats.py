#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析隐蔽性（TPR平均、AUC平均）与迁移性，按三种模型（resnet18, mobilenet, vgg）分别分析。
支持 CIFAR-10（data_cifar10_*）与 Tiny ImageNet Target-Domain（data_tiny_imagenet_*），见 --dataset。
输出按数据集分目录：{--output-dir}/cifar10/、{--output-dir}/tiny_imagenet/（含 per-arch 图与 combined 散点/箱线等）。

# 1. no_nc 数据：所有图
python analysis/analyze_stats.py --data-dir analysis --data-suffix _no_nc --all-plots --output-dir analysis_outputs_no_nc
# 2. nc 数据：所有图（含 S_stealth）
python analysis/analyze_stats.py --data-dir analysis --data-suffix _nc --all-plots --output-dir analysis_outputs_nc

# 分析 nc 数据（含 S_stealth）
python analysis/analyze_stats.py --data-dir analysis --data-suffix _nc --all-plots

# 只生成整体散点图
python analysis/analyze_stats.py --data-dir analysis --data-suffix _nc --combined-scatter --output-dir analysis_outputs_nc

# 指定输出目录
python analysis/analyze_stats.py --data-dir analysis --data-suffix _nc --output-dir analysis_outputs_nc

# 只分析某个 arch
python analysis/analyze_stats.py --data-dir analysis --arch mobilenet --all-plots

# Tiny ImageNet Target-Domain（结果在 analysis_outputs_no_nc/tiny_imagenet/）
python analysis/analyze_stats.py --dataset tiny_imagenet --data-dir analysis --data-suffix _no_nc --all-plots --output-dir analysis_outputs_no_nc

# 同时跑 CIFAR-10 与 Tiny ImageNet Target-Domain（输出到 analysis_outputs_no_nc/cifar10 与 .../tiny_imagenet）
python analysis/analyze_stats.py --dataset all --data-suffix _no_nc --all-plots --output-dir analysis_outputs_no_nc

# 指定部分图类型
python analysis/analyze_stats.py --data-dir analysis --violin-plot --box-plot --combined-scatter

图表类型：
  - 原有：violin, correlation(热力图+散点), pareto
  - 新增：box(箱线图), bar(柱状图),
         scatter-3d(三维散点),
         box-poison(按poison_rate), box-trigger(按trigger_type),
         line-plot(折线图，每个方法一张),
         combined-scatter(整体散点图：合并所有arch，形状=arch，颜色=ASR)
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARCHS = ['resnet18', 'mobilenet', 'vgg']
DATASETS = ['cifar10', 'tiny_imagenet', 'mnistm']
DATASET_DISPLAY = {
    'cifar10': 'CIFAR-10',
    'tiny_imagenet': 'Tiny ImageNet -> Target Domain',
    'mnistm': 'MNIST-M',
}
STEALTH_COLS = ['stealth_tpr_avg', 'stealth_auc_avg']
METRIC_COLS = STEALTH_COLS + ['transfer_rate']
# 与 extract 一致：stealth_* = 1 - 各防御原始 TPR/AUC 的均值，越大隐蔽性越好
LABEL_STEALTH_TPR = 'Stealth (1-TPR)'
LABEL_STEALTH_AUC = 'Stealth (1-AUC)'
METRIC_LABELS = {
    'stealth_tpr_avg': LABEL_STEALTH_TPR,
    'stealth_auc_avg': LABEL_STEALTH_AUC,
    'transfer_rate': 'Transfer Rate',
}

# 比率类指标以 [0,1] 为有效范围，坐标轴略扩边以免贴边难看
RATE_AXIS_PAD = 0.05
RATE_LIM = (-RATE_AXIS_PAD, 1.0 + RATE_AXIS_PAD)
# 色条 / 归一化仍用严格 [0,1]，语义不变
RATE_NORM_LIM = (0.0, 1.0)


def _xlim_rate(ax) -> None:
    ax.set_xlim(*RATE_LIM)


def _ylim_rate(ax) -> None:
    ax.set_ylim(*RATE_LIM)


def _set_xy_rate(ax) -> None:
    ax.set_xlim(*RATE_LIM)
    ax.set_ylim(*RATE_LIM)


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """从 CSV 加载数据，要求含 stealth_tpr_avg, stealth_auc_avg, transfer_rate。
    若含 S_stealth / S_stealth_tpr 列则保留，供 nc 模式分析使用。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")
    extra = ['asr', 'S_stealth', 'S_stealth_tpr', 'nc_max_anomaly_index', 'nc_is_poisoned']
    for col in METRIC_COLS + [c for c in extra if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # ASR 若以百分数存储，缩放到 [0,1] 以便与坐标轴/色条一致
    if 'asr' in df.columns:
        asr_s = df['asr'].dropna()
        if len(asr_s) and asr_s.max() > 1.5:
            df['asr'] = df['asr'] / 100.0
    df = df.dropna(subset=METRIC_COLS)
    if 'attack_type' in df.columns:
        df = df[~df['attack_type'].isin(['none', ''])]
    if 'group_id' not in df.columns:
        df['group_id'] = range(1, len(df) + 1)
    return df


def discover_arch_csvs(data_dir: str, suffix: str = '', dataset: str = 'cifar10') -> dict:
    """发现各 arch 的 CSV 文件，返回 {arch: path}。
    suffix: '' 表示 data_{dataset}_{arch}.csv；'_no_nc' / '_nc' 表示带后缀的文件。
    """
    found = {}
    for arch in ARCHS:
        path = os.path.join(data_dir, f'data_{dataset}_{arch}{suffix}.csv')
        if os.path.exists(path):
            found[arch] = path
    return found


def plot_violin_by_attack_type(df: pd.DataFrame, output_path: str, arch: str) -> None:
    """按攻击类型绘制三种指标的小提琴图：stealth_tpr_avg, stealth_auc_avg, transfer_rate"""
    if df.empty or 'attack_type' not in df.columns:
        return
    attack_types = sorted(df['attack_type'].unique())
    metrics = [
        ('stealth_tpr_avg', LABEL_STEALTH_TPR),
        ('stealth_auc_avg', LABEL_STEALTH_AUC),
        ('transfer_rate', 'Transfer Rate'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Metrics by Attack Type ({arch})', fontsize=16, fontweight='bold', y=1.02)
    base_color = '#4A90E2'
    for idx, (col, name) in enumerate(metrics):
        ax = axes[idx]
        data_by_type = [df[df['attack_type'] == at][col].dropna().values for at in attack_types]
        parts = ax.violinplot(data_by_type, positions=range(len(attack_types)),
                              showmeans=True, showmedians=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(base_color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(3.0)
        parts['cmeans'].set_color('darkred')
        ax.set_xticks(range(len(attack_types)))
        ax.set_xticklabels(attack_types, rotation=45, ha='right')
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} by Attack Type', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        _ylim_rate(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  小提琴图: {output_path}")


def run_correlation_analysis(df: pd.DataFrame, output_dir: str, arch: str) -> None:
    """相关性热力图 + 成对散点图"""
    if df.empty:
        return
    corr = df[METRIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    # 相关系数线性映射到 [0,1]，色条与坐标统一为 0–1
    corr01 = (corr.values + 1.0) / 2.0
    im = ax.imshow(corr01, cmap='coolwarm', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation ((ρ+1)/2)')
    labels = [LABEL_STEALTH_TPR, LABEL_STEALTH_AUC, 'Transfer Rate']
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr.iloc[i, j]:.3f}', ha='center', va='center', fontsize=9)
    ax.set_title(f'Correlation ({arch})')
    plt.tight_layout()
    path = os.path.join(output_dir, f'correlation_heatmap_{arch}.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  相关性热力图: {path}")

    # 成对散点：stealth_tpr vs transfer, stealth_auc vs transfer；颜色=ASR(0=白)，形状=attack_type
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    attack_markers = {'SIG': '^', 'WaNet': 'o', 'adaptive_blend': 's', 'adaptive_patch': 'D',
                     'basic': 'p', 'belt': 'h', 'blend': 'v', 'badnet': '*', 'upgd': 'X'}
    for stealth_col, x_label, name_key in [
        ('stealth_tpr_avg', LABEL_STEALTH_TPR, 'tpr'),
        ('stealth_auc_avg', LABEL_STEALTH_AUC, 'auc'),
    ]:
        sub = df.dropna(subset=[stealth_col, 'transfer_rate'])
        if len(sub) < 2:
            continue
        x, y = sub[stealth_col].to_numpy(), sub['transfer_rate'].to_numpy()
        asr_vals = sub['asr'].fillna(0) if 'asr' in sub.columns else pd.Series([0] * len(sub))
        if len(asr_vals) and asr_vals.max() > 1.5:
            asr_vals = asr_vals / 100.0
        asr_min, asr_max = 0.0, 1.0
        slope, intercept = np.polyfit(x, y, 1)
        lx = np.linspace(0, 1, 100)
        ly = np.clip(slope * lx + intercept, 0, 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        add_method_parallelograms(ax, sub, stealth_col, 'transfer_rate')
        for at in sub['attack_type'].unique():
            mask = sub['attack_type'] == at
            if mask.sum() == 0:
                continue
            ax.scatter(sub.loc[mask, stealth_col], sub.loc[mask, 'transfer_rate'],
                       c=asr_vals[mask], cmap='Blues', vmin=asr_min, vmax=asr_max,
                       marker=attack_markers.get(at, 'o'), s=60, edgecolors='none', zorder=2.5)
        ax.plot(lx, ly, 'gray', lw=1.5, alpha=0.7, zorder=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Transfer Rate')
        ax.set_title(f'{x_label} vs Transfer ({arch})')
        _set_xy_rate(ax)
        sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=asr_min, vmax=asr_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('ASR')
        leg_handles = [Line2D([0], [0], marker=attack_markers.get(a, 'o'), color='w',
                              markerfacecolor='#4A90E2', markersize=10, label=a, markeredgecolor='none')
                      for a in sorted(sub['attack_type'].unique())]
        leg_handles.append(Line2D([0], [0], color='gray', lw=2, label=f'y={slope:.3f}x+{intercept:.3f}'))
        ax.legend(handles=leg_handles, fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f'scatter_stealth_{name_key}_vs_transfer_{arch}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  散点图: {path}")


def compute_pareto_2d(df: pd.DataFrame, x_col: str, y_col: str,
                      x_ascending: bool = False, y_ascending: bool = False) -> pd.DataFrame:
    """二维帕累托前沿。默认两者越大越优；x_ascending=True 表示 x 越小越优"""
    if df.empty:
        return pd.DataFrame()
    working = df.copy()
    working = working.sort_values(by=[x_col, y_col], ascending=[x_ascending, y_ascending], ignore_index=True)
    front_rows = []
    best_y = -np.inf if not y_ascending else np.inf
    for _, row in working.iterrows():
        yv = row[y_col]
        if y_ascending:
            better = yv <= best_y
            if better:
                best_y = min(best_y, yv) if best_y != np.inf else yv
        else:
            better = yv >= best_y
            if better:
                best_y = max(best_y, yv)
        if better:
            front_rows.append(row)
    if not front_rows:
        return pd.DataFrame()
    return pd.DataFrame(front_rows).drop_duplicates(subset=[x_col, y_col]).reset_index(drop=True)


def plot_pareto_2d(df: pd.DataFrame, pareto_df: pd.DataFrame, x_col: str, y_col: str,
                   output_path: str, title: str, x_label: str = None, y_label: str = None) -> None:
    if df.empty or pareto_df.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.4, label='All', edgecolors='black', linewidths=0.3)
    plt.scatter(pareto_df[x_col], pareto_df[y_col], color='red', s=60, label='Pareto Front')
    sorted_df = pareto_df.sort_values(x_col)
    plt.plot(sorted_df[x_col], sorted_df[y_col], 'r-', lw=2)
    plt.xlabel(x_label or x_col.replace('_', ' ').title())
    plt.ylabel(y_label or y_col.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.xlim(*RATE_LIM)
    plt.ylim(*RATE_LIM)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  帕累托图: {output_path}")


def plot_box_by_attack_type(df: pd.DataFrame, output_path: str, arch: str) -> None:
    """按攻击类型绘制三种指标的箱线图"""
    if df.empty or 'attack_type' not in df.columns:
        return
    attack_types = sorted(df['attack_type'].unique())
    metrics = [
        ('stealth_tpr_avg', LABEL_STEALTH_TPR),
        ('stealth_auc_avg', LABEL_STEALTH_AUC),
        ('transfer_rate', 'Transfer Rate'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Box Plot by Attack Type ({arch})', fontsize=16, fontweight='bold', y=1.02)
    for idx, (col, name) in enumerate(metrics):
        ax = axes[idx]
        data_by_type = [df[df['attack_type'] == at][col].dropna().values for at in attack_types]
        bp = ax.boxplot(data_by_type, positions=range(len(attack_types)), patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor('#4A90E2')
            patch.set_alpha(0.7)
        ax.set_xticks(range(len(attack_types)))
        ax.set_xticklabels(attack_types, rotation=45, ha='right')
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} by Attack Type', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        _ylim_rate(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  箱线图: {output_path}")


def plot_box_by_attack_type_combined_all_archs(
    data_dir: str, output_dir: str, suffix: str = '', dataset: str = 'cifar10'
) -> None:
    """合并所有 arch 的数据，按 attack_type 绘制三种指标箱线图。"""
    found = discover_arch_csvs(data_dir, suffix=suffix, dataset=dataset)
    if len(found) < 2:
        return
    dfs = []
    for arch, path in found.items():
        df = load_from_csv(path).copy()
        if df.empty:
            continue
        df['arch'] = arch
        dfs.append(df)
    if not dfs:
        return
    combined = pd.concat(dfs, ignore_index=True)
    if combined.empty or 'attack_type' not in combined.columns:
        return
    attack_types = sorted(combined['attack_type'].dropna().unique())
    if not attack_types:
        return
    metrics = [
        ('stealth_tpr_avg', LABEL_STEALTH_TPR),
        ('stealth_auc_avg', LABEL_STEALTH_AUC),
        ('transfer_rate', 'Transfer Rate'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ds_label = DATASET_DISPLAY.get(dataset, dataset)
    fig.suptitle(f'Box Plot by Attack Type (All Archs Combined, {ds_label})', fontsize=16, fontweight='bold', y=1.02)
    for idx, (col, name) in enumerate(metrics):
        ax = axes[idx]
        data_by_type = [combined[combined['attack_type'] == at][col].dropna().values for at in attack_types]
        bp = ax.boxplot(data_by_type, positions=range(len(attack_types)), patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor('#4A90E2')
            patch.set_alpha(0.7)
        ax.set_xticks(range(len(attack_types)))
        ax.set_xticklabels(attack_types, rotation=45, ha='right')
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} by Attack Type', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        _ylim_rate(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(output_dir, 'box_by_attack_type_combined.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  箱线图(全部数据): {path}")


def plot_bar_by_attack_type(df: pd.DataFrame, output_path: str, arch: str) -> None:
    """按攻击类型的均值柱状图（带误差条）"""
    if df.empty or 'attack_type' not in df.columns:
        return
    agg = df.groupby('attack_type')[METRIC_COLS].agg(['mean', 'std']).reset_index()
    attack_types = agg['attack_type'].tolist()
    x = np.arange(len(attack_types))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, col in enumerate(METRIC_COLS):
        means = agg[(col, 'mean')].values
        stds = agg[(col, 'std')].fillna(0).values
        ax.bar(x + i * width, means, width, label=METRIC_LABELS.get(col, col.replace('_', ' ').title()), yerr=stds, capsize=3)
    ax.set_xticks(x + width)
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'Mean Metrics by Attack Type ({arch})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _ylim_rate(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  柱状图: {output_path}")


def plot_3d_scatter(df: pd.DataFrame, output_path: str, arch: str) -> None:
    """三维散点图：stealth_tpr, stealth_auc, transfer_rate"""
    if df.empty or len(df) < 3:
        return
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['stealth_tpr_avg'], df['stealth_auc_avg'], df['transfer_rate'],
                         c=df['transfer_rate'], cmap='viridis', alpha=0.6, s=30)
    ax.set_xlabel(LABEL_STEALTH_TPR)
    ax.set_ylabel(LABEL_STEALTH_AUC)
    ax.set_zlabel('Transfer Rate')
    ax.set_title(f'3D: {LABEL_STEALTH_TPR} vs {LABEL_STEALTH_AUC} vs Transfer ({arch})')
    ax.set_xlim(*RATE_LIM)
    ax.set_ylim(*RATE_LIM)
    ax.set_zlim(*RATE_LIM)
    scatter.set_clim(*RATE_NORM_LIM)
    plt.colorbar(scatter, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  三维散点图: {output_path}")


def plot_box_by_poison_rate(df: pd.DataFrame, output_path: str, arch: str) -> None:
    """按 poison_rate 分组的箱线图"""
    if df.empty or 'poison_rate' not in df.columns:
        return
    df_plot = df.copy()
    df_plot['poison_rate'] = df_plot['poison_rate'].astype(str)
    try:
        rates = sorted(df_plot['poison_rate'].unique(), key=lambda x: float(x))
    except (ValueError, TypeError):
        rates = sorted(df_plot['poison_rate'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Box Plot by Poison Rate ({arch})', fontsize=16, fontweight='bold', y=1.02)
    for idx, col in enumerate(METRIC_COLS):
        ax = axes[idx]
        data_by_rate = [df_plot[df_plot['poison_rate'] == r][col].dropna().values for r in rates]
        bp = ax.boxplot(data_by_rate, positions=range(len(rates)), patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor('#E27D60')
            patch.set_alpha(0.7)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(rates, rotation=45, ha='right')
        ax.set_ylabel(METRIC_LABELS.get(col, col.replace('_', ' ').title()))
        ax.grid(True, alpha=0.3, axis='y')
        _ylim_rate(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  箱线图(按poison_rate): {output_path}")


def plot_box_by_trigger_type(df: pd.DataFrame, output_path: str, arch: str) -> None:
    """按 trigger_type 分组的箱线图"""
    if df.empty or 'trigger_type' not in df.columns:
        return
    trigger_types = sorted(df['trigger_type'].dropna().unique())
    if not trigger_types:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Box Plot by Trigger Type ({arch})', fontsize=16, fontweight='bold', y=1.02)
    for idx, col in enumerate(METRIC_COLS):
        ax = axes[idx]
        data_by_type = [df[df['trigger_type'] == t][col].dropna().values for t in trigger_types]
        bp = ax.boxplot(data_by_type, positions=range(len(trigger_types)), patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor('#85CDCA')
            patch.set_alpha(0.7)
        ax.set_xticks(range(len(trigger_types)))
        ax.set_xticklabels(trigger_types, rotation=45, ha='right')
        ax.set_ylabel(METRIC_LABELS.get(col, col.replace('_', ' ').title()))
        ax.grid(True, alpha=0.3, axis='y')
        _ylim_rate(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  箱线图(按trigger_type): {output_path}")


# 各攻击类型的触发器强度参数名（用于折线图横坐标）
TRIGGER_PARAM_LABEL = {
    'WaNet': 's', 'SIG': 'delta', 'basic': 'alpha', 'blend': 'alpha',
    'adaptive_blend': 'alpha', 'adaptive_patch': 'alpha', 'badnet': 'alpha',
    'upgd': 'eps', 'belt': 'mask_rate',
}


# 折线图：不同 poison_rate 用不同色系，点颜色=ASR(0=白)
POISON_RATE_CMAPS = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'YlOrBr', 'YlGnBu', 'PuBu']
POISON_RATE_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']


def plot_line_by_attack_type(df: pd.DataFrame, output_dir: str, arch: str) -> None:
    """每个攻击类型单独画折线图：X=触发器强度，不同中毒率不同色系+形状，点颜色=ASR(0=白)"""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    if df.empty or 'attack_type' not in df.columns:
        return
    df = df.copy()
    for col in ['poison_rate', 'train_param_value', 'asr']:
        if col not in df.columns:
            df[col] = np.nan if col == 'asr' else 0
    df['asr'] = pd.to_numeric(df['asr'], errors='coerce').fillna(0)
    attack_types = sorted(df['attack_type'].unique())
    for at in attack_types:
        sub = df[df['attack_type'] == at].copy()
        if sub.empty:
            continue
        sub = sub.dropna(subset=['train_param_value'])
        if sub.empty:
            continue
        x_param = TRIGGER_PARAM_LABEL.get(at, 'param')
        poison_rates = sorted(sub['poison_rate'].dropna().unique(), key=lambda x: float(x) if pd.notna(x) else 0)
        asr_max = 1.0
        if sub['asr'].max() > 1.5:
            sub = sub.copy()
            sub['asr'] = sub['asr'] / 100.0
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f'Line Chart: {at} ({arch}) — X={x_param} (Trigger Strength)', fontsize=16, fontweight='bold', y=1.02)
        metrics = [
            ('stealth_tpr_avg', LABEL_STEALTH_TPR),
            ('stealth_auc_avg', LABEL_STEALTH_AUC),
            ('transfer_rate', 'Transfer Rate'),
        ]
        for idx, (col, name) in enumerate(metrics):
            ax = axes[idx]
            for pr_idx, pr in enumerate(poison_rates):
                pr_sub = sub[sub['poison_rate'] == pr].sort_values('train_param_value')
                if pr_sub.empty:
                    continue
                x_vals = pr_sub['train_param_value'].values
                y_vals = pr_sub[col].values
                asr_vals = pr_sub['asr'].values
                cmap_name = POISON_RATE_CMAPS[pr_idx % len(POISON_RATE_CMAPS)]
                marker = POISON_RATE_MARKERS[pr_idx % len(POISON_RATE_MARKERS)]
                ax.scatter(x_vals, y_vals, c=asr_vals, cmap=cmap_name, vmin=0, vmax=1.0,
                           marker=marker, s=70, edgecolors='none')
                ax.plot(x_vals, y_vals, '-', color='gray', linewidth=1.5, alpha=0.5)
            ax.set_xlabel(f'{x_param} (Trigger Strength)')
            ax.set_ylabel(name)
            ax.set_title(name)
            _ylim_rate(ax)
            sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=0, vmax=1.0))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('ASR')
            leg_handles = [Line2D([0], [0], marker=POISON_RATE_MARKERS[i % len(POISON_RATE_MARKERS)],
                                  color='w', markerfacecolor='#4A90E2', markersize=10, label=f'poison_rate={pr}',
                                  markeredgecolor='none') for i, pr in enumerate(poison_rates)]
            ax.legend(handles=leg_handles, loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(output_dir, f'line_{at}_{arch}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  折线图({at}): {path}")


def run_pareto_analysis(df: pd.DataFrame, output_dir: str, arch: str) -> None:
    """隐蔽性 vs 迁移性 帕累托。stealth_tpr_avg / stealth_auc_avg 均为 1-原始均值，越大越隐蔽。"""
    if df.empty:
        return
    df_temp = df.copy()
    pareto_tpr = compute_pareto_2d(df_temp, 'stealth_tpr_avg', 'transfer_rate')
    if not pareto_tpr.empty:
        plot_pareto_2d(
            df_temp, pareto_tpr,
            'stealth_tpr_avg', 'transfer_rate',
            os.path.join(output_dir, f'pareto_stealth_tpr_vs_transfer_{arch}.png'),
            f'Pareto: {LABEL_STEALTH_TPR} vs Transfer ({arch})',
            x_label=LABEL_STEALTH_TPR, y_label='Transfer Rate'
        )
    pareto_auc = compute_pareto_2d(df_temp, 'stealth_auc_avg', 'transfer_rate')
    if not pareto_auc.empty:
        plot_pareto_2d(
            df_temp, pareto_auc,
            'stealth_auc_avg', 'transfer_rate',
            os.path.join(output_dir, f'pareto_stealth_auc_vs_transfer_{arch}.png'),
            f'Pareto: {LABEL_STEALTH_AUC} vs Transfer ({arch})',
            x_label=LABEL_STEALTH_AUC, y_label='Transfer Rate'
        )


# 整体图：arch 对应标记形状与图例颜色（统一蓝色）
ARCH_MARKERS = {'resnet18': '^', 'mobilenet': 'o', 'vgg': 's'}
ARCH_LEGEND_COLOR = '#4A90E2'

# 攻击方法编号说明（8种方法，用于总图）
ATTACK_TYPE_ORDER = [
    'blend', 'basic', 'SIG', 'WaNet', 'adaptive_blend', 'adaptive_patch', 'belt', 'upgd'
]


def _method_envelope_rgba() -> dict:
    """与气泡图一致：每种攻击方法一种颜色，用于包围平行四边形描边/填充。"""
    arr = plt.cm.tab10(np.linspace(0, 1, 10))[: len(ATTACK_TYPE_ORDER)]
    return {at: arr[i] for i, at in enumerate(ATTACK_TYPE_ORDER)}


def axis_aligned_bbox_corners_xy(x: np.ndarray, y: np.ndarray, pad_frac: float = 0.02):
    """轴对齐矩形的四个顶点 (4,2)；作退化/数值失败时的回退。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n == 0:
        return None
    span_axis = RATE_LIM[1] - RATE_LIM[0]
    if n == 1:
        pad = max(span_axis * pad_frac, 0.012)
        xmin, xmax = x[0] - pad, x[0] + pad
        ymin, ymax = y[0] - pad, y[0] + pad
    else:
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        dx = (xmax - xmin) * pad_frac + 1e-9
        dy = (ymax - ymin) * pad_frac + 1e-9
        xmin -= dx
        xmax += dx
        ymin -= dy
        ymax += dy
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)
    corners[:, 0] = np.clip(corners[:, 0], RATE_LIM[0], RATE_LIM[1])
    corners[:, 1] = np.clip(corners[:, 1], RATE_LIM[0], RATE_LIM[1])
    return corners


def tradeoff_parallelogram_xy(
    x: np.ndarray,
    y: np.ndarray,
    pad_frac: float = 0.02,
    *,
    intercept_pct_low: float = 5.0,
    intercept_pct_high: float = 95.0,
    x_pad: float = 0.02,
):
    """Linear Regression Corridor：OLS 斜率 k，截距带用分位数抗离群；左右各扩固定 x_pad。

    1) 一元线性回归 Y = k*X + b（k、b 为 OLS）；上下边斜率均为 k。
    2) b_i = y_i - k*x_i；b_max = percentile(b_i, intercept_pct_high)，b_min = percentile(b_i, intercept_pct_low)。
    3) x_left = min(X) - x_pad，x_right = max(X) + x_pad，再按 RATE_LIM 裁剪 x 后算顶点 y。
    4) 逆时针顶点（理论几何）：(X_min, k·X_min+b_max) → (X_max, k·X_max+b_max) →
       (X_max, k·X_max+b_min) → (X_min, k·X_min+b_min)。上下边与 OLS 线 Y=k·X+b 斜率同为 k。

    返回 (corners, k, b, b_max, b_min, x_min, x_max)。主情形 corners 的 y 不裁剪，便于与平行线一致；
    退化情形 b_max/b_min/x_min/x_max 为 nan，仅用 corners 画多边形。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n == 0:
        return None

    span_axis = RATE_LIM[1] - RATE_LIM[0]

    if n == 1:
        xm, ym = float(x[0]), float(y[0])
        pad_x = max(x_pad, span_axis * pad_frac, 0.012)
        pad_y = max(span_axis * pad_frac, 0.012)
        x_min = np.clip(xm - pad_x, RATE_LIM[0], RATE_LIM[1])
        x_max = np.clip(xm + pad_x, RATE_LIM[0], RATE_LIM[1])
        if x_min >= x_max:
            x_min, x_max = RATE_LIM[0], RATE_LIM[1]
        corners = np.array(
            [
                [x_min, ym + pad_y],
                [x_max, ym + pad_y],
                [x_max, ym - pad_y],
                [x_min, ym - pad_y],
            ],
            dtype=float,
        )
        corners[:, 1] = np.clip(corners[:, 1], RATE_LIM[0], RATE_LIM[1])
        nan4 = (np.nan, np.nan, np.nan, np.nan)
        return corners, 0.0, ym, *nan4

    x_range = float(x.max() - x.min())
    if x_range < 1e-12:
        c = axis_aligned_bbox_corners_xy(x, y, pad_frac=pad_frac)
        nan4 = (np.nan, np.nan, np.nan, np.nan)
        return c, 0.0, float(np.mean(y)), *nan4

    coef = np.polyfit(x, y, 1)
    k, b = float(coef[0]), float(coef[1])
    b_i = y - k * x
    b_max = float(np.percentile(b_i, intercept_pct_high))
    b_min = float(np.percentile(b_i, intercept_pct_low))
    if b_max <= b_min:
        mid = 0.5 * (b_max + b_min)
        eps = max(1e-9, span_axis * 1e-6)
        b_max, b_min = mid + eps, mid - eps

    x_min = float(np.min(x)) - x_pad
    x_max = float(np.max(x)) + x_pad
    x_min = float(np.clip(x_min, RATE_LIM[0], RATE_LIM[1]))
    x_max = float(np.clip(x_max, RATE_LIM[0], RATE_LIM[1]))
    if x_min >= x_max:
        x_min, x_max = float(RATE_LIM[0]), float(RATE_LIM[1])

    # y 不 clip：保证 Y=kX+b_max / kX+b_min 与 Y=kX+b 严格平行；作图时由坐标轴裁切
    corners = np.array(
        [
            [x_min, k * x_min + b_max],
            [x_max, k * x_max + b_max],
            [x_max, k * x_max + b_min],
            [x_min, k * x_min + b_min],
        ],
        dtype=float,
    )
    return corners, k, b, b_max, b_min, x_min, x_max


def add_method_parallelograms(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str = 'transfer_rate',
    *,
    attack_col: str = 'attack_type',
    zorder: float = 0.5,
    pad_frac: float = 0.02,
    face_alpha: float = 0.14,
    edge_lw: float = 1.65,
    intercept_pct_low: float = 5.0,
    intercept_pct_high: float = 95.0,
    x_pad: float = 0.02,
) -> None:
    """每种攻击方法：上下界为 Y=kX+b_max / kX+b_min，与 OLS 拟合线 Y=kX+b 平行；保留中间拟合线。"""
    from matplotlib.patches import Polygon

    if df.empty or attack_col not in df.columns:
        return
    colors = _method_envelope_rgba()
    for at in ATTACK_TYPE_ORDER:
        g = df[df[attack_col] == at].dropna(subset=[x_col, y_col])
        if len(g) < 1:
            continue
        band = tradeoff_parallelogram_xy(
            g[x_col].to_numpy(),
            g[y_col].to_numpy(),
            pad_frac=pad_frac,
            intercept_pct_low=intercept_pct_low,
            intercept_pct_high=intercept_pct_high,
            x_pad=x_pad,
        )
        if band is None:
            continue
        corners, k, b, b_max, b_min, x_lo, x_hi = band
        rgba = colors.get(at, (0.5, 0.5, 0.5, 1.0))
        rgb = tuple(rgba[:3])
        line_kw = dict(color=rgb, solid_capstyle='round', zorder=zorder + 0.15)

        if np.isfinite(b_max) and np.isfinite(b_min) and np.isfinite(x_lo) and np.isfinite(x_hi):
            xspan_ax = RATE_LIM[1] - RATE_LIM[0]
            npt = max(50, int(200 * (x_hi - x_lo) / max(xspan_ax, 1e-9)))
            xs = np.linspace(x_lo, x_hi, npt)
            y_top = k * xs + b_max
            y_bot = k * xs + b_min
            y_mid = k * xs + b
            ax.fill_between(
                xs,
                y_bot,
                y_top,
                facecolor=rgb,
                alpha=face_alpha,
                edgecolor='none',
                zorder=zorder,
            )
            ax.plot(xs, y_top, lw=max(1.0, edge_lw * 0.9), ls='-', alpha=0.9, **line_kw)
            ax.plot(xs, y_bot, lw=max(1.0, edge_lw * 0.9), ls='-', alpha=0.9, **line_kw)
            ax.plot(
                xs,
                y_mid,
                lw=max(1.15, edge_lw * 0.95),
                ls='-',
                alpha=0.98,
                **line_kw,
            )
        else:
            poly = Polygon(
                corners,
                closed=True,
                facecolor=rgb,
                alpha=face_alpha,
                edgecolor=rgb,
                linewidth=edge_lw,
                zorder=zorder,
            )
            ax.add_patch(poly)
            xa, xb = float(corners[0, 0]), float(corners[1, 0])
            if abs(xb - xa) < 1e-9:
                xs_line = np.linspace(RATE_LIM[0], RATE_LIM[1], 200)
            else:
                xs_line = np.linspace(xa, xb, 200)
            ax.plot(
                xs_line,
                k * xs_line + b,
                lw=max(1.0, edge_lw * 0.85),
                ls='-',
                **line_kw,
            )


def plot_scatter_combined_all_archs(
    data_dir: str, output_dir: str, suffix: str = '', dataset: str = 'cifar10'
) -> None:
    """合并某数据集下所有 arch：stealth_auc/stealth_tpr vs transfer，形状区分 arch，颜色表示 ASR"""
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    found = discover_arch_csvs(data_dir, suffix=suffix, dataset=dataset)
    if len(found) < 2:
        return
    dfs = []
    for arch, path in found.items():
        df = pd.read_csv(path, encoding='utf-8')
        for col in ['stealth_tpr_avg', 'stealth_auc_avg', 'transfer_rate', 'asr', 'S_stealth', 'S_stealth_tpr']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['arch'] = arch
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    if combined.empty:
        return
    if 'attack_type' in combined.columns:
        combined = combined[~combined['attack_type'].isin(['none', ''])]
    if combined.empty:
        return
    if 'asr' in combined.columns:
        av = combined['asr'].dropna()
        if len(av) and av.max() > 1.5:
            combined['asr'] = combined['asr'] / 100.0
    asr_min, asr_max = 0.0, 1.0

    plot_cols = [
        ('stealth_auc_avg', LABEL_STEALTH_AUC, 'auc'),
        ('stealth_tpr_avg', LABEL_STEALTH_TPR, 'tpr'),
    ]
    if 'S_stealth' in combined.columns:
        plot_cols.append(('S_stealth', 'S_stealth (NC, AUC)', 's_stealth_auc'))
    if 'S_stealth_tpr' in combined.columns:
        plot_cols.append(('S_stealth_tpr', 'S_stealth (NC, TPR)', 's_stealth_tpr'))
    # 方法编号映射
    method_to_num = {at: i + 1 for i, at in enumerate(ATTACK_TYPE_ORDER)}

    for x_col, x_label, file_suffix in plot_cols:
        sub_combined = combined.dropna(subset=[x_col, 'transfer_rate'])
        if sub_combined.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.subplots_adjust(left=0.1, right=0.65, top=0.92, bottom=0.1)
        add_method_parallelograms(ax, sub_combined, x_col, 'transfer_rate')
        for arch in ARCHS:
            if arch not in found:
                continue
            sub = sub_combined[sub_combined['arch'] == arch]
            if sub.empty:
                continue
            x = sub[x_col].values
            y = sub['transfer_rate'].values
            c = sub['asr'].fillna(asr_min + (asr_max - asr_min) / 2).values
            attack_types = sub['attack_type'].values if 'attack_type' in sub.columns else [''] * len(x)
            ax.scatter(x, y, c=c, cmap='Blues', vmin=asr_min, vmax=asr_max,
                       marker=ARCH_MARKERS.get(arch, 'o'), s=80, edgecolors='none', zorder=2.5)
            for xi, yi, at in zip(x, y, attack_types):
                num = method_to_num.get(at, '')
                if num:
                    ax.annotate(str(num), (xi, yi), fontsize=5, ha='center', va='center',
                               color='gray', alpha=0.8, zorder=3)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Transfer Rate')
        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        ax.set_title(f'{ds_label}: {x_label} vs Transfer (shape=arch, color=ASR)')
        legend_elements = [Line2D([0], [0], marker=ARCH_MARKERS.get(a, 'o'), color='w',
                                  markerfacecolor=ARCH_LEGEND_COLOR, markersize=12, label=a,
                                  markeredgecolor='none') for a in ARCHS if a in found]
        ax.legend(handles=legend_elements, loc='upper right', frameon=False)

        # colorbar 与 方法说明 放在图外右侧
        sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=asr_min, vmax=asr_max))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.72, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('ASR')
        method_lines = [f"{i+1}={at}" for i, at in enumerate(ATTACK_TYPE_ORDER)]
        method_text = 'Method:\n' + '\n'.join(method_lines)
        fig.text(0.78, 0.5, method_text, fontsize=8, va='center', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.grid(True, alpha=0.3)
        _set_xy_rate(ax)
        path = os.path.join(output_dir, f'scatter_stealth_{file_suffix}_vs_transfer_combined.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  整体散点图(Stealth {file_suffix.upper()}): {path}")

    # 额外：按方法分图，每个方法一张，形状=arch
    _plot_scatter_by_method(combined, found, output_dir, asr_min, asr_max, dataset=dataset)

    # 额外：气泡图，大小=ASR，颜色=方法，形状=arch
    _plot_scatter_combined_bubble(combined, found, output_dir, dataset=dataset)


def _plot_scatter_by_method(combined, found, output_dir, asr_min, asr_max, dataset: str = 'cifar10'):
    """按方法分图：每个方法一张，形状=arch，颜色=ASR"""
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plot_cols = [
        ('stealth_auc_avg', LABEL_STEALTH_AUC, 'auc'),
        ('stealth_tpr_avg', LABEL_STEALTH_TPR, 'tpr'),
    ]
    if 'S_stealth' in combined.columns:
        plot_cols.append(('S_stealth', 'S_stealth (NC, AUC)', 's_stealth_auc'))
    if 'S_stealth_tpr' in combined.columns:
        plot_cols.append(('S_stealth_tpr', 'S_stealth (NC, TPR)', 's_stealth_tpr'))

    for at in ATTACK_TYPE_ORDER:
        sub = combined[combined['attack_type'] == at]
        if sub.empty:
            continue
        for x_col, x_label, file_suffix in plot_cols:
            sub2 = sub.dropna(subset=[x_col, 'transfer_rate'])
            if sub2.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            add_method_parallelograms(ax, sub2, x_col, 'transfer_rate')
            for arch in ARCHS:
                if arch not in found:
                    continue
                s = sub2[sub2['arch'] == arch]
                if s.empty:
                    continue
                c = s['asr'].fillna(asr_min + (asr_max - asr_min) / 2).values
                ax.scatter(s[x_col], s['transfer_rate'], c=c, cmap='Blues',
                           vmin=asr_min, vmax=asr_max, marker=ARCH_MARKERS.get(arch, 'o'),
                           s=80, edgecolors='none', zorder=2.5)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Transfer Rate')
            ds_label = DATASET_DISPLAY.get(dataset, dataset)
            ax.set_title(f'{ds_label} | {at}: {x_label} vs Transfer (shape=arch, color=ASR)')
            leg = [Line2D([0], [0], marker=ARCH_MARKERS.get(a, 'o'), color='w',
                          markerfacecolor=ARCH_LEGEND_COLOR, markersize=10, label=a, markeredgecolor='none')
                   for a in ARCHS if a in found]
            ax.legend(handles=leg, loc='upper right', frameon=False)
            sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=asr_min, vmax=asr_max))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='ASR')
            ax.grid(True, alpha=0.3)
            _set_xy_rate(ax)
            path = os.path.join(output_dir, f'scatter_stealth_{file_suffix}_vs_transfer_by_method_{at}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  按方法散点图({at}): {path}")


def _plot_scatter_combined_bubble(combined, found, output_dir, dataset: str = 'cifar10'):
    """气泡图：大小=ASR，颜色=方法，形状=arch"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # 方法颜色映射（8种）
    METHOD_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:8]
    method_to_color = {at: METHOD_COLORS[i] for i, at in enumerate(ATTACK_TYPE_ORDER)}

    plot_cols = [
        ('stealth_auc_avg', LABEL_STEALTH_AUC, 'auc'),
        ('stealth_tpr_avg', LABEL_STEALTH_TPR, 'tpr'),
    ]
    if 'S_stealth' in combined.columns:
        plot_cols.append(('S_stealth', 'S_stealth (NC, AUC)', 's_stealth_auc'))
    if 'S_stealth_tpr' in combined.columns:
        plot_cols.append(('S_stealth_tpr', 'S_stealth (NC, TPR)', 's_stealth_tpr'))

    combined_b = combined.copy()
    if 'asr' in combined_b.columns and combined_b['asr'].fillna(0).max() > 1.5:
        combined_b['asr'] = combined_b['asr'] / 100.0
    s_min, s_max = 50, 300
    asr_min, asr_max = 0.0, 1.0

    def size_from_asr(a):
        if pd.isna(a) or a <= 0:
            return s_min
        t = (float(a) - asr_min) / (asr_max - asr_min) if asr_max > asr_min else 0
        return s_min + (s_max - s_min) * t

    for x_col, x_label, file_suffix in plot_cols:
        sub = combined_b.dropna(subset=[x_col, 'transfer_rate'])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.subplots_adjust(right=0.75)
        add_method_parallelograms(ax, sub, x_col, 'transfer_rate')
        for at in ATTACK_TYPE_ORDER:
            for arch in ARCHS:
                if arch not in found:
                    continue
                s = sub[(sub['attack_type'] == at) & (sub['arch'] == arch)]
                if s.empty:
                    continue
                sizes = s['asr'].apply(size_from_asr).values
                n = len(s)
                ax.scatter(s[x_col], s['transfer_rate'],
                           c=[method_to_color[at]] * n, marker=ARCH_MARKERS.get(arch, 'o'),
                           s=sizes, alpha=0.7, edgecolors='gray', linewidths=0.5, zorder=2.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Transfer Rate')
        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        ax.set_title(f'{ds_label}: {x_label} vs Transfer (size=ASR, color=method, shape=arch)')
        leg1 = [Patch(facecolor=method_to_color[at], label=at, edgecolor='gray') for at in ATTACK_TYPE_ORDER]
        leg2 = [Line2D([0], [0], marker=ARCH_MARKERS.get(a, 'o'), color='w', markerfacecolor='gray',
                       markersize=10, label=a, markeredgecolor='none') for a in ARCHS if a in found]
        ax.legend(handles=leg1 + leg2, loc='upper right', ncol=2, fontsize=7)
        ax.grid(True, alpha=0.3)
        _set_xy_rate(ax)
        path = os.path.join(output_dir, f'scatter_stealth_{file_suffix}_vs_transfer_combined_bubble.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  气泡图(Stealth {file_suffix.upper()}): {path}")


def analyze_single_arch(arch: str, csv_path: str, output_dir: str,
                       do_corr: bool, do_violin: bool, do_pareto: bool,
                       do_box: bool, do_bar: bool,
                       do_3d: bool, do_box_poison: bool, do_box_trigger: bool,
                       do_line: bool) -> None:
    """对单个 arch 执行分析"""
    df = load_from_csv(csv_path)
    print(f"\n[{arch}] 加载 {len(df)} 条数据")
    os.makedirs(output_dir, exist_ok=True)
    if do_corr:
        run_correlation_analysis(df, output_dir, arch)
    if do_violin:
        path = os.path.join(output_dir, f'violin_by_attack_type_{arch}.png')
        plot_violin_by_attack_type(df, path, arch)
    if do_pareto:
        run_pareto_analysis(df, output_dir, arch)
    if do_box:
        path = os.path.join(output_dir, f'box_by_attack_type_{arch}.png')
        plot_box_by_attack_type(df, path, arch)
    if do_bar:
        path = os.path.join(output_dir, f'bar_by_attack_type_{arch}.png')
        plot_bar_by_attack_type(df, path, arch)
    if do_3d:
        path = os.path.join(output_dir, f'scatter_3d_{arch}.png')
        plot_3d_scatter(df, path, arch)
    if do_box_poison:
        path = os.path.join(output_dir, f'box_by_poison_rate_{arch}.png')
        plot_box_by_poison_rate(df, path, arch)
    if do_box_trigger:
        path = os.path.join(output_dir, f'box_by_trigger_type_{arch}.png')
        plot_box_by_trigger_type(df, path, arch)
    if do_line:
        plot_line_by_attack_type(df, output_dir, arch)


def resolve_output_dir_for_dataset(script_dir: str, output_dir_arg: str, dataset_key: str) -> str:
    """图表输出始终按数据集分子目录：{output_dir_arg}/{dataset_key}/（绝对路径则 {abs}/{dataset_key}/）。"""
    base = (output_dir_arg or 'analysis_outputs').strip() or 'analysis_outputs'
    if os.path.isabs(base):
        return os.path.join(base, dataset_key)
    return os.path.join(script_dir, base, dataset_key)


def main() -> None:
    parser = argparse.ArgumentParser(description='隐蔽性与迁移性分析（按模型分开）')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'tiny_imagenet', 'mnistm', 'all'],
                        help='分析的数据集：cifar10 / tiny_imagenet / mnistm / all（默认 cifar10）')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录，包含 data_{dataset}_{arch}.csv')
    parser.add_argument('--data-suffix', type=str, default='_no_nc',
                        choices=['', '_no_nc', '_nc'],
                        help='数据文件后缀："" 旧格式；"_no_nc" 无 NC；"_nc" 含 S_stealth（默认 _no_nc）')
    parser.add_argument('--data-csv', type=str, default=None,
                        help='单个 CSV 路径（用于指定 arch 时）')
    parser.add_argument('--arch', type=str, choices=ARCHS, default=None,
                        help='仅分析指定 arch（需配合 --data-csv 或 --data-dir）')
    parser.add_argument('--output-dir', type=str, default='analysis_outputs',
                        help='输出根目录；实际文件写入 {output-dir}/cifar10/ 或 .../tiny_imagenet/（按 --dataset 分项）')
    parser.add_argument('--correlation-analysis', action='store_true')
    parser.add_argument('--violin-plot', action='store_true')
    parser.add_argument('--pareto-front', action='store_true')
    parser.add_argument('--box-plot', action='store_true', help='箱线图(按attack_type)')
    parser.add_argument('--bar-plot', action='store_true', help='柱状图(按attack_type均值)')
    parser.add_argument('--scatter-3d', action='store_true', help='三维散点图')
    parser.add_argument('--box-poison', action='store_true', help='箱线图(按poison_rate)')
    parser.add_argument('--box-trigger', action='store_true', help='箱线图(按trigger_type)')
    parser.add_argument('--line-plot', action='store_true', help='折线图(每个方法一张)')
    parser.add_argument('--combined-scatter', action='store_true',
                        help='整体散点图：合并所有arch，形状=arch，颜色=ASR')
    parser.add_argument('--combined-box', action='store_true',
                        help='整体箱线图：合并所有arch，按attack_type分组')
    parser.add_argument('--all-plots', action='store_true', help='生成所有类型的图')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = args.data_dir or script_dir
    if not os.path.isabs(data_dir):
        # 相对路径相对于项目根目录，便于 --data-dir analysis 正确解析
        data_dir = os.path.join(project_root, data_dir)
    suffix = args.data_suffix

    datasets_to_run = DATASETS if args.dataset == 'all' else [args.dataset]
    if args.data_csv:
        csv_bn = os.path.basename(args.data_csv).lower()
        if 'tiny_imagenet' in csv_bn:
            datasets_to_run = ['tiny_imagenet']
        else:
            datasets_to_run = ['cifar10']

    do_corr = args.correlation_analysis
    do_violin = args.violin_plot
    do_pareto = args.pareto_front
    do_box = args.box_plot
    do_bar = args.bar_plot
    do_3d = args.scatter_3d
    do_box_poison = args.box_poison
    do_box_trigger = args.box_trigger
    do_line = args.line_plot
    do_combined = args.combined_scatter
    do_combined_box = args.combined_box

    if args.all_plots:
        do_corr = do_violin = do_pareto = do_box = do_bar = True
        do_3d = do_box_poison = do_box_trigger = do_line = do_combined = do_combined_box = True
    elif not any([do_corr, do_violin, do_pareto, do_box, do_bar, do_3d, do_box_poison, do_box_trigger, do_line, do_combined, do_combined_box]):
        do_corr = do_violin = do_pareto = do_box = do_bar = True
        do_3d = do_box_poison = do_box_trigger = do_line = do_combined = do_combined_box = True

    for ds in datasets_to_run:
        output_dir = resolve_output_dir_for_dataset(script_dir, args.output_dir, ds)

        # 确定要分析的 (arch, csv_path) 列表
        to_analyze = []
        if args.data_csv:
            path = args.data_csv if os.path.isabs(args.data_csv) else os.path.join(script_dir, args.data_csv)
            if os.path.exists(path):
                arch = args.arch or 'unknown'
                if arch == 'unknown':
                    for a in ARCHS:
                        if a in path.lower():
                            arch = a
                            break
                to_analyze = [(arch, path)]
        else:
            found = discover_arch_csvs(data_dir, suffix=suffix, dataset=ds)
            if not found:
                print(f"[{ds}] 未在 {data_dir} 找到 data_{ds}_{{arch}}{suffix}.csv，跳过")
                continue
            for arch, path in found.items():
                if args.arch is None or args.arch == arch:
                    to_analyze.append((arch, path))

        if not to_analyze:
            print(f"[{ds}] 无待分析数据")
            continue

        if do_combined and len(discover_arch_csvs(data_dir, suffix=suffix, dataset=ds)) >= 2:
            os.makedirs(output_dir, exist_ok=True)
            plot_scatter_combined_all_archs(data_dir, output_dir, suffix=suffix, dataset=ds)
        if do_combined_box and len(discover_arch_csvs(data_dir, suffix=suffix, dataset=ds)) >= 2:
            os.makedirs(output_dir, exist_ok=True)
            plot_box_by_attack_type_combined_all_archs(data_dir, output_dir, suffix=suffix, dataset=ds)

        ds_label = DATASET_DISPLAY.get(ds, ds)
        print(f"\n[{ds_label}] 将分析 {len(to_analyze)} 个模型: {[a for a, _ in to_analyze]}")
        for arch, csv_path in to_analyze:
            analyze_single_arch(arch, csv_path, output_dir,
                                do_corr, do_violin, do_pareto,
                                do_box, do_bar, do_3d, do_box_poison, do_box_trigger, do_line)
        print(f"[{ds_label}] 输出目录: {output_dir}")


if __name__ == '__main__':
    main()

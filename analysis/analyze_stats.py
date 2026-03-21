#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析隐蔽性（TPR平均、AUC平均）与迁移性，按三种模型（resnet18, mobilenet, vgg）分别分析。

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
STEALTH_COLS = ['stealth_tpr_avg', 'stealth_auc_avg']
METRIC_COLS = STEALTH_COLS + ['transfer_rate']


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """从 CSV 加载数据，要求含 stealth_tpr_avg, stealth_auc_avg, transfer_rate。
    若含 S_stealth 列则保留，供 nc 模式分析使用。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")
    extra = ['asr', 'S_stealth', 'nc_max_anomaly_index', 'nc_is_poisoned']
    for col in METRIC_COLS + [c for c in extra if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=METRIC_COLS)
    if 'attack_type' in df.columns:
        df = df[~df['attack_type'].isin(['none', ''])]
    if 'group_id' not in df.columns:
        df['group_id'] = range(1, len(df) + 1)
    return df


def discover_arch_csvs(data_dir: str, suffix: str = '') -> dict:
    """发现各 arch 的 CSV 文件，返回 {arch: path}。
    suffix: '' 表示 data_cifar10_{arch}.csv；'_no_nc' / '_nc' 表示带后缀的文件。
    """
    found = {}
    for arch in ARCHS:
        path = os.path.join(data_dir, f'data_cifar10_{arch}{suffix}.csv')
        if os.path.exists(path):
            found[arch] = path
    return found


def plot_violin_by_attack_type(df: pd.DataFrame, output_path: str, arch: str) -> None:
    """按攻击类型绘制三种指标的小提琴图：stealth_tpr_avg, stealth_auc_avg, transfer_rate"""
    if df.empty or 'attack_type' not in df.columns:
        return
    attack_types = sorted(df['attack_type'].unique())
    metrics = [
        ('stealth_tpr_avg', 'Stealth TPR Avg'),
        ('stealth_auc_avg', 'Stealth AUC Avg'),
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
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    labels = ['Stealth TPR', 'Stealth AUC', 'Transfer Rate']
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr.iloc[i, j]:.3f}', ha='center', va='center')
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
    for stealth_col, name in [('stealth_tpr_avg', 'TPR'), ('stealth_auc_avg', 'AUC')]:
        sub = df.dropna(subset=[stealth_col, 'transfer_rate'])
        if len(sub) < 2:
            continue
        x, y = sub[stealth_col].to_numpy(), sub['transfer_rate'].to_numpy()
        asr_vals = sub['asr'].fillna(0) if 'asr' in sub.columns else pd.Series([0] * len(sub))
        asr_min, asr_max = 0, max(asr_vals.max(), 0.01)
        slope, intercept = np.polyfit(x, y, 1)
        lx = np.linspace(x.min(), x.max(), 100)
        ly = slope * lx + intercept
        fig, ax = plt.subplots(figsize=(6, 5))
        for at in sub['attack_type'].unique():
            mask = sub['attack_type'] == at
            if mask.sum() == 0:
                continue
            ax.scatter(sub.loc[mask, stealth_col], sub.loc[mask, 'transfer_rate'],
                       c=asr_vals[mask], cmap='Blues', vmin=asr_min, vmax=asr_max,
                       marker=attack_markers.get(at, 'o'), s=60, edgecolors='none')
        ax.plot(lx, ly, 'gray', lw=1.5, alpha=0.7)
        ax.set_xlabel(f'Stealth {name} Avg')
        ax.set_ylabel('Transfer Rate')
        ax.set_title(f'Stealth {name} vs Transfer ({arch})')
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
        path = os.path.join(output_dir, f'scatter_stealth_{name.lower()}_vs_transfer_{arch}.png')
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
                   output_path: str, title: str) -> None:
    if df.empty or pareto_df.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.4, label='All', edgecolors='black', linewidths=0.3)
    plt.scatter(pareto_df[x_col], pareto_df[y_col], color='red', s=60, label='Pareto Front')
    sorted_df = pareto_df.sort_values(x_col)
    plt.plot(sorted_df[x_col], sorted_df[y_col], 'r-', lw=2)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
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
        ('stealth_tpr_avg', 'Stealth TPR Avg'),
        ('stealth_auc_avg', 'Stealth AUC Avg'),
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
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  箱线图: {output_path}")


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
        ax.bar(x + i * width, means, width, label=col.replace('_', ' ').title(), yerr=stds, capsize=3)
    ax.set_xticks(x + width)
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'Mean Metrics by Attack Type ({arch})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
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
    ax.set_xlabel('Stealth TPR')
    ax.set_ylabel('Stealth AUC')
    ax.set_zlabel('Transfer Rate')
    ax.set_title(f'3D: Stealth TPR vs AUC vs Transfer ({arch})')
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
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
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
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
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
        asr_max = max(sub['asr'].max(), 0.01)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f'Line Chart: {at} ({arch}) — X={x_param} (Trigger Strength)', fontsize=16, fontweight='bold', y=1.02)
        metrics = [
            ('stealth_tpr_avg', 'Stealth TPR Avg'),
            ('stealth_auc_avg', 'Stealth AUC Avg'),
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
                ax.scatter(x_vals, y_vals, c=asr_vals, cmap=cmap_name, vmin=0, vmax=asr_max,
                           marker=marker, s=70, edgecolors='none')
                ax.plot(x_vals, y_vals, '-', color='gray', linewidth=1.5, alpha=0.5)
            ax.set_xlabel(f'{x_param} (Trigger Strength)')
            ax.set_ylabel(name)
            ax.set_title(name)
            sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=0, vmax=asr_max))
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
    """隐蔽性 vs 迁移性 帕累托分析。隐蔽性：TPR 越低越好，AUC 越接近 0.5 越好"""
    if df.empty:
        return
    # Stealth TPR vs Transfer: TPR 越低越隐蔽，Transfer 越高越好
    # 用 1 - stealth_tpr 作为“隐蔽得分”（越高越隐蔽），与 transfer 一起做帕累托
    df_temp = df.copy()
    df_temp['stealth_score_tpr'] = 1 - df_temp['stealth_tpr_avg']
    pareto_tpr = compute_pareto_2d(df_temp, 'stealth_score_tpr', 'transfer_rate')
    if not pareto_tpr.empty:
        plot_pareto_2d(
            df_temp, pareto_tpr,
            'stealth_score_tpr', 'transfer_rate',
            os.path.join(output_dir, f'pareto_stealth_tpr_vs_transfer_{arch}.png'),
            f'Pareto: Stealth(1-TPR) vs Transfer ({arch})'
        )
    # Stealth AUC: 0.5 最理想，用 1 - 2*|auc-0.5| 作为隐蔽得分
    df_temp['stealth_score_auc'] = 1 - 2 * np.abs(df_temp['stealth_auc_avg'] - 0.5)
    pareto_auc = compute_pareto_2d(df_temp, 'stealth_score_auc', 'transfer_rate')
    if not pareto_auc.empty:
        plot_pareto_2d(
            df_temp, pareto_auc,
            'stealth_score_auc', 'transfer_rate',
            os.path.join(output_dir, f'pareto_stealth_auc_vs_transfer_{arch}.png'),
            f'Pareto: Stealth(AUC) vs Transfer ({arch})'
        )


# 整体图：arch 对应标记形状与图例颜色（统一蓝色）
ARCH_MARKERS = {'resnet18': '^', 'mobilenet': 'o', 'vgg': 's'}
ARCH_LEGEND_COLOR = '#4A90E2'

# 攻击方法编号说明（8种方法，用于总图）
ATTACK_TYPE_ORDER = [
    'blend', 'basic', 'SIG', 'WaNet', 'adaptive_blend', 'adaptive_patch', 'belt', 'upgd'
]


def plot_scatter_combined_all_archs(data_dir: str, output_dir: str, suffix: str = '') -> None:
    """合并 cifar10 所有 arch 数据：stealth_auc/stealth_tpr vs transfer，形状区分 arch，颜色表示 ASR"""
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    found = discover_arch_csvs(data_dir, suffix=suffix)
    if len(found) < 2:
        return
    dfs = []
    for arch, path in found.items():
        df = pd.read_csv(path, encoding='utf-8')
        for col in ['stealth_tpr_avg', 'stealth_auc_avg', 'transfer_rate', 'asr', 'S_stealth']:
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
    asr_vals = combined['asr']
    asr_valid = asr_vals.dropna()
    asr_min = asr_valid.min() if len(asr_valid) > 0 else 0
    asr_max = asr_valid.max() if len(asr_valid) > 0 else 1
    if asr_max <= asr_min:
        asr_min, asr_max = 0, 1

    plot_cols = [
        ('stealth_auc_avg', 'Stealth AUC Avg', 'auc'),
        ('stealth_tpr_avg', 'Stealth TPR Avg', 'tpr'),
    ]
    if 'S_stealth' in combined.columns:
        plot_cols.append(('S_stealth', 'S_stealth (NC)', 's_stealth'))
    # 方法编号映射
    method_to_num = {at: i + 1 for i, at in enumerate(ATTACK_TYPE_ORDER)}

    for x_col, x_label, file_suffix in plot_cols:
        sub_combined = combined.dropna(subset=[x_col, 'transfer_rate'])
        if sub_combined.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.subplots_adjust(left=0.1, right=0.65, top=0.92, bottom=0.1)
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
            sc = ax.scatter(x, y, c=c, cmap='Blues', vmin=asr_min, vmax=asr_max,
                            marker=ARCH_MARKERS.get(arch, 'o'), s=80, edgecolors='none')
            for xi, yi, at in zip(x, y, attack_types):
                num = method_to_num.get(at, '')
                if num:
                    ax.annotate(str(num), (xi, yi), fontsize=5, ha='center', va='center',
                               color='gray', alpha=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Transfer Rate')
        ax.set_title(f'CIFAR-10: {x_label} vs Transfer (shape=arch, color=ASR)')
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
        path = os.path.join(output_dir, f'scatter_stealth_{file_suffix}_vs_transfer_combined.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  整体散点图(Stealth {file_suffix.upper()}): {path}")

    # 额外：按方法分图，每个方法一张，形状=arch
    _plot_scatter_by_method(combined, found, output_dir, asr_min, asr_max)

    # 额外：气泡图，大小=ASR，颜色=方法，形状=arch
    _plot_scatter_combined_bubble(combined, found, output_dir)


def _plot_scatter_by_method(combined, found, output_dir, asr_min, asr_max):
    """按方法分图：每个方法一张，形状=arch，颜色=ASR"""
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plot_cols = [
        ('stealth_auc_avg', 'Stealth AUC Avg', 'auc'),
        ('stealth_tpr_avg', 'Stealth TPR Avg', 'tpr'),
    ]
    if 'S_stealth' in combined.columns:
        plot_cols.append(('S_stealth', 'S_stealth (NC)', 's_stealth'))

    for at in ATTACK_TYPE_ORDER:
        sub = combined[combined['attack_type'] == at]
        if sub.empty:
            continue
        for x_col, x_label, file_suffix in plot_cols:
            sub2 = sub.dropna(subset=[x_col, 'transfer_rate'])
            if sub2.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            for arch in ARCHS:
                if arch not in found:
                    continue
                s = sub2[sub2['arch'] == arch]
                if s.empty:
                    continue
                c = s['asr'].fillna(asr_min + (asr_max - asr_min) / 2).values
                ax.scatter(s[x_col], s['transfer_rate'], c=c, cmap='Blues',
                           vmin=asr_min, vmax=asr_max, marker=ARCH_MARKERS.get(arch, 'o'),
                           s=80, edgecolors='none')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Transfer Rate')
            ax.set_title(f'{at}: {x_label} vs Transfer (shape=arch, color=ASR)')
            leg = [Line2D([0], [0], marker=ARCH_MARKERS.get(a, 'o'), color='w',
                          markerfacecolor=ARCH_LEGEND_COLOR, markersize=10, label=a, markeredgecolor='none')
                   for a in ARCHS if a in found]
            ax.legend(handles=leg, loc='upper right', frameon=False)
            sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=asr_min, vmax=asr_max))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='ASR')
            ax.grid(True, alpha=0.3)
            path = os.path.join(output_dir, f'scatter_stealth_{file_suffix}_vs_transfer_by_method_{at}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  按方法散点图({at}): {path}")


def _plot_scatter_combined_bubble(combined, found, output_dir):
    """气泡图：大小=ASR，颜色=方法，形状=arch"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # 方法颜色映射（8种）
    METHOD_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:8]
    method_to_color = {at: METHOD_COLORS[i] for i, at in enumerate(ATTACK_TYPE_ORDER)}

    plot_cols = [
        ('stealth_auc_avg', 'Stealth AUC Avg', 'auc'),
        ('stealth_tpr_avg', 'Stealth TPR Avg', 'tpr'),
    ]
    if 'S_stealth' in combined.columns:
        plot_cols.append(('S_stealth', 'S_stealth (NC)', 's_stealth'))

    asr_vals = combined['asr'].fillna(0)
    s_min, s_max = 50, 300
    asr_min, asr_max = asr_vals.min(), max(asr_vals.max(), 0.01)
    if asr_max <= asr_min:
        asr_min, asr_max = 0, 1

    def size_from_asr(a):
        if pd.isna(a) or a <= 0:
            return s_min
        t = (float(a) - asr_min) / (asr_max - asr_min) if asr_max > asr_min else 0
        return s_min + (s_max - s_min) * t

    for x_col, x_label, file_suffix in plot_cols:
        sub = combined.dropna(subset=[x_col, 'transfer_rate'])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.subplots_adjust(right=0.75)
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
                           s=sizes, alpha=0.7, edgecolors='gray', linewidths=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Transfer Rate')
        ax.set_title(f'CIFAR-10: {x_label} vs Transfer (size=ASR, color=method, shape=arch)')
        leg1 = [Patch(facecolor=method_to_color[at], label=at, edgecolor='gray') for at in ATTACK_TYPE_ORDER]
        leg2 = [Line2D([0], [0], marker=ARCH_MARKERS.get(a, 'o'), color='w', markerfacecolor='gray',
                       markersize=10, label=a, markeredgecolor='none') for a in ARCHS if a in found]
        ax.legend(handles=leg1 + leg2, loc='upper right', ncol=2, fontsize=7)
        ax.grid(True, alpha=0.3)
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


def main() -> None:
    parser = argparse.ArgumentParser(description='隐蔽性与迁移性分析（按模型分开）')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录，包含 data_cifar10_{arch}.csv')
    parser.add_argument('--data-suffix', type=str, default='_no_nc',
                        choices=['', '_no_nc', '_nc'],
                        help='数据文件后缀："" 旧格式；"_no_nc" 无 NC；"_nc" 含 S_stealth（默认 _no_nc）')
    parser.add_argument('--data-csv', type=str, default=None,
                        help='单个 CSV 路径（用于指定 arch 时）')
    parser.add_argument('--arch', type=str, choices=ARCHS, default=None,
                        help='仅分析指定 arch（需配合 --data-csv 或 --data-dir）')
    parser.add_argument('--output-dir', type=str, default='analysis_outputs')
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
    parser.add_argument('--all-plots', action='store_true', help='生成所有类型的图')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = args.data_dir or script_dir
    if not os.path.isabs(data_dir):
        # 相对路径相对于项目根目录，便于 --data-dir analysis 正确解析
        data_dir = os.path.join(project_root, data_dir)
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    suffix = args.data_suffix

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
        found = discover_arch_csvs(data_dir, suffix=suffix)
        if not found:
            print(f"未在 {data_dir} 找到 data_cifar10_{{arch}}{suffix}.csv")
            return
        for arch, path in found.items():
            if args.arch is None or args.arch == arch:
                to_analyze.append((arch, path))

    if not to_analyze:
        print("无待分析数据")
        return

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

    if args.all_plots:
        do_corr = do_violin = do_pareto = do_box = do_bar = True
        do_3d = do_box_poison = do_box_trigger = do_line = do_combined = True
    elif not any([do_corr, do_violin, do_pareto, do_box, do_bar, do_3d, do_box_poison, do_box_trigger, do_line, do_combined]):
        do_corr = do_violin = do_pareto = do_box = do_bar = True
        do_3d = do_box_poison = do_box_trigger = do_line = do_combined = True

    if do_combined and len(discover_arch_csvs(data_dir, suffix=suffix)) >= 2:
        os.makedirs(output_dir, exist_ok=True)
        plot_scatter_combined_all_archs(data_dir, output_dir, suffix=suffix)

    print(f"将分析 {len(to_analyze)} 个模型: {[a for a, _ in to_analyze]}")
    for arch, csv_path in to_analyze:
        analyze_single_arch(arch, csv_path, output_dir,
                            do_corr, do_violin, do_pareto,
                            do_box, do_bar, do_3d, do_box_poison, do_box_trigger, do_line)
    print(f"\n输出目录: {output_dir}")


if __name__ == '__main__':
    main()

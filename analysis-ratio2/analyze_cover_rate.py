#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 cover_rate 对攻击效果的影响

用法:
  python analyze_cover_rate.py --data-csv data_cifar10.csv --output-dir cover
  # 或者使用默认值（CSV文件默认为data_cifar10.csv，输出目录默认为cover）
  python analyze_cover_rate.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(csv_path: str) -> pd.DataFrame:
    """加载数据"""
    # csv_path 应该已经是绝对路径或正确的相对路径（由main函数处理）
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 确保数值列为浮点数类型
    for col in ['cover_rate', 'attack_rate', 'stealth_rate', 'transfer_rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 移除包含 NaN 的行
    df = df.dropna(subset=['cover_rate', 'attack_rate', 'stealth_rate', 'transfer_rate'])
    
    print(f"成功加载 {len(df)} 个数据点")
    return df


def descriptive_statistics(df: pd.DataFrame, output_dir: str) -> None:
    """描述性统计分析"""
    print("\n" + "="*80)
    print("1. 描述性统计分析")
    print("="*80)
    
    # 按 cover_rate 分组统计
    stats_by_cover = df.groupby('cover_rate').agg({
        'attack_rate': ['count', 'mean', 'std', 'median', 'min', 'max'],
        'stealth_rate': ['mean', 'std', 'median', 'min', 'max'],
        'transfer_rate': ['mean', 'std', 'median', 'min', 'max']
    })
    
    print("\n按 Cover Rate 分组的统计信息:")
    print(stats_by_cover.to_string())
    
    # 保存统计结果
    stats_file = os.path.join(output_dir, 'cover_rate_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Cover Rate 影响分析 - 描述性统计\n")
        f.write("="*80 + "\n\n")
        f.write(stats_by_cover.to_string())
    
    print(f"\n统计结果已保存到: {stats_file}")


def correlation_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """相关性分析"""
    print("\n" + "="*80)
    print("2. 相关性分析")
    print("="*80)
    
    correlations = {
        'attack_rate': df['cover_rate'].corr(df['attack_rate']),
        'stealth_rate': df['cover_rate'].corr(df['stealth_rate']),
        'transfer_rate': df['cover_rate'].corr(df['transfer_rate'])
    }
    
    print("\nCover Rate 与各指标的 Pearson 相关系数:")
    for metric, corr in correlations.items():
        print(f"  {metric:20s}: {corr:8.4f}")
    
    # 保存相关性结果
    corr_file = os.path.join(output_dir, 'correlations.txt')
    with open(corr_file, 'w', encoding='utf-8') as f:
        f.write("Cover Rate 与各指标的相关性分析\n")
        f.write("="*80 + "\n\n")
        for metric, corr in correlations.items():
            f.write(f"{metric:20s}: {corr:8.4f}\n")
    
    print(f"\n相关性结果已保存到: {corr_file}")


def plot_trend_lines(df: pd.DataFrame, output_dir: str) -> None:
    """绘制趋势线图"""
    print("\n" + "="*80)
    print("3. 生成趋势线图")
    print("="*80)
    
    # 按 cover_rate 计算平均值和标准差
    df_agg = df.groupby('cover_rate').agg({
        'attack_rate': ['mean', 'std'],
        'stealth_rate': ['mean', 'std'],
        'transfer_rate': ['mean', 'std']
    }).reset_index()
    
    df_agg.columns = ['cover_rate', 'attack_mean', 'attack_std', 
                      'stealth_mean', 'stealth_std', 
                      'transfer_mean', 'transfer_std']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('attack_rate', 'Attack Rate', 'attack_mean', 'attack_std'),
        ('stealth_rate', 'Stealth Rate', 'stealth_mean', 'stealth_std'),
        ('transfer_rate', 'Transfer Rate', 'transfer_mean', 'transfer_std')
    ]
    
    for idx, (metric_name, metric_title, mean_col, std_col) in enumerate(metrics):
        ax = axes[idx]
        
        # 转换为 numpy array
        cover_rates = df_agg['cover_rate'].values
        means = df_agg[mean_col].values
        stds = df_agg[std_col].values
        
        # 绘制平均值线
        ax.plot(cover_rates, means, 
               marker='o', linewidth=2.5, markersize=8, 
               label='Mean', color='#2E86AB')
        
        # 绘制误差带（标准差）
        ax.fill_between(cover_rates, 
                        means - stds,
                        means + stds,
                        alpha=0.3, color='#2E86AB', label='±1 Std')
        
        # 添加线性回归线
        z = np.polyfit(cover_rates, means, 1)
        p = np.poly1d(z)
        ax.plot(cover_rates, p(cover_rates), 
               "r--", alpha=0.7, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        
        ax.set_xlabel('Cover Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_title} vs Cover Rate', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        # 设置边框样式
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'trend_lines.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"趋势线图已保存到: {output_path}")


def plot_boxplots(df: pd.DataFrame, output_dir: str) -> None:
    """绘制箱线图"""
    print("\n" + "="*80)
    print("4. 生成箱线图")
    print("="*80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('attack_rate', 'Attack Rate'),
        ('stealth_rate', 'Stealth Rate'),
        ('transfer_rate', 'Transfer Rate')
    ]
    
    for idx, (metric_name, metric_title) in enumerate(metrics):
        ax = axes[idx]
        
        # 准备数据
        cover_rates = sorted(df['cover_rate'].unique())
        data_by_cover = [df[df['cover_rate'] == cr][metric_name].values 
                        for cr in cover_rates]
        
        # 绘制箱线图
        bp = ax.boxplot(data_by_cover, labels=[f'{cr:.3f}' for cr in cover_rates],
                       patch_artist=True, widths=0.6)
        
        # 设置颜色
        for patch in bp['boxes']:
            patch.set_facecolor('#4A90E2')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Cover Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_title} Distribution by Cover Rate', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 旋转 x 轴标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'boxplots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"箱线图已保存到: {output_path}")


def plot_scatter_with_regression(df: pd.DataFrame, output_dir: str) -> None:
    """绘制散点图与回归线"""
    print("\n" + "="*80)
    print("5. 生成散点图与回归分析")
    print("="*80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('attack_rate', 'Attack Rate'),
        ('stealth_rate', 'Stealth Rate'),
        ('transfer_rate', 'Transfer Rate')
    ]
    
    regression_results = {}
    
    for idx, (metric_name, metric_title) in enumerate(metrics):
        ax = axes[idx]
        
        # 绘制散点图
        ax.scatter(df['cover_rate'], df[metric_name], 
                  alpha=0.6, s=50, edgecolors='black', linewidths=0.5,
                  color='#4A90E2', label='Data Points')
        
        # 线性回归
        X = df[['cover_rate']].values
        y = df[metric_name].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # 绘制回归线
        x_line = np.linspace(df['cover_rate'].min(), df['cover_rate'].max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, 'r--', linewidth=2.5, 
               label=f'Linear: R²={r2:.4f}')
        
        # 多项式回归（2次）
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model_poly = LinearRegression()
        model_poly.fit(X_poly, y)
        y_pred_poly = model_poly.predict(X_poly)
        r2_poly = r2_score(y, y_pred_poly)
        
        x_line_poly = np.linspace(df['cover_rate'].min(), df['cover_rate'].max(), 100)
        X_line_poly = poly.transform(x_line_poly.reshape(-1, 1))
        y_line_poly = model_poly.predict(X_line_poly)
        ax.plot(x_line_poly, y_line_poly, 'g--', linewidth=2.5, 
               label=f'Polynomial (deg=2): R²={r2_poly:.4f}')
        
        regression_results[metric_name] = {
            'linear': {'slope': model.coef_[0], 'intercept': model.intercept_, 'r2': r2},
            'polynomial': {'r2': r2_poly}
        }
        
        ax.set_xlabel('Cover Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_title} vs Cover Rate', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scatter_regression.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"散点图与回归分析已保存到: {output_path}")
    
    # 保存回归结果
    reg_file = os.path.join(output_dir, 'regression_results.txt')
    with open(reg_file, 'w', encoding='utf-8') as f:
        f.write("回归分析结果\n")
        f.write("="*80 + "\n\n")
        for metric, results in regression_results.items():
            f.write(f"{metric}:\n")
            f.write(f"  线性回归: y = {results['linear']['slope']:.6f} * x + {results['linear']['intercept']:.6f}\n")
            f.write(f"  R² (线性): {results['linear']['r2']:.6f}\n")
            f.write(f"  R² (多项式): {results['polynomial']['r2']:.6f}\n\n")
    
    print(f"回归结果已保存到: {reg_file}")


def anova_test(df: pd.DataFrame, output_dir: str) -> None:
    """方差分析（ANOVA）"""
    print("\n" + "="*80)
    print("6. 方差分析 (ANOVA)")
    print("="*80)
    
    anova_results = {}
    cover_rates = sorted(df['cover_rate'].unique())
    
    for metric in ['attack_rate', 'stealth_rate', 'transfer_rate']:
        groups = [df[df['cover_rate'] == cr][metric].values 
                 for cr in cover_rates]
        
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results[metric] = {'f_stat': f_stat, 'p_value': p_value}
            print(f"{metric:20s}: F={f_stat:10.4f}, p={p_value:.6f}", end='')
            if p_value < 0.05:
                print(" *** (显著)")
            elif p_value < 0.1:
                print(" ** (边缘显著)")
            else:
                print(" (不显著)")
        except Exception as e:
            print(f"{metric:20s}: ANOVA 失败 - {e}")
            anova_results[metric] = {'f_stat': None, 'p_value': None}
    
    # 保存 ANOVA 结果
    anova_file = os.path.join(output_dir, 'anova_results.txt')
    with open(anova_file, 'w', encoding='utf-8') as f:
        f.write("方差分析 (ANOVA) 结果\n")
        f.write("="*80 + "\n\n")
        f.write("H0: 不同 cover_rate 下的指标均值相等\n")
        f.write("H1: 至少有一组均值不同\n\n")
        for metric, results in anova_results.items():
            f.write(f"{metric}:\n")
            f.write(f"  F统计量: {results['f_stat']:.6f}\n")
            f.write(f"  p值: {results['p_value']:.6f}\n")
            if results['p_value'] is not None:
                if results['p_value'] < 0.05:
                    f.write("  结论: 拒绝H0，不同cover_rate下的均值存在显著差异 ***\n")
                elif results['p_value'] < 0.1:
                    f.write("  结论: 边缘显著，可能存在差异 **\n")
                else:
                    f.write("  结论: 接受H0，不同cover_rate下的均值无显著差异\n")
            f.write("\n")
    
    print(f"\nANOVA 结果已保存到: {anova_file}")


def plot_by_attack_type(df: pd.DataFrame, output_dir: str) -> None:
    """按攻击类型分组分析"""
    print("\n" + "="*80)
    print("7. 按攻击类型分组分析")
    print("="*80)
    
    attack_types = df['attack_type'].unique()
    
    for attack_type in attack_types:
        df_subset = df[df['attack_type'] == attack_type]
        
        print(f"\n{attack_type}:")
        for metric in ['attack_rate', 'stealth_rate', 'transfer_rate']:
            corr = df_subset['cover_rate'].corr(df_subset[metric])
            print(f"  {metric:20s} 与 cover_rate 的相关性: {corr:8.4f}")
        
        # 绘制该攻击类型的趋势图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = [
            ('attack_rate', 'Attack Rate'),
            ('stealth_rate', 'Stealth Rate'),
            ('transfer_rate', 'Transfer Rate')
        ]
        
        for idx, (metric_name, metric_title) in enumerate(metrics):
            ax = axes[idx]
            
            # 按 cover_rate 计算平均值
            df_agg = df_subset.groupby('cover_rate')[metric_name].mean().reset_index()
            
            # 转换为 numpy array
            cover_rates = df_agg['cover_rate'].values
            values = df_agg[metric_name].values
            
            ax.plot(cover_rates, values, 
                   marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
            
            # 添加线性回归线
            z = np.polyfit(cover_rates, values, 1)
            p = np.poly1d(z)
            ax.plot(cover_rates, p(cover_rates), 
                   "r--", alpha=0.7, linewidth=2, 
                   label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
            
            ax.set_xlabel('Cover Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric_title} vs Cover Rate ({attack_type})', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        safe_name = attack_type.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f'trend_{safe_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  {attack_type} 趋势图已保存到: {output_path}")


def analyze_by_test_alpha(df: pd.DataFrame, output_dir: str) -> None:
    """按test_alpha分组，分析各指标随cover_rate的变化"""
    print("\n" + "="*80)
    print("9. 按 test_alpha 分组分析 - 各指标随 cover_rate 的变化")
    print("="*80)
    
    # 检查是否有test_param_value列
    if 'test_param_value' not in df.columns:
        print("⚠ 警告: 数据中缺少 test_param_value 列，跳过此分析")
        return
    
    # 获取所有test_alpha值并排序
    test_alphas = sorted(df['test_param_value'].unique())
    cover_rates = sorted(df['cover_rate'].unique())
    
    print(f"\n找到 {len(test_alphas)} 个不同的 test_alpha 值: {test_alphas}")
    print(f"找到 {len(cover_rates)} 个不同的 cover_rate 值: {cover_rates}")
    
    # 为每个指标创建表格
    metrics = ['attack_rate', 'stealth_rate', 'transfer_rate']
    metric_names = ['Attack Rate', 'Stealth Rate', 'Transfer Rate']
    
    # 创建汇总表格（所有指标在一个表格中）
    summary_data = []
    
    for test_alpha in test_alphas:
        df_subset = df[df['test_param_value'] == test_alpha]
        
        for cover_rate in cover_rates:
            df_cr = df_subset[df_subset['cover_rate'] == cover_rate]
            
            if len(df_cr) > 0:
                row = {
                    'test_alpha': test_alpha,
                    'cover_rate': cover_rate,
                    'attack_rate': df_cr['attack_rate'].mean(),
                    'stealth_rate': df_cr['stealth_rate'].mean(),
                    'transfer_rate': df_cr['transfer_rate'].mean()
                }
                summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # 创建透视表：行是cover_rate，列是test_alpha
    tables = {}
    for metric, metric_name in zip(metrics, metric_names):
        pivot_table = summary_df.pivot_table(
            index='cover_rate',
            columns='test_alpha',
            values=metric,
            aggfunc='mean'
        )
        tables[metric] = pivot_table
        
        print(f"\n{metric_name} 随 cover_rate 的变化（按 test_alpha 分组）:")
        print("-" * 80)
        print(pivot_table.round(4).to_string())
    
    # 保存详细表格到文件
    table_file = os.path.join(output_dir, 'metrics_by_test_alpha_and_cover_rate.txt')
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write("各指标随 cover_rate 的变化（按 test_alpha 分组）\n")
        f.write("=" * 80 + "\n\n")
        
        for metric, metric_name in zip(metrics, metric_names):
            f.write(f"\n{metric_name} ({metric}):\n")
            f.write("-" * 80 + "\n")
            f.write(tables[metric].round(4).to_string())
            f.write("\n\n")
        
        # 添加汇总说明
        f.write("\n" + "=" * 80 + "\n")
        f.write("说明:\n")
        f.write("- 行: cover_rate 值\n")
        f.write("- 列: test_alpha 值\n")
        f.write("- 数值: 对应指标的平均值\n")
        f.write("- 空白: 该组合下无数据\n")
    
    print(f"\n详细表格已保存到: {table_file}")
    
    # 创建CSV格式的表格（便于后续分析）
    csv_file = os.path.join(output_dir, 'metrics_by_test_alpha_and_cover_rate.csv')
    summary_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"CSV格式表格已保存到: {csv_file}")


def analyze_specific_test_alpha(df: pd.DataFrame, output_dir: str, test_alpha: float = 0.2) -> None:
    """分析特定test_alpha下各指标随cover_rate的变化"""
    print("\n" + "="*80)
    print(f"10. test_alpha={test_alpha} 下各指标随 cover_rate 的变化")
    print("="*80)
    
    # 检查是否有test_param_value列
    if 'test_param_value' not in df.columns:
        print("⚠ 警告: 数据中缺少 test_param_value 列，跳过此分析")
        return
    
    # 筛选特定test_alpha的数据
    df_subset = df[df['test_param_value'] == test_alpha].copy()
    
    if len(df_subset) == 0:
        print(f"⚠ 警告: 未找到 test_alpha={test_alpha} 的数据")
        return
    
    # 按cover_rate排序
    cover_rates = sorted(df_subset['cover_rate'].unique())
    
    print(f"\n找到 {len(cover_rates)} 个不同的 cover_rate 值: {cover_rates}")
    
    # 创建表格数据
    table_data = []
    for cover_rate in cover_rates:
        df_cr = df_subset[df_subset['cover_rate'] == cover_rate]
        if len(df_cr) > 0:
            table_data.append({
                'cover_rate': cover_rate,
                'attack_rate': df_cr['attack_rate'].mean(),
                'stealth_rate': df_cr['stealth_rate'].mean(),
                'transfer_rate': df_cr['transfer_rate'].mean()
            })
    
    table_df = pd.DataFrame(table_data)
    
    # 打印表格
    print(f"\ntest_alpha={test_alpha} 下各指标随 cover_rate 的变化:")
    print("-" * 80)
    print(table_df.round(4).to_string(index=False))
    
    # 保存表格到文件
    table_file = os.path.join(output_dir, f'test_alpha_{test_alpha}_by_cover_rate.txt')
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write(f"test_alpha={test_alpha} 下各指标随 cover_rate 的变化\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_df.round(4).to_string(index=False))
        f.write("\n\n")
        f.write("说明:\n")
        f.write("- cover_rate: 覆盖率值\n")
        f.write("- attack_rate: 攻击成功率\n")
        f.write("- stealth_rate: 隐蔽性（防御检测失败率）\n")
        f.write("- transfer_rate: 迁移率（目标域 ASR / 源域 ASR）\n")
    
    print(f"\n表格已保存到: {table_file}")
    
    # 保存CSV格式
    csv_file = os.path.join(output_dir, f'test_alpha_{test_alpha}_by_cover_rate.csv')
    table_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"CSV格式表格已保存到: {csv_file}")
    
    # 生成图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('attack_rate', 'Attack Rate', '#2E86AB'),
        ('stealth_rate', 'Stealth Rate', '#A23B72'),
        ('transfer_rate', 'Transfer Rate', '#F18F01')
    ]
    
    for idx, (metric_name, metric_title, color) in enumerate(metrics):
        ax = axes[idx]
        
        # 转换为numpy array
        cover_rates = table_df['cover_rate'].values
        metric_values = table_df[metric_name].values
        
        # 绘制数据点和连线
        ax.plot(cover_rates, metric_values, 
               marker='o', linewidth=2.5, markersize=10, 
               color=color, label=metric_title, markerfacecolor=color,
               markeredgecolor='white', markeredgewidth=1.5)
        
        # 添加线性回归线
        z = np.polyfit(cover_rates, metric_values, 1)
        p = np.poly1d(z)
        ax.plot(cover_rates, p(cover_rates), 
               '--', alpha=0.5, linewidth=2, color='gray',
               label=f'Trend: y={z[0]:.2f}x+{z[1]:.3f}')
        
        # 在数据点上标注数值
        for i, (cr, val) in enumerate(zip(cover_rates, metric_values)):
            ax.annotate(f'{val:.3f}', 
                       xy=(cr, val),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Cover Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_title} vs Cover Rate (test_alpha={test_alpha})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        # 设置边框样式
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'test_alpha_{test_alpha}_by_cover_rate.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存到: {plot_file}")


def optimal_cover_rate_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """最优 cover_rate 分析"""
    print("\n" + "="*80)
    print("8. 最优 Cover Rate 分析")
    print("="*80)
    
    # 按 cover_rate 计算平均值
    df_agg = df.groupby('cover_rate').agg({
        'attack_rate': 'mean',
        'stealth_rate': 'mean',
        'transfer_rate': 'mean'
    }).reset_index()
    
    # 不同权重组合下的最优 cover_rate
    weight_combinations = [
        {'attack_rate': 0.5, 'stealth_rate': 0.3, 'transfer_rate': 0.2},
        {'attack_rate': 0.4, 'stealth_rate': 0.4, 'transfer_rate': 0.2},
        {'attack_rate': 0.3, 'stealth_rate': 0.5, 'transfer_rate': 0.2},
        {'attack_rate': 0.4, 'stealth_rate': 0.3, 'transfer_rate': 0.3},
    ]
    
    optimal_results = []
    
    for weights in weight_combinations:
        df_agg['composite_score'] = (
            weights['attack_rate'] * df_agg['attack_rate'] +
            weights['stealth_rate'] * df_agg['stealth_rate'] +
            weights['transfer_rate'] * df_agg['transfer_rate']
        )
        
        optimal_idx = df_agg['composite_score'].idxmax()
        optimal_cover = df_agg.loc[optimal_idx, 'cover_rate']
        optimal_score = df_agg.loc[optimal_idx, 'composite_score']
        
        optimal_results.append({
            'weights': weights,
            'optimal_cover_rate': optimal_cover,
            'optimal_score': optimal_score,
            'attack_rate': df_agg.loc[optimal_idx, 'attack_rate'],
            'stealth_rate': df_agg.loc[optimal_idx, 'stealth_rate'],
            'transfer_rate': df_agg.loc[optimal_idx, 'transfer_rate']
        })
        
        print(f"\n权重组合 {weights}:")
        print(f"  最优 cover_rate: {optimal_cover:.4f}")
        print(f"  综合得分: {optimal_score:.4f}")
        print(f"  Attack Rate: {df_agg.loc[optimal_idx, 'attack_rate']:.4f}")
        print(f"  Stealth Rate: {df_agg.loc[optimal_idx, 'stealth_rate']:.4f}")
        print(f"  Transfer Rate: {df_agg.loc[optimal_idx, 'transfer_rate']:.4f}")
    
    # 保存最优结果
    optimal_file = os.path.join(output_dir, 'optimal_cover_rate.txt')
    with open(optimal_file, 'w', encoding='utf-8') as f:
        f.write("最优 Cover Rate 分析\n")
        f.write("="*80 + "\n\n")
        for result in optimal_results:
            f.write(f"权重组合: {result['weights']}\n")
            f.write(f"  最优 cover_rate: {result['optimal_cover_rate']:.4f}\n")
            f.write(f"  综合得分: {result['optimal_score']:.4f}\n")
            f.write(f"  Attack Rate: {result['attack_rate']:.4f}\n")
            f.write(f"  Stealth Rate: {result['stealth_rate']:.4f}\n")
            f.write(f"  Transfer Rate: {result['transfer_rate']:.4f}\n\n")
    
    print(f"\n最优结果已保存到: {optimal_file}")


def main():
    parser = argparse.ArgumentParser(description='分析 cover_rate 对攻击效果的影响')
    parser.add_argument('--data-csv', type=str, default='data_cifar10.csv',
                       help='CSV 数据文件路径（默认: data_cifar10.csv，相对于脚本目录）')
    parser.add_argument('--output-dir', type=str, default='cover',
                       help='输出目录（默认: cover，相对于脚本目录）')
    args = parser.parse_args()
    
    # 获取脚本所在目录（analysis文件夹）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理CSV文件路径：如果是相对路径，则相对于脚本目录
    if not os.path.isabs(args.data_csv):
        csv_path = os.path.join(script_dir, args.data_csv)
    else:
        csv_path = args.data_csv
    
    # 如果输出目录是相对路径，则相对于脚本目录
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"CSV文件路径: {csv_path}")
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    df = load_data(csv_path)
    
    # 执行各项分析
    descriptive_statistics(df, output_dir)
    correlation_analysis(df, output_dir)
    plot_trend_lines(df, output_dir)
    plot_boxplots(df, output_dir)
    plot_scatter_with_regression(df, output_dir)
    anova_test(df, output_dir)
    plot_by_attack_type(df, output_dir)
    analyze_by_test_alpha(df, output_dir)
    analyze_specific_test_alpha(df, output_dir, test_alpha=0.2)
    optimal_cover_rate_analysis(df, output_dir)
    
    print("\n" + "="*80)
    print("✓ 所有分析完成！")
    print(f"✓ 结果已保存到: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()


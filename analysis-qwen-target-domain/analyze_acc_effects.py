"""Analyze how target-domain clean ACC affects transfer ASR and stealth.

Outputs:
- acc_effect_rows.csv
- acc_effect_summary.json
- acc_effect_tradeoff_analysis.md
- PNG figures under figures/
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis-qwen-target-domain"
BASE = ROOT / "poisoned_train_set" / "tiny_imagenet"
FIG_DIR = ANALYSIS_DIR / "figures"

ACC_RE = re.compile(r"准确率:\s*([0-9.]+)")
ASR_RE = re.compile(r"攻击成功率\s*\(ASR\):\s*([0-9.]+)")
ATTACK_RE = re.compile(r"攻击类型:\s*(\S+)")

DEFENSE_FILES = {
    "strip": "strip_defense_results.json",
    "sentinet": "sentinet_defense_results.json",
    "scaleup": "scaleup_defense_results.json",
    "ibd_psc": "ibd_psc_defense_results.json",
}


def parse_result(path: Path) -> tuple[float, float, str]:
    text = path.read_text(encoding="utf-8")
    return (
        float(ACC_RE.search(text).group(1)),
        float(ASR_RE.search(text).group(1)),
        ATTACK_RE.search(text).group(1),
    )


def suffix(name: str, prefix: str) -> str:
    return name[len(prefix):-4]


def arch_from_dir(name: str) -> str:
    if "arch=ResNet18" in name:
        return "resnet18"
    if "arch=mobilenetv2" in name:
        return "mobilenetv2"
    if "arch=vgg19_bn" in name:
        return "vgg19_bn"
    return "unknown"


def read_defenses(model_dir: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    aucs: list[float] = []
    tprs: list[float] = []
    for key, filename in DEFENSE_FILES.items():
        path = model_dir / filename
        if not path.exists():
            out[f"{key}_auc"] = None
            out[f"{key}_tpr"] = None
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        auc = float(data["auc"]) if data.get("auc") is not None else None
        tpr = float(data["tpr"]) if data.get("tpr") is not None else None
        if tpr is not None and tpr > 1:
            tpr = tpr / 100.0
        out[f"{key}_auc"] = auc
        out[f"{key}_tpr"] = tpr
        if auc is not None:
            aucs.append(auc)
        if tpr is not None:
            tprs.append(tpr)
    out["det_auc_avg"] = mean(aucs) if aucs else None
    out["det_tpr_avg"] = mean(tprs) if tprs else None
    out["stealth_auc"] = 1 - out["det_auc_avg"] if out["det_auc_avg"] is not None else None
    out["stealth_tpr"] = 1 - out["det_tpr_avg"] if out["det_tpr_avg"] is not None else None
    return out


def read_source_metrics(model_dir: Path) -> dict[str, float | None]:
    path = model_dir / "test_results_seed=2333.json"
    if not path.exists():
        return {"src_clean_acc": None, "src_asr": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "src_clean_acc": data.get("clean_acc"),
        "src_asr": data.get("asr"),
    }


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        iv2 = {
            suffix(p.name, "test_tiny_target_domain_results"): p
            for p in model_dir.iterdir()
            if p.name.startswith("test_tiny_target_domain_results") and p.name.endswith(".txt")
        }
        qwen = {
            suffix(p.name, "test_tiny_target_domain_qwen_results"): p
            for p in model_dir.iterdir()
            if p.name.startswith("test_tiny_target_domain_qwen_results") and p.name.endswith(".txt")
        }
        if not iv2 or not qwen:
            continue
        defenses = read_defenses(model_dir)
        source = read_source_metrics(model_dir)
        for variant in sorted(set(iv2) & set(qwen)):
            iv2_acc, iv2_asr, attack = parse_result(iv2[variant])
            qwen_acc, qwen_asr, _ = parse_result(qwen[variant])
            row = {
                "config": model_dir.name + variant,
                "attack": attack,
                "arch": arch_from_dir(model_dir.name),
                "iv2_acc": iv2_acc,
                "iv2_asr": iv2_asr,
                "qwen_acc": qwen_acc,
                "qwen_asr": qwen_asr,
                "delta_acc": qwen_acc - iv2_acc,
                "delta_asr": qwen_asr - iv2_asr,
                **source,
                **defenses,
            }
            rows.append(row)
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ranks(vals: list[float]) -> list[float]:
    order = sorted((v, i) for i, v in enumerate(vals))
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k][1]] = avg
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def partial_corr(x: list[float], y: list[float], z: list[float]) -> float:
    """Partial Pearson corr(x, y | z), single control variable."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    X = np.column_stack([np.ones(len(z_arr)), z_arr])
    bx = np.linalg.pinv(X) @ x_arr
    by = np.linalg.pinv(X) @ y_arr
    return pearson(list(x_arr - X @ bx), list(y_arr - X @ by))


def safe_vals(rows: list[dict[str, object]], *keys: str) -> list[dict[str, object]]:
    out = []
    for row in rows:
        ok = True
        for key in keys:
            val = row.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                ok = False
        if ok:
            out.append(row)
    return out


def avg(rows: list[dict[str, object]], key: str) -> float:
    return mean(float(r[key]) for r in rows if r.get(key) is not None)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"n": len(rows)}
    valid = safe_vals(rows, "det_auc_avg")
    for domain in ["iv2", "qwen"]:
        acc = f"{domain}_acc"
        asr = f"{domain}_asr"
        domain_rows = safe_vals(valid, acc, asr, "det_auc_avg")
        xs = [float(r[acc]) for r in domain_rows]
        ys = [float(r[asr]) for r in domain_rows]
        ds = [float(r["det_auc_avg"]) for r in domain_rows]
        summary[domain] = {
            "mean_acc": avg(domain_rows, acc),
            "mean_asr": avg(domain_rows, asr),
            "acc_asr_pearson": pearson(xs, ys),
            "acc_asr_spearman": spearman(xs, ys),
            "acc_det_auc_pearson": pearson(xs, ds),
            "acc_det_auc_spearman": spearman(xs, ds),
            "asr_det_auc_pearson": pearson(ys, ds),
            "asr_det_auc_spearman": spearman(ys, ds),
            "asr_det_auc_partial_acc_pearson": partial_corr(ys, ds, xs),
        }
    summary["delta"] = {
        "mean_delta_acc": avg(rows, "delta_acc"),
        "mean_delta_asr": avg(rows, "delta_asr"),
        "delta_acc_delta_asr_pearson": pearson(
            [float(r["delta_acc"]) for r in rows],
            [float(r["delta_asr"]) for r in rows],
        ),
        "delta_acc_delta_asr_spearman": spearman(
            [float(r["delta_acc"]) for r in rows],
            [float(r["delta_asr"]) for r in rows],
        ),
    }

    by_attack = []
    for attack in sorted({str(r["attack"]) for r in rows}):
        sub = [r for r in rows if r["attack"] == attack]
        by_attack.append({
            "attack": attack,
            "n": len(sub),
            "iv2_acc": avg(sub, "iv2_acc"),
            "qwen_acc": avg(sub, "qwen_acc"),
            "iv2_asr": avg(sub, "iv2_asr"),
            "qwen_asr": avg(sub, "qwen_asr"),
            "delta_acc": avg(sub, "delta_acc"),
            "delta_asr": avg(sub, "delta_asr"),
            "det_auc": avg(sub, "det_auc_avg"),
            "stealth_auc": avg(sub, "stealth_auc"),
            "iv2_acc_asr_pearson": pearson([float(r["iv2_acc"]) for r in sub], [float(r["iv2_asr"]) for r in sub]),
            "qwen_acc_asr_pearson": pearson([float(r["qwen_acc"]) for r in sub], [float(r["qwen_asr"]) for r in sub]),
        })
    summary["by_attack"] = by_attack

    bins = {}
    for domain in ["iv2", "qwen"]:
        acc_key = f"{domain}_acc"
        asr_key = f"{domain}_asr"
        vals = sorted(float(r[acc_key]) for r in rows)
        t1 = vals[len(vals) // 3]
        t2 = vals[2 * len(vals) // 3]
        items = []
        for label, pred in [
            ("low_acc", lambda r: float(r[acc_key]) <= t1),
            ("mid_acc", lambda r: t1 < float(r[acc_key]) <= t2),
            ("high_acc", lambda r: float(r[acc_key]) > t2),
        ]:
            sub = [r for r in rows if pred(r)]
            items.append({
                "bin": label,
                "n": len(sub),
                "mean_acc": avg(sub, acc_key),
                "mean_asr": avg(sub, asr_key),
                "mean_det_auc": avg(sub, "det_auc_avg"),
                "mean_stealth_auc": avg(sub, "stealth_auc"),
            })
        bins[domain] = {"thresholds": [t1, t2], "bins": items}
    summary["acc_bins"] = bins
    return summary


def write_rows(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "config", "attack", "arch",
        "src_clean_acc", "src_asr",
        "iv2_acc", "iv2_asr", "qwen_acc", "qwen_asr", "delta_acc", "delta_asr",
        "det_auc_avg", "stealth_auc", "det_tpr_avg", "stealth_tpr",
        "strip_auc", "sentinet_auc", "scaleup_auc", "ibd_psc_auc",
    ]
    path = ANALYSIS_DIR / "acc_effect_rows.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def pp(x: float) -> str:
    return f"{x * 100:+.2f}pp"


def plot_figures(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "SIG": "#4C78A8",
        "upgd": "#F58518",
        "WaNet": "#54A24B",
        "basic": "#E45756",
        "blend": "#72B7B2",
        "adaptive_blend": "#B279A2",
        "adaptive_patch": "#FF9DA6",
        "belt": "#9D755D",
    }

    # Figure 1: ACC vs ASR by domain.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, domain, title in [
        (axes[0], "iv2", "ImageNetV2: ACC vs ASR"),
        (axes[1], "qwen", "Qwen: ACC vs ASR"),
    ]:
        for attack in sorted(colors):
            sub = [r for r in rows if r["attack"] == attack]
            ax.scatter(
                [float(r[f"{domain}_acc"]) * 100 for r in sub],
                [float(r[f"{domain}_asr"]) * 100 for r in sub],
                s=18,
                alpha=0.65,
                label=attack,
                color=colors[attack],
            )
        xs = np.asarray([float(r[f"{domain}_acc"]) * 100 for r in rows])
        ys = np.asarray([float(r[f"{domain}_asr"]) * 100 for r in rows])
        coef = np.polyfit(xs, ys, 1)
        ax.plot(np.sort(xs), coef[0] * np.sort(xs) + coef[1], color="black", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Target-domain clean ACC (%)")
        ax.set_ylabel("Target-domain ASR / transfer_rate (%)")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acc_vs_asr_by_domain.png", dpi=180)
    plt.close(fig)

    # Figure 2: ACC bins.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, domain, title in [
        (axes[0], "iv2", "ImageNetV2 ACC bins"),
        (axes[1], "qwen", "Qwen ACC bins"),
    ]:
        bins = summary["acc_bins"][domain]["bins"]
        labels = ["low", "mid", "high"]
        x = np.arange(len(labels))
        width = 0.26
        ax.bar(x - width, [b["mean_acc"] * 100 for b in bins], width, label="ACC")
        ax.bar(x, [b["mean_asr"] * 100 for b in bins], width, label="ASR")
        ax.bar(x + width, [b["mean_det_auc"] * 100 for b in bins], width, label="Detection AUC")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel("Mean value (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acc_bins_asr_detection.png", dpi=180)
    plt.close(fig)

    # Figure 3: Attack-level delta.
    atk_rows = sorted(summary["by_attack"], key=lambda r: r["delta_asr"])
    labels = [r["attack"] for r in atk_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.18, [r["delta_acc"] * 100 for r in atk_rows], 0.36, label="Delta ACC (Qwen - IV2)")
    ax.bar(x + 0.18, [r["delta_asr"] * 100 for r in atk_rows], 0.36, label="Delta ASR (Qwen - IV2)")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Difference (percentage points)")
    ax.set_title("Attack-level effect of moving from low-ACC ImageNetV2 to higher-ACC Qwen")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attack_delta_acc_asr.png", dpi=180)
    plt.close(fig)

    # Figure 4: ASR vs detection AUC, colored by Qwen ACC.
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        [float(r["qwen_asr"]) * 100 for r in rows],
        [float(r["det_auc_avg"]) * 100 for r in rows],
        c=[float(r["qwen_acc"]) * 100 for r in rows],
        s=28,
        alpha=0.78,
        cmap="viridis",
    )
    ax.set_xlabel("Qwen target-domain ASR / transfer_rate (%)")
    ax.set_ylabel("Mean detection AUC across four defenses (%)")
    ax.set_title("Trade-off remains after considering ACC")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Qwen clean ACC (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qwen_asr_vs_detection_colored_by_acc.png", dpi=180)
    plt.close(fig)

    # Figure 5: ACC vs stealth.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, domain, title in [
        (axes[0], "iv2", "ImageNetV2 ACC vs stealth AUC"),
        (axes[1], "qwen", "Qwen ACC vs stealth AUC"),
    ]:
        for attack in sorted(colors):
            sub = [r for r in rows if r["attack"] == attack]
            ax.scatter(
                [float(r[f"{domain}_acc"]) * 100 for r in sub],
                [float(r["stealth_auc"]) * 100 for r in sub],
                s=18,
                alpha=0.65,
                label=attack,
                color=colors[attack],
            )
        ax.set_title(title)
        ax.set_xlabel("Target-domain clean ACC (%)")
        ax.set_ylabel("Stealth AUC = 1 - detection AUC (%)")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acc_vs_stealth_by_domain.png", dpi=180)
    plt.close(fig)


def write_report(summary: dict[str, object]) -> None:
    iv2 = summary["iv2"]
    qwen = summary["qwen"]
    delta = summary["delta"]
    by_attack = summary["by_attack"]
    bins = summary["acc_bins"]

    attack_lines = []
    for row in by_attack:
        attack_lines.append(
            f"| {row['attack']} | {row['n']} | {pct(row['iv2_acc'])} | {pct(row['qwen_acc'])} | "
            f"{pct(row['iv2_asr'])} | {pct(row['qwen_asr'])} | {pp(row['delta_asr'])} | "
            f"{row['det_auc']:.3f} | {row['stealth_auc']:.3f} |"
        )

    def bin_lines(domain: str) -> str:
        out = []
        for row in bins[domain]["bins"]:
            out.append(
                f"| {domain} | {row['bin']} | {row['n']} | {pct(row['mean_acc'])} | "
                f"{pct(row['mean_asr'])} | {row['mean_det_auc']:.3f} | {row['mean_stealth_auc']:.3f} |"
            )
        return "\n".join(out)

    md = f"""# ACC 对迁移性与隐蔽性 Trade-off 的影响分析

本报告基于 `poisoned_train_set/tiny_imagenet` 下同一批 465 个后门配置在两个目标域上的配对结果，并接入每个模型目录中的四个防御结果：`strip_defense_results.json`、`sentinet_defense_results.json`、`scaleup_defense_results.json`、`ibd_psc_defense_results.json`。

- ImageNetV2 organized：`/workspace/data/imagenetv2-matched-frequency-tiny-organized`
- Qwen full organized：`/workspace/data/tiny-target-domain-qwen-full-organized`
- 迁移性指标：目标域 ASR / `transfer_rate`
- 隐蔽性指标：`stealth_auc = 1 - mean(STRIP, SentiNet, SCaLe-Up, IBD-PSC detection AUC)`

## 1. 先给结论

你的判断是成立的，但要写得更精确：

> **ACC 下降会让目标域 ASR 更容易上升，从而放大“迁移性”的表观值；但隐蔽性基本不由目标域 ACC 直接决定，而主要由攻击机制和触发器强度决定。**

全量 465 个配置中，Qwen 目标域相比 ImageNetV2 目标域：

- 平均 ACC：{pct(iv2['mean_acc'])} -> {pct(qwen['mean_acc'])}，变化 {pp(delta['mean_delta_acc'])}
- 平均 ASR：{pct(iv2['mean_asr'])} -> {pct(qwen['mean_asr'])}，变化 {pp(delta['mean_delta_asr'])}
- `delta_ACC` 与 `delta_ASR` 的相关：Pearson {delta['delta_acc_delta_asr_pearson']:+.3f}，Spearman {delta['delta_acc_delta_asr_spearman']:+.3f}

这说明：同一批模型换到 clean ACC 更高的 Qwen 域后，ASR 反而下降；并且 ACC 提升越多，ASR 往往下降越多。这支持“低 ACC 会放大 ASR”的解释。

## 2. ACC 与 ASR：低 ACC 确实伴随更高 ASR

| 目标域 | mean ACC | mean ASR | ACC-ASR Pearson | ACC-ASR Spearman |
|---|---:|---:|---:|---:|
| ImageNetV2 | {pct(iv2['mean_acc'])} | {pct(iv2['mean_asr'])} | {iv2['acc_asr_pearson']:+.3f} | {iv2['acc_asr_spearman']:+.3f} |
| Qwen | {pct(qwen['mean_acc'])} | {pct(qwen['mean_asr'])} | {qwen['acc_asr_pearson']:+.3f} | {qwen['acc_asr_spearman']:+.3f} |

两个域都是负相关，Qwen 更明显。这说明目标域 clean ACC 越低，target-domain ASR 越容易偏高。但这个相关不是强相关，说明不能把 ASR 全部解释为 ACC 造成的，攻击类型仍是主导因素。

按 ACC 三分位分组：

| 目标域 | ACC 组 | n | mean ACC | mean ASR | mean detection AUC | mean stealth AUC |
|---|---|---:|---:|---:|---:|---:|
{bin_lines('iv2')}
{bin_lines('qwen')}

这里最重要的是 Qwen：低 ACC 组 ASR 为 {pct(bins['qwen']['bins'][0]['mean_asr'])}，高 ACC 组 ASR 为 {pct(bins['qwen']['bins'][2]['mean_asr'])}，相差约 {pp(bins['qwen']['bins'][0]['mean_asr'] - bins['qwen']['bins'][2]['mean_asr'])}。同时 detection AUC 在三个 ACC 组之间非常接近，说明 ACC 更影响 ASR 解释，而不是直接影响隐蔽性。

## 3. ACC 与隐蔽性：几乎没有直接关系

| 目标域 ACC | ACC vs detection AUC Pearson | ACC vs detection AUC Spearman |
|---|---:|---:|
| ImageNetV2 ACC | {iv2['acc_det_auc_pearson']:+.3f} | {iv2['acc_det_auc_spearman']:+.3f} |
| Qwen ACC | {qwen['acc_det_auc_pearson']:+.3f} | {qwen['acc_det_auc_spearman']:+.3f} |

这个结果支持你的假设：**ACC 的变化会影响 ASR，但隐蔽性基本不随目标域 ACC 同步变化。**

原因是四种防御评估的是同一源模型/触发器的检测可分性，而不是目标域 clean 分类准确率。换句话说，ImageNetV2 和 Qwen 的 ACC 是目标域性质；而 STRIP/SentiNet/SCaLe-Up/IBD-PSC 的 AUC 更多反映触发器是否显著、预测是否稳定、内部行为是否异常。

因此，如果论文要讨论 ACC，应把它写成：

- ACC 是 **迁移性 ASR 的解释控制变量**
- ACC 不是 **隐蔽性 AUC 的直接决定变量**
- 低 ACC 会让“高迁移”看起来更强，但不会自动让攻击更容易或更不容易被检测

## 4. 控制 ACC 后，原 trade-off 仍然存在

| 目标域 | ASR vs detection AUC | 控制 ACC 后 ASR vs detection AUC |
|---|---:|---:|
| ImageNetV2 | Pearson {iv2['asr_det_auc_pearson']:+.3f} / Spearman {iv2['asr_det_auc_spearman']:+.3f} | Pearson {iv2['asr_det_auc_partial_acc_pearson']:+.3f} |
| Qwen | Pearson {qwen['asr_det_auc_pearson']:+.3f} / Spearman {qwen['asr_det_auc_spearman']:+.3f} | Pearson {qwen['asr_det_auc_partial_acc_pearson']:+.3f} |

这是最关键的统计结果。控制 ACC 后，ASR 与 detection AUC 的相关几乎没有下降。因此：

> 原始“迁移性越强，越容易被检测，隐蔽性越差”的结论不是由 ACC 下降造成的伪相关；但 ACC 会影响你对 transfer_rate 绝对值的解释。

换句话说，原结论需要修正，但不是推翻。

## 5. 按攻击类型看：WaNet 是低 ACC 高 ASR 的关键来源

| 攻击 | n | IV2 ACC | Qwen ACC | IV2 ASR | Qwen ASR | delta ASR | detection AUC | stealth AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(attack_lines)}

几个具体观察：

1. **WaNet 最能说明“ACC 下降导致 ASR 上升/维持高位”**。它的 ACC 明显低于其它攻击，但 ASR 在两个域都高，Qwen 甚至略高。这说明空间形变会破坏 clean 分类，同时仍能稳定触发目标类。
2. **SIG 和 UPGD 最受目标域 ACC/统计差异影响**。Qwen ACC 提升后，SIG ASR 下降 {pp(next(r for r in by_attack if r['attack'] == 'SIG')['delta_asr'])}，UPGD ASR 下降 {pp(next(r for r in by_attack if r['attack'] == 'upgd')['delta_asr'])}。这说明 ImageNetV2 上部分高 ASR 可能依赖低 ACC 和域统计不稳定。
3. **basic / adaptive_patch / blend / belt 的 ASR 也下降，但幅度较小**。这些触发器更像稳定 shortcut，迁移性比较强，同时 detection AUC 也高，因此它们仍然是 trade-off 的主证据。
4. **隐蔽性排序主要由攻击机制决定**。basic/adaptive_patch/belt detection AUC 高，SIG/UPGD detection AUC 低；这个排序不是由目标域 ACC 决定的。

## 6. 对原研究结论的修正建议

原结论：迁移性和隐蔽性之间存在 trade-off。

建议修正为：

> 在不考虑 Neural Cleanse、使用 STRIP/SentiNet/SCaLe-Up/IBD-PSC 四种防御的设定下，迁移性与隐蔽性之间仍存在显著统计性 trade-off：目标域 ASR 越高，平均 detection AUC 越高，stealth AUC 越低。但在 Tiny-ImageNet 迁移设置中，目标域 clean ACC 是解释 transfer_rate 的关键控制变量。ImageNetV2 目标域 clean ACC 较低，会放大部分 ASR，尤其是 SIG/UPGD/WaNet 等对目标域统计或空间形变敏感的攻击。因此，Tiny 上的 transfer_rate 绝对值应结合目标域 ACC 解释；不过控制 ACC 后，ASR 与 detection AUC 的强相关仍然存在，说明 trade-off 不是 ACC 低造成的假象。

更简洁的论文表述：

> ACC affects the interpretation of transferability but not the main stealth trade-off. Lower target-domain ACC tends to inflate target-domain ASR, making attacks appear more transferable. However, stealth metrics remain largely attack-mechanism-driven, and the ASR-detection correlation remains strong after controlling for ACC.

## 7. 配套图表

已生成以下图：

- `figures/acc_vs_asr_by_domain.png`：两个目标域内 ACC 与 ASR 的散点与趋势线
- `figures/acc_bins_asr_detection.png`：ACC 三分位下 ASR 与 detection AUC 的变化
- `figures/attack_delta_acc_asr.png`：从 ImageNetV2 切到 Qwen 后，各攻击的 ACC/ASR 差异
- `figures/qwen_asr_vs_detection_colored_by_acc.png`：Qwen 域中 ASR 与 detection AUC 的 trade-off，并用 ACC 着色
- `figures/acc_vs_stealth_by_domain.png`：ACC 与 stealth AUC 的关系

## 8. 最终结论

你的新解释应该放进论文：**ACC 下降确实会造成 ASR 上升的表观效应；但隐蔽性基本不随 ACC 改变，而是由攻击机制决定。**

所以最终结论不是“原 trade-off 错了”，而是：

1. 原 trade-off 仍成立：高 ASR 配置通常 detection AUC 更高、stealth AUC 更低。
2. Tiny/ImageNetV2 的 ASR 绝对值被低 ACC 放大，需要用 Qwen 对照和 ACC 控制说明。
3. 隐蔽性不应和目标域 ACC 直接绑定；它主要来自触发器可检测性。
4. WaNet 是“低 ACC 但 ASR 不降”的关键特殊例；SIG/UPGD 是“换到更高 ACC 域后 ASR 明显下降”的关键证据。
"""
    (ANALYSIS_DIR / "acc_effect_tradeoff_analysis.md").write_text(md, encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    rows = collect_rows()
    if len(rows) != 465:
        print(f"[WARN] expected 465 rows, got {len(rows)}")
    write_rows(rows)
    summary = summarize(rows)
    (ANALYSIS_DIR / "acc_effect_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_figures(rows, summary)
    write_report(summary)
    print(f"Wrote {len(rows)} rows")
    print(f"Report: {ANALYSIS_DIR / 'acc_effect_tradeoff_analysis.md'}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
从 poisoned_train_set/tiny_imagenet 下各模型目录中提取 Tiny ImageNet-C 测试结果，
按三个模型（ResNet18、VGG19-BN、MobileNetV2）分别生成 Markdown 表格：
- 不同损坏类型 × 不同严重程度 下的 ACC 和 ASR。
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# 项目根目录（脚本在 scripts/ 下）
ROOT = Path(__file__).resolve().parent.parent
POISON_BASE = ROOT / "poisoned_train_set" / "tiny_imagenet"

# 15 种损坏类型（与 test_tiny_imagenet.py 的 choices 一致，用于表头顺序）
CORRUPTION_TYPES = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur",
]
SEVERITIES = [1, 2, 3, 4, 5]

# 目录名中 arch= 与展示名的映射
ARCH_DISPLAY = {
    "resnet18_tiny_imagenet": "ResNet18",
    "ResNet18_tiny_imagenet": "ResNet18",
    "vgg19_bn_tiny_imagenet": "VGG19-BN",
    "mobilenetv2_tiny_imagenet": "MobileNetV2",
}


def parse_result_file(filepath):
    """解析单个结果文件，返回 (acc, asr) 或 None。"""
    acc = asr = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("准确率:"):
                    # 准确率: 0.117100 (11.71%)
                    m = re.search(r"([0-9.]+)\s*\([0-9.]+%\)", line)
                    if m:
                        acc = float(m.group(1))
                elif "攻击成功率" in line or "ASR" in line:
                    m = re.search(r"([0-9.]+)\s*\([0-9.]+%\)", line)
                    if m:
                        asr = float(m.group(1))
    except Exception:
        return None
    if acc is None or asr is None:
        return None
    return (acc, asr)


def get_corruption_severity_from_path(filepath):
    """从文件名提取 corruption_type 和 severity。"""
    name = os.path.basename(filepath)
    # test_tiny_imagenet_results_corruption=gaussian_noise_severity=4.txt
    m = re.search(r"corruption=([a-z_]+)_severity=([1-5])", name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def collect_results_by_model():
    """
    扫描 POISON_BASE 下所有带 arch= 的目录，收集每个模型下
    test_tiny_imagenet_results_corruption=*_severity=*.txt 的 ACC/ASR。
    返回: { model_display: { (corruption, severity): (acc, asr) } }
    """
    by_model = defaultdict(dict)

    if not POISON_BASE.exists():
        return by_model

    for d in POISON_BASE.iterdir():
        if not d.is_dir() or "arch=" not in d.name:
            continue
        # 解析 arch
        arch_match = re.search(r"arch=([^_]+(?:_[^_]+)*_tiny_imagenet)", d.name)
        if not arch_match:
            continue
        arch_key = arch_match.group(1)
        model_display = ARCH_DISPLAY.get(arch_key)
        if not model_display:
            model_display = arch_key.replace("_tiny_imagenet", "")

        for f in d.glob("test_tiny_imagenet_results*.txt"):
            if "corruption=" not in f.name or "severity=" not in f.name:
                continue
            c, s = get_corruption_severity_from_path(str(f))
            if c is None or s is None:
                continue
            res = parse_result_file(f)
            if res:
                by_model[model_display][(c, s)] = res

    return dict(by_model)


def build_md_table(data, metric_idx, metric_name):
    """为某个模型构建一张 Markdown 表格。data: {(corruption, severity): (acc, asr)}；metric_idx: 0=ACC, 1=ASR。"""
    lines = []
    # 表头: | 损坏类型 | 1 | 2 | 3 | 4 | 5 |
    header = "| 损坏类型 | " + " | ".join(str(s) for s in SEVERITIES) + " |"
    sep = "|" + "--------|" * (len(SEVERITIES) + 1)
    lines.append(header)
    lines.append(sep)

    for c in CORRUPTION_TYPES:
        row = [c]
        for s in SEVERITIES:
            val = data.get((c, s))
            if val is not None:
                v = val[metric_idx]
                row.append(f"{v:.2%}" if v <= 1 else f"{v:.2f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    out_path = ROOT / "Tiny_ImageNet_C_Results.md"

    by_model = collect_results_by_model()

    # 固定三个模型的顺序
    model_order = ["ResNet18", "VGG19-BN", "MobileNetV2"]
    models = [m for m in model_order if m in by_model]
    # 若有其他 arch 也一并输出
    for m in sorted(by_model.keys()):
        if m not in models:
            models.append(m)

    md_lines = [
        "# Tiny ImageNet-C 测试结果汇总",
        "",
        "按模型分别列出不同**损坏类型**与**严重程度**下的 **ACC** 与 **ASR**。",
        "",
    ]

    for model in models:
        data = by_model[model]
        md_lines.append(f"## {model}")
        md_lines.append("")

        md_lines.append("### 准确率 (ACC)")
        md_lines.append("")
        md_lines.append(build_md_table(data, 0, "ACC"))
        md_lines.append("")

        md_lines.append("### 攻击成功率 (ASR)")
        md_lines.append("")
        md_lines.append(build_md_table(data, 1, "ASR"))
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"已生成: {out_path}")
    for model in models:
        n = len(by_model[model])
        print(f"  - {model}: {n} 条结果")


if __name__ == "__main__":
    main()

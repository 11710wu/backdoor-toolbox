"""Compare Tiny target-domain transfer results between two target domains.

It scans poisoned model directories and pairs, per model dir:
- ImageNetV2 organized result:  ``test_tiny_target_domain_results*.txt``
- Qwen full organized result:   ``test_tiny_target_domain_qwen_results*.txt``

Both files are produced by, respectively, ``test_tiny_target_domain.py`` and
``test_tiny_target_domain_qwen.py`` and share the exact same model / poison /
ASR logic, so ACC and ASR are directly comparable across the two target domains.

Terminology (kept consistent with the comparison report):
- ACC = clean accuracy on the target domain (no trigger)
- ASR = attack success rate on the target domain (trigger added) = "transfer rate"

Usage examples:
    # Full sweep over the default root (poisoned_train_set1), markdown table
    python compare_tiny_target_domain_results.py

    # A specific root and dataset, write a full report
    python compare_tiny_target_domain_results.py --poisoned-root poisoned_train_set1 \
        --dataset tiny_imagenet --output report.md

    # Single model dir (single-config validation)
    python compare_tiny_target_domain_results.py --model-dir \
        poisoned_train_set1/tiny_imagenet/adaptive_blend_0.010_alpha=0.050_cover=0.010_trigger=hellokitty_64.png_poison_seed=2333_arch=ResNet18_tiny_imagenet
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

# 准确率: 0.598000 (59.80%)
ACC_RE = re.compile(r"准确率:\s*([0-9.]+)")
# 攻击成功率 (ASR): 0.189950 (18.99%)
ASR_RE = re.compile(r"攻击成功率\s*\(ASR\):\s*([0-9.]+)")
# 攻击类型: adaptive_blend
ATTACK_RE = re.compile(r"攻击类型:\s*(\S+)")


@dataclass
class DomainResult:
    acc: Optional[float] = None
    asr: Optional[float] = None
    path: Optional[str] = None


def _parse_result_file(path: str) -> DomainResult:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return DomainResult(path=path)
    acc_m = ACC_RE.search(text)
    asr_m = ASR_RE.search(text)
    return DomainResult(
        acc=float(acc_m.group(1)) if acc_m else None,
        asr=float(asr_m.group(1)) if asr_m else None,
        path=path,
    )


def _attack_from_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = ATTACK_RE.search(f.read())
        return m.group(1) if m else None
    except OSError:
        return None


@dataclass
class PairedRow:
    model_dir: str
    config: str
    attack: str
    imagenetv2: DomainResult
    qwen: DomainResult


def _list_result_files(model_dir: str, prefix: str) -> list:
    """Return result files in model_dir whose name starts with prefix and ends with .txt.

    For the ImageNetV2 prefix we must exclude the qwen files, since
    'test_tiny_target_domain_results' is NOT a prefix of the qwen file name
    ('test_tiny_target_domain_qwen_results'), so a startswith check is safe.
    """
    out = []
    if not os.path.isdir(model_dir):
        return out
    for name in sorted(os.listdir(model_dir)):
        if name.startswith(prefix) and name.endswith(".txt"):
            out.append(os.path.join(model_dir, name))
    return out


def _variant_suffix(filename: str, base_prefix: str) -> str:
    """Extract the variant tag, e.g. '_test_alpha=0.2' or '' for the base file."""
    stem = filename[:-len(".txt")] if filename.endswith(".txt") else filename
    return stem[len(base_prefix):] if stem.startswith(base_prefix) else ""


def collect_pairs(model_dir: str) -> list:
    """Pair imagenetv2 and qwen result files within a single model dir by variant tag."""
    iv2_prefix = "test_tiny_target_domain_results"
    qwen_prefix = "test_tiny_target_domain_qwen_results"

    iv2_files = {
        _variant_suffix(os.path.basename(p), iv2_prefix): p
        for p in _list_result_files(model_dir, iv2_prefix)
        # exclude qwen files which would also match the iv2 prefix only if
        # startswith were loose; here prefixes differ so this is just defensive
        if not os.path.basename(p).startswith(qwen_prefix)
    }
    qwen_files = {
        _variant_suffix(os.path.basename(p), qwen_prefix): p
        for p in _list_result_files(model_dir, qwen_prefix)
    }

    config = os.path.basename(model_dir.rstrip("/"))
    rows = []
    all_variants = sorted(set(iv2_files) | set(qwen_files))
    for variant in all_variants:
        iv2_path = iv2_files.get(variant)
        qwen_path = qwen_files.get(variant)
        iv2 = _parse_result_file(iv2_path) if iv2_path else DomainResult()
        qwen = _parse_result_file(qwen_path) if qwen_path else DomainResult()
        attack = (
            (_attack_from_file(iv2_path) if iv2_path else None)
            or (_attack_from_file(qwen_path) if qwen_path else None)
            or config.split("_")[0]
        )
        rows.append(PairedRow(
            model_dir=model_dir,
            config=config + variant,
            attack=attack,
            imagenetv2=iv2,
            qwen=qwen,
        ))
    return rows


def walk_dataset(poisoned_root: str, dataset: str) -> list:
    base = os.path.join(poisoned_root, dataset)
    rows = []
    if not os.path.isdir(base):
        return rows
    for name in sorted(os.listdir(base)):
        model_dir = os.path.join(base, name)
        if os.path.isdir(model_dir):
            rows.extend(collect_pairs(model_dir))
    return rows


def _fmt(x: Optional[float]) -> str:
    return f"{x * 100:.2f}%" if x is not None else "—"


def _delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "—"
    return f"{(b - a) * 100:+.2f}pp"


def render_markdown(rows: list) -> str:
    paired = [r for r in rows if r.imagenetv2.asr is not None and r.qwen.asr is not None]
    only_one = [r for r in rows if (r.imagenetv2.asr is None) ^ (r.qwen.asr is None)]

    lines = []
    lines.append("| 配置 | 攻击 | IV2 ACC | Qwen ACC | ΔACC | IV2 ASR | Qwen ASR | ΔASR |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in paired:
        lines.append(
            f"| `{r.config}` | {r.attack} | {_fmt(r.imagenetv2.acc)} | {_fmt(r.qwen.acc)} | "
            f"{_delta(r.imagenetv2.acc, r.qwen.acc)} | {_fmt(r.imagenetv2.asr)} | "
            f"{_fmt(r.qwen.asr)} | {_delta(r.imagenetv2.asr, r.qwen.asr)} |"
        )

    summary = [
        "",
        f"- 配对成功（两个目标域都有结果）: {len(paired)} 条",
        f"- 仅单侧有结果（待补测）: {len(only_one)} 条",
    ]
    if paired:
        def _avg(getter):
            vals = [getter(r) for r in paired if getter(r) is not None]
            return sum(vals) / len(vals) if vals else None
        summary += [
            f"- IV2 平均 ASR: {_fmt(_avg(lambda r: r.imagenetv2.asr))}, "
            f"Qwen 平均 ASR: {_fmt(_avg(lambda r: r.qwen.asr))}",
            f"- IV2 平均 ACC: {_fmt(_avg(lambda r: r.imagenetv2.acc))}, "
            f"Qwen 平均 ACC: {_fmt(_avg(lambda r: r.qwen.acc))}",
        ]
    if only_one:
        summary.append("")
        summary.append("仅单侧有结果的配置：")
        for r in only_one:
            side = "仅 IV2" if r.qwen.asr is None else "仅 Qwen"
            summary.append(f"  - `{r.config}` ({side})")

    return "\n".join(lines + summary)


def main():
    parser = argparse.ArgumentParser(description="对比 ImageNetV2 与 Qwen 目标域迁移结果")
    parser.add_argument("--poisoned-root", default="poisoned_train_set1",
                        help="投毒实验结果根目录（默认 poisoned_train_set1）")
    parser.add_argument("--dataset", default="tiny_imagenet")
    parser.add_argument("--model-dir", default=None,
                        help="只比较单个模型目录（单配置验证）")
    parser.add_argument("--output", default=None,
                        help="输出 Markdown 文件路径；缺省时打印到标准输出")
    args = parser.parse_args()

    if args.model_dir:
        rows = collect_pairs(args.model_dir)
    else:
        rows = walk_dataset(args.poisoned_root, args.dataset)

    if not rows:
        print("未找到任何结果文件（检查 --poisoned-root / --dataset / --model-dir）", file=sys.stderr)
        sys.exit(1)

    md = render_markdown(rows)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("# Tiny Target Domain: ImageNetV2 vs Qwen 对比\n\n")
            f.write(md + "\n")
        print(f"报告已写入: {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""从 poisoned_train_set* 提取隐蔽性和迁移性，按数据集和模型分别输出（默认 poisoned_train_set1）。

迁移性定义（transfer_rate）：
  transfer_rate = asr * transfer_asr / (transfer_asr + asr)
  - asr：源域（训练数据集）测试 ASR，来自 test_results*.json
  - transfer_asr：目标域 test-ASR（STL-10 / target-domain / MNIST）
  分析阶段另做 ASR>30% 筛选（见 analysis_filters.py），提取不过滤。

两种模式 (--mode)：
  no_nc: 仅提取隐蔽性(TPR/AUC)、迁移性、ASR，不提取 NC
  nc:    提取隐蔽性、迁移性、ASR、NC，并计算综合隐蔽性 S（AUC 与 TPR 两套）
         NC 项（仅 is_poisoned=True）：在当次全量结果上对 MaxNorm min-max 后取反再乘 0.2，
           nc_part = 0.2 * (max - val) / (max - min)，异常指数越大 NC 贡献越小（与「越大越易感」一致）
         S_stealth     = 0.8 * stealth_auc_avg + nc_part
         S_stealth_tpr = 0.8 * stealth_tpr_avg + nc_part
         is_poisoned=False 时 nc_part = 0

输出：data_{dataset}_{arch}_{suffix}.csv/json
  no_nc -> _no_nc,  nc -> _nc

数据集分支：
  - cifar10：迁移率仅来自 test_stl10_results*.txt（STL-10）；不读取 test_tiny_imagenet_*。
  - tiny_imagenet：迁移率仅来自 test_tiny_target_domain_results*.txt（target-domain）。
  - mnistm：迁移率来自 test_mnistm_results*.txt（主）或 test_mnist_cross_results*.txt（兼容旧命名）。

防御结果 *defense_results*.json 与上述迁移结果文件，均通过同一套文件名规则解析 test 维度
（兼容原 results_* 命名，并支持 test_* 后缀）。

分组键（cifar10 与 tiny 相同）：attack_type + poison_rate + train_param_value + alpha + cover_rate，
避免 belt 等「仅 alpha/cover 不同」的目录被合并；CSV 含 alpha 列。
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

DEFENSE_METHODS = ["STRIP", "SCaLe-Up", "SentiNet", "IBD_PSC"]
DEFENSE_PREFIX = {"STRIP": "strip", "SCaLe-Up": "scaleup", "SentiNet": "sentinet", "IBD_PSC": "ibd_psc"}

# 实验结果根目录（相对仓库根目录）；与 validate / build_non_nc_richer_report 保持一致
DEFAULT_POISONED_ROOT = "poisoned_train_set1"

# 数据集与模型（arch）映射：文件夹中的 arch 名 -> 输出文件名
DATASETS = ["cifar10", "tiny_imagenet", "mnistm"]
ARCH_TO_OUTPUT = {"ResNet18": "resnet18", "mobilenetv2": "mobilenet", "vgg19_bn": "vgg"}

def get_training_param_type(attack_type: str) -> Optional[str]:
    attack_lower = (attack_type or '').lower()
    if 'upgd' in attack_lower:
        return 'eps'
    if 'wanet' in attack_lower:
        return 's'
    if 'sig' in attack_lower:
        return 'delta'
    if 'belt' in attack_lower:
        return 'mask_rate'
    if any(t in attack_lower for t in ['blend', 'badnet', 'basic', 'patch']):
        return 'alpha'
    return None


def get_test_param_type(train_param_type: str) -> Optional[str]:
    m = {'alpha': 'test_alpha', 's': 'test_s', 'delta': 'test_delta', 'eps': 'test_eps', 'mask_rate': 'test_mask_rate'}
    return m.get(train_param_type)


def parse_folder_name(folder_name: str) -> Dict[str, Any]:
    params = {}
    patterns = [
        (r'^adaptive_blend', 'adaptive_blend'), (r'^adaptive_patch', 'adaptive_patch'),
        (r'^belt', 'belt'), (r'^upgd', 'upgd'), (r'^WaNet', 'WaNet'), (r'^SIG', 'SIG'),
        (r'^basic', 'basic'), (r'^badnet', 'badnet'), (r'^blend', 'blend'),
    ]
    attack_type = None
    for pat, atype in patterns:
        if re.match(pat, folder_name):
            attack_type = atype
            break
    params['attack_type'] = attack_type or 'unknown'
    if m := re.search(r'^.+?_(\d+\.?\d*)_', folder_name):
        params['poison_rate'] = float(m.group(1))
    for k, pat in [('alpha', r'alpha=([\d.]+)'), ('cover_rate', r'cover=([\d.]+)'), ('delta', r'delta=([\d.]+)'),
                   ('s', r's=([\d.]+)'), ('eps', r'eps=([\d.]+)'), ('mask_rate', r'mask=([\d.]+)')]:
        if m := re.search(pat, folder_name):
            params[k] = float(m.group(1))
    # 解析 arch：arch=ResNet18_cifar10 或 arch=mobilenetv2_cifar10
    if m := re.search(r'arch=([\w]+)_(cifar10|tiny_imagenet|mnistm)', folder_name):
        params['arch_raw'] = m.group(1)
        params['dataset_from_folder'] = m.group(2)
    return params


def _parse_defense_json(path: Path) -> Optional[Dict[str, float]]:
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
        tpr = float(d.get('tpr', 0) or 0)
        if tpr > 1:
            tpr /= 100.0
        auc = float(d.get('auc', 0) or 0)
        return {"tpr": tpr, "auc": auc}
    except Exception:
        return None


def _parse_auxiliary_result_filename(name: str) -> Optional[tuple]:
    """从辅助结果文件名解析 test 参数：*defense_results*.json、test_stl10_results*.txt、test_tiny_imagenet_results*.txt。"""
    patterns = [
        (r'test_alpha=([\d.]+)', 'test_alpha'),
        (r'test_s=([\d.]+)', 'test_s'),
        (r'test_delta=([\d.]+)', 'test_delta'),
        (r'test_eps=([\d.]+)', 'test_eps'),
        (r'test_mask_rate=([\d.]+)', 'test_mask_rate'),
        (r'results_alpha=([\d.]+)', 'test_alpha'),
        (r'results_s=([\d.]+)', 'test_s'),
        (r'results_delta=([\d.]+)', 'test_delta'),
        (r'results_eps=([\d.]+)', 'test_eps'),
        (r'results_mask_rate=([\d.]+)', 'test_mask_rate'),
    ]
    for pat, ptype in patterns:
        if m := re.search(pat, name, re.I):
            return ptype, float(m.group(1).rstrip('.'))
    return None


def _extract_cross_dataset_param_from_filename(name: str) -> Optional[tuple]:
    """兼容 validate_extraction 等外部引用；与 _parse_auxiliary_result_filename 相同。"""
    return _parse_auxiliary_result_filename(name)


def _extract_param_from_filename(name: str) -> Optional[tuple]:
    """兼容旧名。"""
    return _parse_auxiliary_result_filename(name)


def _extract_param_from_test_results(name: str) -> Optional[tuple]:
    """解析 test_results_seed=2333_delta=4.0.json 等文件名中的参数"""
    patterns = [
        (r'[_-]alpha=([\d.]+)', 'test_alpha'), (r'[_-]s=([\d.]+)', 'test_s'),
        (r'[_-]delta=([\d.]+)', 'test_delta'), (r'[_-]eps=([\d.]+)', 'test_eps'),
        (r'[_-]mask_rate=([\d.]+)', 'test_mask_rate'),
    ]
    for pat, ptype in patterns:
        if m := re.search(pat, name, re.I):
            return ptype, float(m.group(1).rstrip('.'))
    return None


def _normalize_rate(v: Optional[float]) -> Optional[float]:
    from analysis_filters import normalize_asr_value
    return normalize_asr_value(v)


def _compute_transfer_metric(target_asr: Optional[float], source_asr: Optional[float]) -> Optional[float]:
    from analysis_filters import compute_transfer_metric
    return compute_transfer_metric(target_asr, source_asr)


# 兼容 validate_extraction 等旧引用
_compute_transfer_ratio = _compute_transfer_metric


def _parse_transfer_rate_from_text(c: str) -> Optional[float]:
    """从测试结果文本解析目标域攻击成功率 ASR（兼容 STL-10 与 target-domain 等写法）。"""
    for line in c.splitlines():
        line = line.strip()
        if not line:
            continue
        if '攻击成功率' in line or ('ASR' in line and ':' in line):
            m = re.search(r'攻击成功率(?:\s*\([^)]*\))?[:：]\s*([\d.]+)', line)
            if m:
                return float(m.group(1))
            if '攻击成功率' in line or 'ASR' in line:
                m = re.search(r'([0-9.]+)\s*\([0-9.]*%?\)', line)
                if m:
                    return float(m.group(1))
    if m := re.search(r'攻击成功率[:：]\s*([\d.]+)', c):
        return float(m.group(1))
    return None


def extract_folder_results(
    folder: Path, params: Dict[str, Any], include_nc: bool = True, dataset: str = "cifar10"
) -> List[Dict[str, Any]]:
    attack_type = params.get('attack_type')
    train_param_type = get_training_param_type(attack_type)
    train_param_value = params.get(train_param_type) if train_param_type else None
    test_param_type = get_test_param_type(train_param_type) if train_param_type else None

    test_params: Dict[float, Dict[str, Any]] = {}

    # 1. 从防御文件收集 pval 及 tpr/auc（cifar10 / tiny 共用同一解析规则）
    for method, prefix in DEFENSE_PREFIX.items():
        for f in folder.glob(f"{prefix}_defense_results*.json"):
            param_info = _parse_auxiliary_result_filename(f.name)
            if param_info:
                ptype, pval = param_info
                if pval not in test_params:
                    test_params[pval] = {'param_type': ptype, 'defense_tprs': {}, 'defense_aucs': {}}
                if rec := _parse_defense_json(f):
                    test_params[pval]['defense_tprs'][method] = rec['tpr']
                    test_params[pval]['defense_aucs'][method] = rec['auc']
        # 无后缀 base 文件
        base = folder / f"{prefix}_defense_results.json"
        if base.exists() and train_param_value is not None and test_param_type:
            pval = train_param_value
            if pval not in test_params:
                test_params[pval] = {'param_type': test_param_type, 'defense_tprs': {}, 'defense_aucs': {}}
            if rec := _parse_defense_json(base):
                test_params[pval]['defense_tprs'][method] = rec['tpr']
                test_params[pval]['defense_aucs'][method] = rec['auc']

    # 2. 迁移性：
    #    - cifar10：仅 STL-10
    #    - tiny_imagenet：仅 Tiny Target Domain
    #    - mnistm：MNIST 迁移结果（test_mnistm_results；兼容旧 test_mnist_cross_results）
    if dataset == "tiny_imagenet":
        tiny_target_files = list(folder.glob("test_tiny_target_domain_results*.txt"))
        for f in tiny_target_files:
            param_info = _parse_auxiliary_result_filename(f.name)
            if param_info:
                _, pval = param_info
            elif train_param_value is not None:
                pval = train_param_value
            else:
                continue

            try:
                text = f.read_text(encoding='utf-8')
                tr = _parse_transfer_rate_from_text(text)
            except Exception:
                continue
            if tr is None:
                continue
            if pval in test_params:
                test_params[pval]['transfer_asr'] = tr
    elif dataset == "mnistm":
        # mnistm：优先新命名 test_mnistm_results*.txt，同时兼容旧命名 test_mnist_cross_results*.txt
        mnist_files = list(folder.glob("test_mnistm_results*.txt"))
        if not mnist_files:
            mnist_files = list(folder.glob("test_mnist_cross_results*.txt"))
        for f in mnist_files:
            param_info = _parse_auxiliary_result_filename(f.name)
            if param_info:
                _, pval = param_info
                if pval in test_params:
                    try:
                        c = f.read_text(encoding='utf-8')
                        tr = _parse_transfer_rate_from_text(c)
                        if tr is not None:
                            test_params[pval]['transfer_asr'] = tr
                    except Exception:
                        pass
        base_mnist = folder / "test_mnistm_results.txt"
        base_mnist_cross = folder / "test_mnist_cross_results.txt"
        for base_file in [base_mnist, base_mnist_cross]:
            if base_file.exists() and train_param_value is not None and train_param_value in test_params:
                try:
                    c = base_file.read_text(encoding='utf-8')
                    tr = _parse_transfer_rate_from_text(c)
                    if tr is not None:
                        test_params[train_param_value]['transfer_asr'] = tr
                except Exception:
                    pass
    else:
        # cifar10（及将来其它非 tiny 数据集）：仅匹配 STL-10 迁移结果，不混入 Tiny-C 文件
        for f in folder.glob("test_stl10_results*.txt"):
            param_info = _parse_auxiliary_result_filename(f.name)
            if param_info:
                _, pval = param_info
                if pval in test_params:
                    try:
                        c = f.read_text(encoding='utf-8')
                        tr = _parse_transfer_rate_from_text(c)
                        if tr is not None:
                            test_params[pval]['transfer_asr'] = tr
                    except Exception:
                        pass
        base_stl10 = folder / "test_stl10_results.txt"
        if base_stl10.exists() and train_param_value is not None and train_param_value in test_params:
            try:
                c = base_stl10.read_text(encoding='utf-8')
                tr = _parse_transfer_rate_from_text(c)
                if tr is not None:
                    test_params[train_param_value]['transfer_asr'] = tr
            except Exception:
                pass

    # 2b. ASR：test_results_seed=*.json 中的 asr（同数据集攻击成功率）
    for f in folder.glob("test_results*.json"):
        param_info = _extract_param_from_test_results(f.name)
        pval = param_info[1] if param_info else train_param_value
        if pval is not None and pval in test_params:
            try:
                d = json.loads(f.read_text(encoding='utf-8'))
                asr = d.get('asr')
                if asr is not None:
                    test_params[pval]['asr'] = float(asr)
            except Exception:
                pass

    # 3. NC detection（仅 include_nc 时提取）
    if include_nc:
        nc_data = None
        for f in folder.glob("nc_detection*.json"):
            try:
                d = json.loads(f.read_text(encoding='utf-8'))
                nc_data = {
                    'max_anomaly_index': float(d['max_anomaly_index']) if d.get('max_anomaly_index') is not None else None,
                    'is_poisoned': d.get('is_poisoned')
                }
                break
            except Exception:
                pass
        if nc_data:
            for pval in test_params:
                test_params[pval]['nc_max_anomaly_index'] = nc_data['max_anomaly_index']
                test_params[pval]['nc_is_poisoned'] = nc_data['is_poisoned']

    # 4. 构建结果：至少有一种防御数据；仅对有数据的防御方法取平均
    results = []
    for pval in sorted(test_params.keys()):
        d = test_params[pval]
        tprs = {m: d['defense_tprs'].get(m, 0.0) for m in DEFENSE_METHODS}
        aucs = {m: d['defense_aucs'].get(m, 0.0) for m in DEFENSE_METHODS}
        methods_with_tpr = [m for m in DEFENSE_METHODS if m in d['defense_tprs']]
        methods_with_auc = [m for m in DEFENSE_METHODS if m in d['defense_aucs']]
        if not methods_with_tpr and not methods_with_auc:
            continue
        # 隐蔽性 = 1 - 检测率（TPR/AUC 越高越易被检测，隐蔽性越低）
        raw_tpr_avg = np.mean([tprs[m] for m in methods_with_tpr]) if methods_with_tpr else 0.0
        raw_auc_avg = np.mean([aucs[m] for m in methods_with_auc]) if methods_with_auc else 0.0
        stealth_tpr_avg = 1.0 - raw_tpr_avg
        stealth_auc_avg = 1.0 - raw_auc_avg
        transfer_asr = d.get('transfer_asr')
        asr = d.get('asr')
        transfer_rate = _compute_transfer_metric(transfer_asr, asr)
        if transfer_rate is None:
            continue
        row = {
            **params,
            'train_param_value': train_param_value,
            'test_param_value': pval,
            'test_param_type': d.get('param_type'),
            'stealth_tpr_avg': stealth_tpr_avg,
            'stealth_auc_avg': stealth_auc_avg,
            'transfer_asr': float(transfer_asr) if transfer_asr is not None else None,
            'transfer_rate': float(transfer_rate),
            'asr': float(asr) if asr is not None else None,
        }
        if include_nc:
            row['nc_max_anomaly_index'] = d.get('nc_max_anomaly_index')
            row['nc_is_poisoned'] = d.get('nc_is_poisoned')
        results.append(row)
    return results


def compute_s_stealth(all_results: List[Dict[str, Any]]) -> None:
    """为 nc 模式计算 S_stealth（AUC 版）与 S_stealth_tpr（TPR 版），就地修改 all_results。
    min/max 仅对 is_poisoned=True 且指数有效的行统计；nc_part = 0.2 * (max - val) / (max - min)。
    is_poisoned=False 时 nc_part = 0。
    """
    max_norms = [r['nc_max_anomaly_index'] for r in all_results
                 if r.get('nc_is_poisoned') and r.get('nc_max_anomaly_index') is not None]
    max_norm_max = max(max_norms) if max_norms else 1.0
    max_norm_min = min(max_norms) if max_norms else 0.0
    denom = max_norm_max - max_norm_min if max_norm_max > max_norm_min else 1.0

    for r in all_results:
        auc_part = 0.8 * (r.get('stealth_auc_avg') or 0.0)
        tpr_part = 0.8 * (r.get('stealth_tpr_avg') or 0.0)
        if r.get('nc_is_poisoned') and r.get('nc_max_anomaly_index') is not None:
            val = r['nc_max_anomaly_index']
            nc_part = 0.2 * (max_norm_max - val) / denom
        else:
            nc_part = 0.0
        r['S_stealth'] = auc_part + nc_part
        r['S_stealth_tpr'] = tpr_part + nc_part


def _experiment_group_key(r: Dict[str, Any]) -> tuple:
    """区分不同实验目录：除攻击类型/中毒率/训练触发强度外，纳入 alpha、cover_rate（belt 等多目录仅靠 alpha 区分）。"""
    def _n(x):
        if x is None:
            return None
        try:
            return round(float(x), 8)
        except (TypeError, ValueError):
            return x

    return (
        r.get('attack_type'),
        _n(r.get('poison_rate', 0.03)),
        _n(r.get('train_param_value')),
        _n(r.get('alpha')),
        _n(r.get('cover_rate')),
    )


def _experiment_group_sort_key(key: tuple) -> tuple:
    """稳定排序且避免 None 与 float 比较报错。"""
    at, pr, tv, al, cr = key

    def _s(x):
        if x is None:
            return (-1, 0.0)
        return (0, float(x))

    return (at or '', _s(pr), _s(tv), _s(al), _s(cr))


def convert_to_data_groups(all_results: List[Dict[str, Any]], include_nc: bool = True) -> List[Dict[str, Any]]:
    """cifar10 与 tiny_imagenet 共用分组键（含 alpha、cover_rate），与按数据集分支的迁移率提取独立。"""
    base_keys = [
        'stealth_tpr_avg', 'stealth_auc_avg', 'transfer_asr', 'transfer_rate', 'asr',
        'test_param_value', 'test_param_type', 'alpha', 'cover_rate',
    ]
    nc_keys = ['nc_max_anomaly_index', 'nc_is_poisoned', 'S_stealth', 'S_stealth_tpr'] if include_nc else []
    point_keys = base_keys + nc_keys

    groups_dict: Dict[tuple, List] = {}
    for r in all_results:
        key = _experiment_group_key(r)
        groups_dict.setdefault(key, []).append(r)
    groups = []
    for gid, (key, entries) in enumerate(sorted(groups_dict.items(), key=lambda x: _experiment_group_sort_key(x[0])), 1):
        entries.sort(key=lambda e: (e.get('cover_rate') or -1, e.get('test_param_value', 0)))
        pts = [{'group_id': gid, 'point_id': i, **{k: e.get(k) for k in point_keys}} for i, e in enumerate(entries, 1)]
        groups.append({
            'group_id': gid,
            'attack_type': entries[0]['attack_type'],
            'trigger_type': key[0],
            'poison_rate': key[1],
            'train_param_value': key[2],
            'alpha': entries[0].get('alpha'),
            'cover_rate': entries[0].get('cover_rate'),
            'group_size': len(entries),
            'data_points': pts,
        })
    return groups


def export_csv(groups: List[Dict], path: Path, dataset: str = "", arch: str = "", include_nc: bool = True):
    base_fn = ["dataset", "arch", "group_id", "point_id", "attack_type", "trigger_type", "poison_rate", "train_param_value", "alpha",
               "test_param_type", "test_param_value", "stealth_tpr_avg", "stealth_auc_avg", "transfer_asr", "transfer_rate", "asr", "cover_rate"]
    nc_fn = ["nc_max_anomaly_index", "nc_is_poisoned", "S_stealth", "S_stealth_tpr"] if include_nc else []
    fn = base_fn + nc_fn

    def fmt(v):
        if v is None: return ""
        if isinstance(v, bool): return str(v).lower()
        return v

    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for g in groups:
            base = {"dataset": dataset, "arch": arch, "group_id": g["group_id"], "attack_type": g["attack_type"],
                    "trigger_type": g.get("trigger_type"), "poison_rate": g.get("poison_rate"),
                    "train_param_value": g.get("train_param_value"), "alpha": g.get("alpha")}
            for p in g["data_points"]:
                row = {**base, "point_id": p["point_id"], "test_param_type": p.get("test_param_type"),
                       "test_param_value": p.get("test_param_value"), "stealth_tpr_avg": p.get("stealth_tpr_avg"),
                       "stealth_auc_avg": p.get("stealth_auc_avg"), "transfer_asr": p.get("transfer_asr"),
                       "transfer_rate": p.get("transfer_rate"),
                       "asr": p.get("asr"), "cover_rate": p.get("cover_rate")}
                if include_nc:
                    row["nc_max_anomaly_index"] = p.get("nc_max_anomaly_index")
                    row["nc_is_poisoned"] = p.get("nc_is_poisoned")
                    row["S_stealth"] = p.get("S_stealth")
                    row["S_stealth_tpr"] = p.get("S_stealth_tpr")
                w.writerow({k: fmt(v) for k, v in row.items()})


def export_json(groups: List[Dict], path: Path, dataset: str = "", arch: str = ""):
    out = {"dataset": dataset, "arch": arch, "groups": groups,
           "summary": {"total_groups": len(groups), "total_points": sum(g["group_size"] for g in groups)}}
    with path.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='提取隐蔽性、迁移性、ASR，可选 NC 与 S_stealth')
    parser.add_argument('--mode', choices=['no_nc', 'nc', 'all'], default='all',
                        help='no_nc: 仅无 NC 版；nc: 仅 NC 版；all: 两种都提取（默认）')
    parser.add_argument('--poisoned-root', type=str, default=DEFAULT_POISONED_ROOT,
                        help=f'投毒实验结果根目录名或绝对路径（默认: {DEFAULT_POISONED_ROOT}）')
    args = parser.parse_args()

    modes = ['no_nc', 'nc'] if args.mode == 'all' else [args.mode]

    base = Path(__file__).parent.parent
    poisoned_root = Path(args.poisoned_root)
    if not poisoned_root.is_absolute():
        poisoned_root = base / poisoned_root
    out_dir = Path(__file__).parent
    if not poisoned_root.exists():
        print(f"路径不存在: {poisoned_root}")
        return

    for mode_nc in [m == 'nc' for m in modes]:
        suffix = '_nc' if mode_nc else '_no_nc'
        mode_name = 'nc' if mode_nc else 'no_nc'
        for dataset in DATASETS:
            ds_path = poisoned_root / dataset
            if not ds_path.exists() or not ds_path.is_dir():
                continue
            folders = [f for f in ds_path.iterdir() if f.is_dir()]
            for arch_in_folder, arch_out in ARCH_TO_OUTPUT.items():
                arch_folders = []
                for folder in folders:
                    if folder.name.startswith('none_'):
                        continue
                    params = parse_folder_name(folder.name)
                    if params.get('attack_type') == 'none':
                        continue
                    if params.get('arch_raw') != arch_in_folder:
                        continue
                    if params.get('dataset_from_folder') != dataset:
                        continue
                    arch_folders.append((folder, params))

                if not arch_folders:
                    continue
                all_results = []
                for folder, params in arch_folders:
                    res = extract_folder_results(folder, params, include_nc=mode_nc, dataset=dataset)
                    if res:
                        all_results.extend(res)
                if not all_results:
                    continue
                if mode_nc:
                    compute_s_stealth(all_results)
                groups = convert_to_data_groups(all_results, include_nc=mode_nc)
                csv_path = out_dir / f"data_{dataset}_{arch_out}{suffix}.csv"
                json_path = out_dir / f"data_{dataset}_{arch_out}{suffix}.json"
                export_csv(groups, csv_path, dataset=dataset, arch=arch_out, include_nc=mode_nc)
                export_json(groups, json_path, dataset=dataset, arch=arch_out)
                print(f"✓ [{mode_name}] {dataset}/{arch_out}: {csv_path.name} ({len(groups)} 组, {len(all_results)} 点)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""从 poisoned_train_set 提取隐蔽性和迁移性，按数据集和模型分别输出。

两种模式 (--mode)：
  no_nc: 仅提取隐蔽性(TPR/AUC)、迁移性、ASR，不提取 NC
  nc:    提取隐蔽性、迁移性、ASR、NC，并计算 S_stealth 综合隐蔽性
         S_stealth = 0.8 * (1/4)*Σ(1-AUC_i) + 0.2 * norm(MaxNorm)
         is_poisoned=False 时 NC 加权项为 0

输出：data_{dataset}_{arch}_{suffix}.csv/json
  no_nc -> _no_nc,  nc -> _nc
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

# 数据集与模型（arch）映射：文件夹中的 arch 名 -> 输出文件名
DATASETS = ["cifar10", "tiny_imagenet"]
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


def _extract_param_from_filename(name: str) -> Optional[tuple]:
    """解析 test_stl10_results_delta=4.0.txt 等文件名中的参数"""
    patterns = [
        (r'results_alpha=([\d.]+)', 'test_alpha'), (r'results_s=([\d.]+)', 'test_s'),
        (r'results_delta=([\d.]+)', 'test_delta'), (r'results_eps=([\d.]+)', 'test_eps'),
        (r'results_mask_rate=([\d.]+)', 'test_mask_rate'),
    ]
    for pat, ptype in patterns:
        if m := re.search(pat, name, re.I):
            return ptype, float(m.group(1).rstrip('.'))
    return None


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


def extract_folder_results(folder: Path, params: Dict[str, Any], include_nc: bool = True) -> List[Dict[str, Any]]:
    attack_type = params.get('attack_type')
    train_param_type = get_training_param_type(attack_type)
    train_param_value = params.get(train_param_type) if train_param_type else None
    test_param_type = get_test_param_type(train_param_type) if train_param_type else None

    test_params: Dict[float, Dict[str, Any]] = {}

    # 1. 从防御文件收集 pval 及 tpr/auc
    for method, prefix in DEFENSE_PREFIX.items():
        for f in folder.glob(f"{prefix}_defense_results*.json"):
            param_info = _extract_param_from_filename(f.name)
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

    # 2. 迁移性：test_stl10_results_delta=4.0.txt 等
    for f in folder.glob("test_stl10_results*.txt"):
        param_info = _extract_param_from_filename(f.name)
        if param_info:
            _, pval = param_info
            if pval in test_params:
                try:
                    c = f.read_text(encoding='utf-8')
                    if m := re.search(r'攻击成功率[:：]\s*([\d.]+)', c):
                        test_params[pval]['transfer_rate'] = float(m.group(1))
                except Exception:
                    pass
    base_stl10 = folder / "test_stl10_results.txt"
    if base_stl10.exists() and train_param_value is not None and train_param_value in test_params:
        try:
            c = base_stl10.read_text(encoding='utf-8')
            if m := re.search(r'攻击成功率[:：]\s*([\d.]+)', c):
                test_params[train_param_value]['transfer_rate'] = float(m.group(1))
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
        transfer = float(d.get('transfer_rate', 0.0))
        asr = d.get('asr')
        row = {
            **params,
            'train_param_value': train_param_value,
            'test_param_value': pval,
            'test_param_type': d.get('param_type'),
            'stealth_tpr_avg': stealth_tpr_avg,
            'stealth_auc_avg': stealth_auc_avg,
            'transfer_rate': transfer,
            'asr': float(asr) if asr is not None else None,
        }
        if include_nc:
            row['nc_max_anomaly_index'] = d.get('nc_max_anomaly_index')
            row['nc_is_poisoned'] = d.get('nc_is_poisoned')
        results.append(row)
    return results


def compute_s_stealth(all_results: List[Dict[str, Any]]) -> None:
    """为 nc 模式计算 S_stealth，就地修改 all_results。
    S_stealth = 0.8 * (1/4)*Σ(1-AUC_i) + 0.2 * (MaxNorm-min)/(max-min)
    is_poisoned=False 时，将 max_norm 视为最小值（归一化后为 0）。
    """
    max_norms = [r['nc_max_anomaly_index'] for r in all_results
                 if r.get('nc_is_poisoned') and r.get('nc_max_anomaly_index') is not None]
    max_norm_min = min(max_norms) if max_norms else 0.0
    max_norm_max = max(max_norms) if max_norms else 1.0
    denom = max_norm_max - max_norm_min if max_norm_max > max_norm_min else 1.0

    for r in all_results:
        auc_part = 0.8 * (r.get('stealth_auc_avg') or 0.0)
        # is_poisoned=False 时，max_norm 视为最小值
        if r.get('nc_is_poisoned') and r.get('nc_max_anomaly_index') is not None:
            norm_val = r['nc_max_anomaly_index']
        else:
            norm_val = max_norm_min
        nc_part = 0.2 * (norm_val - max_norm_min) / denom
        r['S_stealth'] = auc_part + nc_part


def convert_to_data_groups(all_results: List[Dict[str, Any]], include_nc: bool = True) -> List[Dict[str, Any]]:
    base_keys = ['stealth_tpr_avg', 'stealth_auc_avg', 'transfer_rate', 'asr', 'test_param_value', 'test_param_type', 'cover_rate']
    nc_keys = ['nc_max_anomaly_index', 'nc_is_poisoned', 'S_stealth'] if include_nc else []
    point_keys = base_keys + nc_keys

    groups_dict: Dict[tuple, List] = {}
    for r in all_results:
        tt = r['attack_type']
        key = (tt, r.get('poison_rate', 0.03), r.get('train_param_value'))
        groups_dict.setdefault(key, []).append(r)
    groups = []
    for gid, (key, entries) in enumerate(sorted(groups_dict.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or -1)), 1):
        entries.sort(key=lambda e: (e.get('cover_rate') or -1, e.get('test_param_value', 0)))
        pts = [{'group_id': gid, 'point_id': i, **{k: e.get(k) for k in point_keys}} for i, e in enumerate(entries, 1)]
        groups.append({'group_id': gid, 'attack_type': entries[0]['attack_type'], 'trigger_type': key[0], 'poison_rate': key[1], 'train_param_value': key[2], 'group_size': len(entries), 'data_points': pts})
    return groups


def export_csv(groups: List[Dict], path: Path, dataset: str = "", arch: str = "", include_nc: bool = True):
    base_fn = ["dataset", "arch", "group_id", "point_id", "attack_type", "trigger_type", "poison_rate", "train_param_value",
               "test_param_type", "test_param_value", "stealth_tpr_avg", "stealth_auc_avg", "transfer_rate", "asr", "cover_rate"]
    nc_fn = ["nc_max_anomaly_index", "nc_is_poisoned", "S_stealth"] if include_nc else []
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
                    "train_param_value": g.get("train_param_value")}
            for p in g["data_points"]:
                row = {**base, "point_id": p["point_id"], "test_param_type": p.get("test_param_type"),
                       "test_param_value": p.get("test_param_value"), "stealth_tpr_avg": p.get("stealth_tpr_avg"),
                       "stealth_auc_avg": p.get("stealth_auc_avg"), "transfer_rate": p.get("transfer_rate"),
                       "asr": p.get("asr"), "cover_rate": p.get("cover_rate")}
                if include_nc:
                    row["nc_max_anomaly_index"] = p.get("nc_max_anomaly_index")
                    row["nc_is_poisoned"] = p.get("nc_is_poisoned")
                    row["S_stealth"] = p.get("S_stealth")
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
    args = parser.parse_args()

    modes = ['no_nc', 'nc'] if args.mode == 'all' else [args.mode]

    base = Path(__file__).parent.parent
    poisoned_root = base / "poisoned_train_set"
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
                    res = extract_folder_results(folder, params, include_nc=mode_nc)
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

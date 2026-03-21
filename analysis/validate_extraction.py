#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证提取结果的正确性。支持两种模式：
  no_nc: 验证 data_*_no_nc 文件（隐蔽性、迁移性，不含 NC）
  nc:    验证 data_*_nc 文件（含 NC、S_stealth）
  # 默认：验证所有 arch、no_nc + nc，抽样 50 个点
python analysis/validate_extraction.py

# 全量验证（校验所有数据点）
python analysis/validate_extraction.py --full

# 指定随机种子（抽样可复现）
python analysis/validate_extraction.py --seed 42

# 只验证某个 arch
python analysis/validate_extraction.py --arch mobilenet

# 只验证某种模式
python analysis/validate_extraction.py --mode no_nc
python analysis/validate_extraction.py --mode nc

# 组合示例：全量验证 mobilenet 的 nc 数据
python analysis/validate_extraction.py --arch mobilenet --mode nc --full

# 修改抽样数量（默认 50）
python analysis/validate_extraction.py --sample-size 30

"""

import csv
import json
import random
import os
import re
import numpy as np

DEFENSE_METHODS = ["STRIP", "SCaLe-Up", "SentiNet", "IBD_PSC"]
DEFENSE_PREFIX = {"STRIP": "strip", "SCaLe-Up": "scaleup", "SentiNet": "sentinet", "IBD_PSC": "ibd_psc"}
# arch 参数 -> 文件夹名中的 arch 后缀 (如 arch=mobilenetv2_cifar10)
ARCH_TO_FOLDER = {"resnet18": "ResNet18", "mobilenet": "mobilenetv2", "vgg": "vgg19_bn"}


def _parse_defense(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        tpr = float(d.get('tpr', 0) or 0)
        if tpr > 1:
            tpr /= 100.0
        auc = float(d.get('auc', 0) or 0)
        return tpr, auc
    except Exception:
        return 0.0, 0.0


def _param_str(val, suffix):
    return str(int(val)) if suffix == 'alpha' and val == int(val) else str(val)


def parse_result_files(dir_path, test_param_value, test_param_type_suffix, include_nc=True):
    """解析：stealth_tpr_avg, stealth_auc_avg, transfer_rate；include_nc 时含 nc_max_anomaly_index, nc_is_poisoned"""
    res = {'stealth_tpr_avg': None, 'stealth_auc_avg': None, 'transfer_rate': None,
           'nc_max_anomaly_index': None, 'nc_is_poisoned': None}
    param_str = _param_str(test_param_value, test_param_type_suffix.replace('test_', ''))

    # 1. 防御：tpr/auc
    tprs, aucs = {}, {}
    for method, prefix in DEFENSE_PREFIX.items():
        p = os.path.join(dir_path, f'{prefix}_defense_results_test_{test_param_type_suffix}={param_str}.json')
        if not os.path.exists(p) and test_param_value == int(test_param_value):
            p = os.path.join(dir_path, f'{prefix}_defense_results_test_{test_param_type_suffix}={test_param_value}.json')
        if os.path.exists(p):
            tprs[method], aucs[method] = _parse_defense(p)
        else:
            bp = os.path.join(dir_path, f'{prefix}_defense_results.json')
            tprs[method], aucs[method] = _parse_defense(bp) if os.path.exists(bp) else (0.0, 0.0)
    if tprs or aucs:
        # 与 extract 一致：隐蔽性 = 1 - 检测率
        raw_tpr = np.mean([tprs.get(m, 0) for m in DEFENSE_METHODS])
        raw_auc = np.mean([aucs.get(m, 0) for m in DEFENSE_METHODS])
        res['stealth_tpr_avg'] = 1.0 - raw_tpr
        res['stealth_auc_avg'] = 1.0 - raw_auc

    # 2. 迁移性：test_stl10_results_delta=4.0.txt 等
    suffix = test_param_type_suffix.replace('test_', '') if test_param_type_suffix else ''
    stl10 = os.path.join(dir_path, f'test_stl10_results_{suffix}={param_str}.txt')
    if not os.path.exists(stl10) and test_param_value == int(test_param_value):
        stl10 = os.path.join(dir_path, f'test_stl10_results_{suffix}={test_param_value}.txt')
    if os.path.exists(stl10):
        try:
            c = open(stl10, encoding='utf-8').read()
            if m := re.search(r'攻击成功率[:：]\s*([\d.]+)', c):
                res['transfer_rate'] = float(m.group(1))
        except Exception:
            pass
    if res['transfer_rate'] is None:
        stl10_base = os.path.join(dir_path, 'test_stl10_results.txt')
        if os.path.exists(stl10_base):
            try:
                c = open(stl10_base, encoding='utf-8').read()
                if m := re.search(r'攻击成功率[:：]\s*([\d.]+)', c):
                    res['transfer_rate'] = float(m.group(1))
            except Exception:
                pass

    # 3. NC detection（仅 include_nc 时解析）
    if include_nc:
        for f in os.listdir(dir_path):
            if f.startswith('nc_detection') and f.endswith('.json'):
                try:
                    d = json.load(open(os.path.join(dir_path, f), encoding='utf-8'))
                    if d.get('max_anomaly_index') is not None:
                        res['nc_max_anomaly_index'] = float(d['max_anomaly_index'])
                    if 'is_poisoned' in d:
                        res['nc_is_poisoned'] = bool(d['is_poisoned'])
                except Exception:
                    pass
                break
    return res


def parse_float(v):
    if v is None or v == "": return None
    try: return float(v)
    except ValueError: return None


def parse_bool(v):
    if v is None or v == "": return None
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in ('true', '1', 'yes'): return True
    if s in ('false', '0', 'no'): return False
    return None


def extract_test_param_suffix(test_param_type):
    if not test_param_type: return None
    return test_param_type[5:] if test_param_type.startswith('test_') else test_param_type


def find_result_directory(base_dir, group_info, arch=None):
    at = group_info['attack_type']
    pr = group_info['poison_rate']
    tv = group_info['train_param_value']
    candidates = []
    if at == 'adaptive_blend':
        cr = group_info.get('cover_rate', 0.03)
        candidates = [f"adaptive_blend_{pr:.3f}_alpha={tv:.3f}_cover={cr:.3f}_trigger=hellokitty_32.png_poison_seed=2333"]
    elif at == 'adaptive_patch':
        cr = group_info.get('cover_rate', 0.06)
        candidates = [f"adaptive_patch_{pr:.3f}_alpha={tv:.3f}_cover={cr:.3f}_poison_seed=2333"]
    elif at == 'basic':
        candidates = [f"basic_{pr:.3f}_alpha={tv:.3f}_trigger=badnet_patch_32.png_poison_seed=2333"]
    elif at == 'blend':
        candidates = [f"blend_{pr:.3f}_alpha={tv:.3f}_trigger=hellokitty_32.png_poison_seed=2333"]
    elif at == 'badnet':
        candidates = [f"badnet_{pr:.3f}_alpha={tv:.3f}_poison_seed=2333"]
    elif at == 'SIG':
        candidates = [f"SIG_{pr:.3f}_delta={int(tv)}_f=6_poison_seed=2333"]
    elif at == 'WaNet':
        cr = group_info.get('cover_rate', 0.06)
        # 文件夹中 s=1.0 写作 s=1，s=0.4 保持 s=0.4
        s_str = str(int(tv)) if tv == int(tv) else str(tv)
        candidates = [f"WaNet_{pr:.3f}_cover={cr:.3f}_s={s_str}_k=4_poison_seed=2333"]
    elif at == 'belt':
        cr = group_info.get('cover_rate', 0.5)
        mr = tv if tv is not None else 0.1
        candidates = [f"belt_{pr:.3f}_cover={cr:.3f}_mask={mr:.3f}_poison_seed=2333"]
    elif at == 'upgd':
        # 文件夹格式为 eps=10.0（整数也带 .0）
        eps_str = f"{int(tv)}.0" if tv == int(tv) else str(tv)
        candidates = [f"upgd_{pr:.3f}_eps={eps_str}_constraint=Linf_steps=100_mult=5_poison_seed=2333"]
    else:
        return None

    arch_suffix = ARCH_TO_FOLDER.get(arch) if arch else None
    for dn in candidates:
        full = os.path.join(base_dir, dn)
        if os.path.exists(full):
            if arch_suffix is None or f"arch={arch_suffix}" in full:
                return full
        try:
            for sub in sorted(os.listdir(base_dir)):
                if sub.startswith(dn + "_") or sub == dn:
                    if arch_suffix and f"arch={arch_suffix}" not in sub:
                        continue
                    p = os.path.join(base_dir, sub)
                    if os.path.isdir(p):
                        return p
        except OSError:
            pass
    return None


def validate_data_point(base_dir, group_info, point_info, arch=None, include_nc=True):
    cover = point_info.get('cover_rate') or group_info.get('cover_rate')
    gi = {**group_info, 'cover_rate': cover} if cover is not None else group_info
    result_dir = find_result_directory(base_dir, gi, arch=arch)
    if not result_dir:
        return {'status': 'DIR_NOT_FOUND', 'message': '未找到结果目录'}

    test_param_type = point_info.get('test_param_type')
    suffix = extract_test_param_suffix(test_param_type)
    if not suffix:
        return {'status': 'DIR_NOT_FOUND', 'message': '无法确定 test_param_suffix'}

    parsed = parse_result_files(result_dir, point_info['test_param_value'], suffix, include_nc=include_nc)
    tol = 1e-6
    errors = []
    for k in ['stealth_tpr_avg', 'stealth_auc_avg', 'transfer_rate']:
        jv, fv = point_info.get(k), parsed.get(k)
        if jv is not None and fv is not None and abs(jv - fv) > tol:
            errors.append(f"{k}: CSV={jv:.6f}, 文件={fv:.6f}")
        elif jv is not None and fv is None:
            errors.append(f"{k}: 原始文件中未找到")
    if include_nc:
        for k in ['nc_max_anomaly_index']:
            jv, fv = point_info.get(k), parsed.get(k)
            if jv is not None and fv is not None and abs(jv - fv) > tol:
                errors.append(f"{k}: CSV={jv:.6f}, 文件={fv:.6f}")
        jp, fp = point_info.get('nc_is_poisoned'), parsed.get('nc_is_poisoned')
        if jp is not None and fp is not None and jp != fp:
            errors.append(f"nc_is_poisoned: CSV={jp}, 文件={fp}")

    return {'status': 'MISMATCH' if errors else 'OK', 'message': '; '.join(errors), 'dir': result_dir} if errors else {'status': 'OK', 'dir': result_dir}


def load_data_from_csv(csv_path):
    groups = {}
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gid = int(row.get('group_id', 0))
            if gid not in groups:
                groups[gid] = {'group_id': gid, 'attack_type': row.get('attack_type'),
                               'poison_rate': parse_float(row.get('poison_rate')) or 0.03,
                               'train_param_value': parse_float(row.get('train_param_value')),
                               'cover_rate': parse_float(row.get('cover_rate')), 'points': []}
            pt = {
                'point_id': int(row.get('point_id', 0)), 'test_param_type': row.get('test_param_type'),
                'test_param_value': parse_float(row.get('test_param_value')),
                'stealth_tpr_avg': parse_float(row.get('stealth_tpr_avg')),
                'stealth_auc_avg': parse_float(row.get('stealth_auc_avg')),
                'transfer_rate': parse_float(row.get('transfer_rate')),
                'cover_rate': parse_float(row.get('cover_rate')),
                'nc_max_anomaly_index': parse_float(row.get('nc_max_anomaly_index')),
                'nc_is_poisoned': parse_bool(row.get('nc_is_poisoned')),
                'S_stealth': parse_float(row.get('S_stealth'))
            }
            groups[gid]['points'].append(pt)
    return [{'group': g, 'point': p} for g in groups.values() for p in g['points']]


def load_data_from_json(json_path):
    """从 JSON 加载数据，返回与 load_data_from_csv 相同格式的列表"""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    result = []
    for g in data.get('groups', []):
        group = {
            'group_id': g.get('group_id'),
            'attack_type': g.get('attack_type'),
            'poison_rate': parse_float(g.get('poison_rate')) or 0.03,
            'train_param_value': parse_float(g.get('train_param_value')),
            'cover_rate': parse_float(g.get('cover_rate')),
        }
        for dp in g.get('data_points', []):
            point = {
                'point_id': dp.get('point_id', 0),
                'test_param_type': dp.get('test_param_type'),
                'test_param_value': parse_float(dp.get('test_param_value')),
                'stealth_tpr_avg': parse_float(dp.get('stealth_tpr_avg')),
                'stealth_auc_avg': parse_float(dp.get('stealth_auc_avg')),
                'transfer_rate': parse_float(dp.get('transfer_rate')),
                'nc_max_anomaly_index': parse_float(dp.get('nc_max_anomaly_index')),
                'nc_is_poisoned': parse_bool(dp.get('nc_is_poisoned')),
                'cover_rate': parse_float(dp.get('cover_rate')),
                'S_stealth': parse_float(dp.get('S_stealth')),
            }
            result.append({'group': group, 'point': point})
    return result


def _point_key(item):
    """生成 (group_id, point_id) 用于匹配"""
    return (item['group']['group_id'], item['point']['point_id'])


def check_data_completeness(csv_path, include_nc):
    """从 CSV 原始行检查数据完整性，返回 (是否完整, 缺失详情列表)"""
    if not os.path.exists(csv_path):
        return True, []
    required_cols = [
        ('attack_type', '攻击类型'),
        ('poison_rate', '中毒率'),
        ('train_param_value', '触发器强度'),
        ('test_param_value', '测试强度'),
        ('test_param_type', '测试参数类型'),
        ('stealth_tpr_avg', '隐蔽性TPR'),
        ('stealth_auc_avg', '隐蔽性AUC'),
        ('transfer_rate', '迁移率'),
    ]
    if include_nc:
        required_cols.extend([
            ('nc_max_anomaly_index', 'NC异常指数'),
            ('nc_is_poisoned', 'NC是否中毒'),
            ('S_stealth', 'S_stealth'),
        ])
    issues = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    total = len(rows)
    if total == 0:
        return True, []
    for col, name in required_cols:
        if col not in fieldnames:
            issues.append((col, name, total, total, '列不存在'))
            continue
        cnt = sum(1 for r in rows if r.get(col) is None or str(r.get(col, '')).strip() == '')
        if cnt > 0:
            issues.append((col, name, cnt, total, '缺失'))
    return len(issues) == 0, issues


def recompute_s_stealth(all_points):
    """按 extract 公式重算 S_stealth，返回 {point_key: s_stealth}。
    is_poisoned=False 时，max_norm 视为最小值。
    """
    max_norms = [it['point']['nc_max_anomaly_index'] for it in all_points
                 if it['point'].get('nc_is_poisoned') and it['point'].get('nc_max_anomaly_index') is not None]
    max_norm_min = min(max_norms) if max_norms else 0.0
    max_norm_max = max(max_norms) if max_norms else 1.0
    denom = max_norm_max - max_norm_min if max_norm_max > max_norm_min else 1.0

    out = {}
    for it in all_points:
        p = it['point']
        auc_part = 0.8 * (p.get('stealth_auc_avg') or 0.0)
        if p.get('nc_is_poisoned') and p.get('nc_max_anomaly_index') is not None:
            norm_val = p['nc_max_anomaly_index']
        else:
            norm_val = max_norm_min
        nc_part = 0.2 * (norm_val - max_norm_min) / denom
        out[_point_key(it)] = auc_part + nc_part
    return out


def compare_csv_json(csv_pt, json_pt, tol=1e-6, include_nc=True):
    """比较 CSV 与 JSON 中同一数据点是否一致，返回错误列表"""
    errors = []
    base_keys = ['stealth_tpr_avg', 'stealth_auc_avg', 'transfer_rate', 'cover_rate']
    for k in base_keys:
        cv, jv = csv_pt.get(k), json_pt.get(k)
        if cv is not None and jv is not None:
            if abs(cv - jv) > tol:
                errors.append(f"{k}: CSV={cv:.6f}, JSON={jv:.6f}")
        elif (cv is None) != (jv is None):
            errors.append(f"{k}: CSV={cv}, JSON={jv} (一方缺失)")
    if include_nc:
        for k in ['nc_max_anomaly_index', 'S_stealth']:
            cv, jv = csv_pt.get(k), json_pt.get(k)
            if cv is not None and jv is not None:
                if abs(cv - jv) > tol:
                    errors.append(f"{k}: CSV={cv:.6f}, JSON={jv:.6f}")
        cp, jp = csv_pt.get('nc_is_poisoned'), json_pt.get('nc_is_poisoned')
        if cp is not None and jp is not None and cp != jp:
            errors.append(f"nc_is_poisoned: CSV={cp}, JSON={jp}")
    return errors


ARCHS = ['resnet18', 'mobilenet', 'vgg']


def main():
    import argparse
    parser = argparse.ArgumentParser(description='验证提取结果的正确性')
    parser.add_argument('--arch', choices=ARCHS + ['all'], default='all',
                        help='要验证的模型架构，all=验证所有 (默认: all)')
    parser.add_argument('--dataset', default='cifar10', help='数据集 (默认: cifar10)')
    parser.add_argument('--mode', choices=['no_nc', 'nc', 'all'], default='all',
                        help='no_nc: 验证 _no_nc 文件；nc: 验证 _nc 文件；all: 两种都验证（默认）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，用于抽样可复现；不指定则每次随机')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='每个文件抽样验证的数据点数 (默认: 50)，--full 时无效')
    parser.add_argument('--full', action='store_true',
                        help='全量验证：校验所有数据点（CSV↔JSON、CSV↔原始文件），不抽样')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[随机种子] seed={args.seed}，抽样可复现\n")

    archs_to_validate = ARCHS if args.arch == 'all' else [args.arch]
    modes = ['no_nc', 'nc'] if args.mode == 'all' else [args.mode]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(script_dir), 'poisoned_train_set', args.dataset)

    # 汇总统计
    total_ok = total_mismatch = total_nf = total_json_err = total_sampled = total_skipped = 0
    completeness_ok = 0
    completeness_fail = 0

    for mode_name in modes:
        suffix = '_nc' if mode_name == 'nc' else '_no_nc'
        include_nc = mode_name == 'nc'
        for arch in archs_to_validate:
            csv_path = os.path.join(script_dir, f'data_{args.dataset}_{arch}{suffix}.csv')
            json_path = os.path.join(script_dir, f'data_{args.dataset}_{arch}{suffix}.json')

            if not os.path.exists(csv_path):
                print(f"[{mode_name}/{arch}] 跳过: CSV 不存在 {csv_path}")
                continue

            print(f"\n{'='*50}\n[{mode_name}] {arch} 验证 {os.path.basename(csv_path)}\n{'='*50}")
            all_pts = load_data_from_csv(csv_path)
            json_pts = {_point_key(it): it['point'] for it in load_data_from_json(json_path)}
            if len(json_pts) == 0 and os.path.exists(json_path):
                print("警告: JSON 文件存在但解析后无有效数据点")
            elif len(json_pts) == 0:
                print(f"警告: JSON 文件不存在: {json_path}")

            # 数据完整性检查（全量，非抽样）
            complete, completeness_issues = check_data_completeness(csv_path, include_nc)
            if complete:
                completeness_ok += 1
                print(f"  数据完整性: ✓ 全部完整 ({len(all_pts)} 条)")
            else:
                completeness_fail += 1
                print(f"  数据完整性: ✗ 存在缺失")
                for col, name, cnt, tot, reason in completeness_issues:
                    pct = cnt / tot * 100 if tot else 0
                    print(f"    - {name}({col}): {cnt}/{tot} 条缺失 ({pct:.1f}%)")

            # nc 模式：验证 S_stealth 计算正确性
            s_stealth_err = 0
            if include_nc and all_pts:
                recomputed = recompute_s_stealth(all_pts)
                tol = 1e-6
                for it in all_pts:
                    k = _point_key(it)
                    exp = recomputed.get(k)
                    got = it['point'].get('S_stealth')
                    if exp is not None and got is not None and abs(exp - got) > tol:
                        s_stealth_err += 1
                if s_stealth_err:
                    print(f"  S_stealth 公式校验: {s_stealth_err} 个点与重算值不一致")
                else:
                    print(f"  S_stealth 公式校验: 通过")

            # CSV vs JSON 全量一致性校验（所有数据点）
            csv_json_mismatch = []
            for it in all_pts:
                jp = json_pts.get(_point_key(it))
                if jp is None:
                    csv_json_mismatch.append((it, 'JSON中无此数据点'))
                else:
                    errs = compare_csv_json(it['point'], jp, include_nc=include_nc)
                    if errs:
                        csv_json_mismatch.append((it, '; '.join(errs)))
            if csv_json_mismatch:
                print(f"  CSV↔JSON 全量校验: ✗ {len(csv_json_mismatch)}/{len(all_pts)} 条不一致")
                for it, msg in csv_json_mismatch[:5]:  # 最多展示 5 条
                    g, p = it['group'], it['point']
                    print(f"    Group {g['group_id']} Point {p['point_id']}: {msg}")
                if len(csv_json_mismatch) > 5:
                    print(f"    ... 还有 {len(csv_json_mismatch) - 5} 条")
            else:
                print(f"  CSV↔JSON 全量校验: ✓ 全部一致 ({len(all_pts)} 条)")

            # 抽样或全量：与原始文件比对
            if args.full:
                to_validate = all_pts
                print(f"总数据点: CSV={len(all_pts)}, JSON={len(json_pts)}, 全量验证: {len(to_validate)} 个\n")
            else:
                to_validate = random.sample(all_pts, min(args.sample_size, len(all_pts)))
                print(f"总数据点: CSV={len(all_pts)}, JSON={len(json_pts)}, 随机抽样: {len(to_validate)} 个\n")
            ok, mismatch, nf, skipped = 0, 0, 0, 0
            json_err = len(csv_json_mismatch)  # 来自全量校验
            for i, item in enumerate(to_validate, 1):
                g, p = item['group'], item['point']
                print(f"[{i}/{len(to_validate)}] Group {g['group_id']} Point {p['point_id']} | {g['attack_type']} | train_param={g['train_param_value']}")
                if g['train_param_value'] is None:
                    skipped += 1
                    print("  跳过 (train_param_value 缺失)\n")
                    continue
                # 验证 CSV/JSON vs 原始文件
                r = validate_data_point(base_dir, g, p, arch=arch, include_nc=include_nc)
                if r['status'] == 'OK': ok += 1
                elif r['status'] == 'MISMATCH': mismatch += 1
                else: nf += 1
                print(f"  状态: {r['status']} | {r.get('message', '')}\n")
            print(f"[{mode_name}/{arch}] ✓ OK: {ok} | ✗ MISMATCH: {mismatch} | ⚠ DIR_NOT_FOUND: {nf} | CSV≠JSON: {json_err} | 跳过: {skipped}")
            total_ok += ok
            total_mismatch += mismatch
            total_nf += nf
            total_json_err += json_err
            total_skipped += skipped
            total_sampled += len(to_validate)

    # 总统计
    total_validated = total_ok + total_mismatch + total_nf
    total_errors = total_mismatch + total_nf + total_json_err
    total_files = completeness_ok + completeness_fail
    print(f"\n{'='*60}")
    print("【总统计】")
    print(f"  数据完整性:   {completeness_ok}/{total_files} 个文件全部完整" + (f", {completeness_fail} 个存在缺失" if completeness_fail > 0 else ""))
    print(f"  CSV↔JSON:     已全量校验所有文件")
    print(f"  原始文件校验: {total_sampled} 个数据点" + (" (全量)" if args.full else " (抽样)"))
    print(f"  有效验证数:   {total_validated} 个")
    if total_skipped > 0:
        print(f"  跳过(缺失):   {total_skipped} 个 (train_param_value 缺失)")
    print(f"  ✓ 正确(OK):  {total_ok}")
    print(f"  ✗ 错误:     {total_errors} (MISMATCH={total_mismatch}, DIR_NOT_FOUND={total_nf}, CSV≠JSON={total_json_err})")
    if total_validated > 0:
        correct_rate = total_ok / total_validated * 100
        error_rate = total_errors / total_validated * 100
        print(f"  正确率: {correct_rate:.2f}%  |  错误率: {error_rate:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

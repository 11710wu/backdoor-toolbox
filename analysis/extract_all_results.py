#!/usr/bin/env python3
"""从 cifar10 文件夹下提取所有攻击方法的结果：
   - 只提取包含 densenet 的文件夹
   - 只提取中毒率为 0.03 的数据
   - 只提取原始攻击成功率（test_model_asr）大于50%的数据点
   
   输出：只生成 CSV 文件（data_cifar10.csv），不再生成 JSON 文件
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

DEFENSE_METHODS = ["AC", "STRIP", "SCaLe-Up", "SentiNet", "IBD_PSC"]
# 不再跳过任何攻击类型，包括 SIG 和 WaNet
# SKIP_ATTACK_TYPES = {"SIG", "WaNet"}


def format_param_value(value: float) -> str:
    """格式化数字，去除多余的零"""
    if value is None:
        return ""
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return f"{value:.6f}".rstrip('0').rstrip('.')


def get_training_param_type(attack_type: str) -> Optional[str]:
    """根据攻击类型确定训练参数类型"""
    attack_lower = (attack_type or '').lower()
    if 'wanet' in attack_lower:
        return 's'
    if 'sig' in attack_lower:
        return 'delta'
    if any(term in attack_lower for term in ['blend', 'badnet', 'basic', 'patch']):
        return 'alpha'
    return None


def get_test_param_type_from_train_type(train_param_type: str) -> Optional[str]:
    """根据训练参数类型获取对应的测试参数类型"""
    mapping = {
        'alpha': 'test_alpha',
        's': 'test_s',
        'delta': 'test_delta'
    }
    return mapping.get(train_param_type)


def is_train_test_param_matched(train_param_type: Optional[str], 
                                 train_param_value: Optional[float],
                                 test_param_type: Optional[str],
                                 test_param_value: Optional[float]) -> bool:
    """检查训练参数和测试参数是否匹配（类型和值都一致）"""
    # 如果训练参数类型或值为 None，无法匹配
    if train_param_type is None or train_param_value is None:
        return False
    
    # 如果测试参数类型或值为 None，无法匹配
    if test_param_type is None or test_param_value is None:
        return False
    
    # 检查测试参数类型是否与训练参数类型对应
    expected_test_type = get_test_param_type_from_train_type(train_param_type)
    if expected_test_type != test_param_type:
        return False
    
    # 检查参数值是否相等（考虑浮点数精度）
    return abs(train_param_value - test_param_value) < 1e-6


def parse_test_stl10_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """解析 test_stl10 结果文件"""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception:
        return None
    
    # 提取 ASR
    asr = 0.0
    match = re.search(r'攻击成功率:\s*([\d.]+)', content)
    if match:
        asr = float(match.group(1))
    
    # 提取测试参数
    param_type = None
    param_value = None
    for pattern, ptype in [(r'test_alpha=([\d.]+)', 'test_alpha'), 
                           (r'test_s=([\d.]+)', 'test_s'), 
                           (r'test_delta=([\d.]+)', 'test_delta')]:
        match = re.search(pattern, file_path.name, re.IGNORECASE)
        if match:
            param_type = ptype
            param_value = float(match.group(1).rstrip('.'))
            break
    
    # 如果没有找到测试参数，尝试从文件内容中提取，或者使用默认值
    # 对于没有 test_alpha/test_s/test_delta 的文件，可能是使用训练时的参数值
    # 这种情况下，我们需要从对应的 test_results 文件中推断参数类型和值
    if not param_type:
        # 尝试从文件名推断攻击类型，然后确定参数类型
        # 但这里我们无法确定参数值，所以返回 None，让调用者处理
        return None
    
    return {"asr": asr, "param_type": param_type, "param_value": param_value}


def parse_test_model_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """解析 test_results 文件"""
    try:
        data = json.loads(file_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    
    asr = float(data.get('asr', 0.0) or 0.0)
    
    # 提取测试参数
    param_type = None
    param_value = None
    for pattern, ptype in [(r'test_alpha=([\d.]+)', 'test_alpha'), 
                           (r'test_s=([\d.]+)', 'test_s'), 
                           (r'test_delta=([\d.]+)', 'test_delta')]:
        match = re.search(pattern, file_path.name, re.IGNORECASE)
        if match:
            param_type = ptype
            param_value = float(match.group(1).rstrip('.'))
            break
    
    if not param_type:
        return None
    
    return {"asr": asr, "param_type": param_type, "param_value": param_value}


def parse_defense_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """解析防御结果文件"""
    try:
        data = json.loads(file_path.read_text(encoding='utf-8'))
        tpr = float(data.get('tpr', 0.0) or 0.0)
        if tpr > 1.0:
            tpr = tpr / 100.0
    except Exception:
        return None
    
    # 提取测试参数
    param_type = None
    param_value = None
    for pattern, ptype in [(r'test_alpha=([\d.]+)', 'test_alpha'), 
                           (r'test_s=([\d.]+)', 'test_s'), 
                           (r'test_delta=([\d.]+)', 'test_delta')]:
        match = re.search(pattern, file_path.name, re.IGNORECASE)
        if match:
            param_type = ptype
            param_value = float(match.group(1).rstrip('.'))
            break
    
    if not param_type:
        return None
    
    return {"tpr": tpr, "param_type": param_type, "param_value": param_value}


def parse_folder_name(folder_name: str) -> Dict[str, Any]:
    """解析文件夹名称提取攻击类型和参数"""
    params = {}
    
    # 提取攻击类型 - 按优先级顺序匹配
    attack_patterns = [
        # adaptive 系列（需要最先匹配，避免被 blend/patch 匹配）
        (r'^adaptive_blend', 'adaptive_blend'),
        (r'^adaptive-blend', 'adaptive_blend'),
        (r'^adaptive_patch', 'adaptive_patch'),
        (r'^adaptive-patch', 'adaptive_patch'),
        (r'^adaptive_badnet', 'adaptive_badnet'),
        (r'^adaptive-badnet', 'adaptive_badnet'),
        # WaNet 和 SIG（大小写敏感）
        (r'^WaNet', 'WaNet'),
        (r'^wanet', 'WaNet'),
        (r'^SIG', 'SIG'),
        (r'^sig', 'SIG'),
        # basic, badnet, blend（basic 需要在 badnet 之前）
        (r'^basic', 'basic'),
        (r'^badnet', 'badnet'),
        (r'^blend', 'blend'),
    ]
    
    attack_type = None
    for pattern, atype in attack_patterns:
        if re.match(pattern, folder_name):
            attack_type = atype
            break
    
    # 如果没匹配到，使用备用方法
    if not attack_type:
        parts = folder_name.split('_')
        if len(parts) > 0:
            # 查找第一个数字位置，之前的部分是攻击类型
            for i, part in enumerate(parts):
                if re.match(r'^\d+\.?\d*$', part):
                    attack_type_str = '_'.join(parts[:i])
                    # 标准化攻击类型名称
                    if 'adaptive' in attack_type_str.lower() and 'blend' in attack_type_str.lower():
                        attack_type = 'adaptive_blend'
                    elif 'adaptive' in attack_type_str.lower() and 'patch' in attack_type_str.lower():
                        attack_type = 'adaptive_patch'
                    elif 'adaptive' in attack_type_str.lower() and 'badnet' in attack_type_str.lower():
                        attack_type = 'adaptive_badnet'
                    elif 'blend' in attack_type_str.lower():
                        attack_type = 'blend'
                    elif 'badnet' in attack_type_str.lower():
                        attack_type = 'badnet'
                    elif 'basic' in attack_type_str.lower():
                        attack_type = 'basic'
                    elif 'wanet' in attack_type_str.lower():
                        attack_type = 'WaNet'
                    elif 'sig' in attack_type_str.lower():
                        attack_type = 'SIG'
                    else:
                        attack_type = attack_type_str
                    break
    
    params['attack_type'] = attack_type if attack_type else 'unknown'
    
    # 提取参数
    match = re.search(r'^.+?_(\d+\.?\d*)_', folder_name)
    if match:
        params['poison_rate'] = float(match.group(1))
    
    match = re.search(r'alpha=([\d.]+)', folder_name)
    if match:
        params['alpha'] = float(match.group(1))
    
    match = re.search(r'cover=([\d.]+)', folder_name)
    if match:
        params['cover_rate'] = float(match.group(1))
    
    match = re.search(r'delta=([\d.]+)', folder_name)
    if match:
        params['delta'] = float(match.group(1))
    
    match = re.search(r's=([\d.]+)', folder_name)
    if match:
        params['s'] = float(match.group(1))
    
    return params


def extract_folder_results(folder: Path, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从文件夹提取所有数据点"""
    attack_type = params.get('attack_type')
    train_param_type = get_training_param_type(attack_type)
    train_param_value = params.get(train_param_type) if train_param_type else None
    
    # 收集所有测试参数值
    test_params: Dict[float, Dict[str, Any]] = {}
    
    # 提取 test_model 结果
    for test_file in folder.glob("test_results_seed=*.json"):
        record = parse_test_model_file(test_file)
        if record:
            pval = record['param_value']
            if pval not in test_params:
                test_params[pval] = {}
            if record.get('param_type'):
                test_params[pval]['param_type'] = record['param_type']
            test_params[pval]['test_model_asr'] = record['asr']
    
    # 提取 test_stl10 结果
    # 先提取所有有明确测试参数的文件
    for stl10_file in folder.glob("test_stl10_results*.txt"):
        record = parse_test_stl10_file(stl10_file)
        if record and record.get('param_type') and record.get('param_value') is not None:
            pval = record['param_value']
            if pval not in test_params:
                test_params[pval] = {}
            if record.get('param_type'):
                test_params[pval]['param_type'] = record['param_type']
            test_params[pval]['test_stl10_asr'] = record['asr']
    
    # 对于没有明确测试参数的文件（如 test_stl10_results.txt），
    # 尝试从对应的 test_results 文件中推断参数值
    # 查找没有 test_alpha/test_s/test_delta 的 test_stl10 文件
    default_stl10_file = folder / "test_stl10_results.txt"
    if default_stl10_file.exists():
        # 尝试从 test_results 文件中找到对应的参数值
        # 查找所有 test_results 文件，看哪个没有 test_alpha 等参数
        for test_file in folder.glob("test_results_seed=*.json"):
            # 检查文件名是否包含 test_alpha/test_s/test_delta
            has_test_param = any(re.search(pattern, test_file.name, re.IGNORECASE) 
                                for pattern in [r'test_alpha=', r'test_s=', r'test_delta='])
            if not has_test_param:
                # 这个 test_results 文件没有测试参数，可能对应默认的 test_stl10_results.txt
                # 但我们需要知道参数类型和值
                # 对于这种情况，我们使用训练参数值作为测试参数值
                if train_param_type and train_param_value is not None:
                    pval = train_param_value
                    if pval not in test_params:
                        test_params[pval] = {}
                    test_param_type = get_test_param_type_from_train_type(train_param_type)
                    if test_param_type:
                        test_params[pval]['param_type'] = test_param_type
                    # 解析 test_stl10_results.txt
                    try:
                        content = default_stl10_file.read_text(encoding='utf-8')
                        match = re.search(r'攻击成功率:\s*([\d.]+)', content)
                        if match:
                            test_params[pval]['test_stl10_asr'] = float(match.group(1))
                    except Exception:
                        pass
                break
    
    # 提取防御结果
    defense_files = {
        "AC": "ac_defense_results",
        "STRIP": "strip_defense_results",
        "SCaLe-Up": "scaleup_defense_results",
        "SentiNet": "sentinet_defense_results",
        "IBD_PSC": "ibd_psc_defense_results"
    }
    
    for method, prefix in defense_files.items():
        for defense_file in folder.glob(f"{prefix}_test_*.json"):
            record = parse_defense_file(defense_file)
            if record:
                pval = record['param_value']
                if pval not in test_params:
                    test_params[pval] = {}
                if record.get('param_type'):
                    test_params[pval]['param_type'] = record['param_type']
                if 'defense_tprs' not in test_params[pval]:
                    test_params[pval]['defense_tprs'] = {}
                test_params[pval]['defense_tprs'][method] = record['tpr']
    
    # 构建数据点
    results = []
    for test_param_value in sorted(test_params.keys()):
        data = test_params[test_param_value]
        
        # 必须要有 test_model_asr 和测试参数类型
        if 'test_model_asr' not in data:
            continue
        
        # 没有提取到测试参数类型时跳过
        test_param_type = data.get('param_type')
        if not test_param_type:
            continue
        
        # 检查原始攻击成功率是否大于50%
        test_model_asr = float(data['test_model_asr'])
        if test_model_asr <= 0.5:
            continue
        
        # test_stl10_asr 和 defense_tprs 可以为空，使用默认值
        test_stl10_asr = float(data.get('test_stl10_asr', 0.0))
        
        # 防御方法数据：允许部分缺失，缺失的使用默认值 0.0
        if 'defense_tprs' not in data:
            data['defense_tprs'] = {}
        defense_tprs = {method: data['defense_tprs'].get(method, 0.0) for method in DEFENSE_METHODS}
        
        # 计算指标
        attack_rate = test_model_asr
        stealth_rate = 0.2 * sum(1.0 - tpr for tpr in defense_tprs.values())
        # 迁移率 = test_stl10_asr（直接使用 STL-10 攻击成功率）
        transfer_rate = test_stl10_asr
        
        results.append({
            **params,
            'train_param_value': train_param_value,
            'test_param_value': test_param_value,
            'test_param_type': test_param_type,
            'attack_rate': attack_rate,
            'stealth_rate': stealth_rate,
            'transfer_rate': transfer_rate
        })
    
    return results


def convert_to_data_groups(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将结果转换为分组格式"""
    # 按 (trigger_type, poison_rate, train_param_value) 分组
    groups_dict: Dict[tuple, List[Dict[str, Any]]] = {}
    
    for result in all_results:
        attack_type = result['attack_type']
        poison_rate = result.get('poison_rate', 0.03)
        train_param_value = result.get('train_param_value')
        
        # 确定触发器类型（保持原有的大小写）
        if 'adaptive' in attack_type.lower():
            trigger_type = attack_type  # adaptive_blend, adaptive_patch
        elif attack_type.lower() == 'basic' or 'badnet' in attack_type.lower():
            trigger_type = 'badnet'  # basic 和 badnet 使用相同的触发器
        elif 'blend' in attack_type.lower() and 'adaptive' not in attack_type.lower():
            trigger_type = 'blend'
        elif 'wanet' in attack_type.lower():
            trigger_type = 'WaNet'  # 保持大写
        elif 'sig' in attack_type.lower():
            trigger_type = 'SIG'  # 保持大写
        else:
            trigger_type = attack_type
        
        key = (trigger_type, poison_rate, train_param_value)
        if key not in groups_dict:
            groups_dict[key] = []
        groups_dict[key].append(result)
    
    # 转换为列表格式
    groups = []
    def sort_key(item):
        trigger_type, poison_rate, train_param_value = item[0]
        normalized_train = float('-inf') if train_param_value is None else train_param_value
        # 获取该组中第一个条目的 cover_rate 用于排序（如果存在）
        entries = item[1]
        cover_rate = entries[0].get('cover_rate') if entries else None
        normalized_cover = float('-inf') if cover_rate is None else cover_rate
        return (trigger_type, poison_rate, normalized_train, normalized_cover)

    for group_id, (key, entries) in enumerate(sorted(groups_dict.items(), key=sort_key), start=1):
        trigger_type, poison_rate, train_param_value = key
        
        # 排序数据点：先按 cover_rate，再按 test_param_value
        entries.sort(key=lambda x: (
            float('-inf') if x.get('cover_rate') is None else x.get('cover_rate'),
            x.get('test_param_value', 0)
        ))
        
        # 构建数据点
        data_points = []
        for point_id, entry in enumerate(entries, start=1):
            data_points.append({
                "group_id": group_id,
                "point_id": point_id,
                "attack_rate": entry['attack_rate'],
                "stealth_rate": entry['stealth_rate'],
                "transfer_rate": entry['transfer_rate'],
                "train_param_value": train_param_value,
                "test_param_value": entry['test_param_value'],
                "test_param_type": entry.get('test_param_type'),
                "cover_rate": entry.get('cover_rate')  # 添加 cover_rate
            })
        
        # 计算统计信息
        attack_rates = [p['attack_rate'] for p in data_points]
        stealth_rates = [p['stealth_rate'] for p in data_points]
        transfer_rates = [p['transfer_rate'] for p in data_points]
        
        group_info = {
            "group_id": group_id,
            "data_points": data_points,
            "group_size": len(data_points),
            "attack_type": entries[0]['attack_type'],
            "poison_rate": poison_rate,
            "trigger_type": trigger_type,
            "train_param_value": train_param_value,
            "avg_attack_rate": float(np.mean(attack_rates)),
            "avg_stealth_rate": float(np.mean(stealth_rates)),
            "avg_transfer_rate": float(np.mean(transfer_rates)),
            "std_attack_rate": float(np.std(attack_rates)) if len(attack_rates) > 1 else 0.0,
            "std_stealth_rate": float(np.std(stealth_rates)) if len(stealth_rates) > 1 else 0.0,
            "std_transfer_rate": float(np.std(transfer_rates)) if len(transfer_rates) > 1 else 0.0
        }
        
        # 添加训练参数类型
        train_param_type = get_training_param_type(entries[0]['attack_type'])
        if train_param_type:
            group_info['train_param_type'] = train_param_type
        
        # 添加覆盖率（如果有）
        if 'cover_rate' in entries[0]:
            group_info['cover_rate'] = entries[0]['cover_rate']
        
        groups.append(group_info)
    
    return groups


def export_groups_to_csv(groups: List[Dict[str, Any]], csv_path: Path) -> None:
    """将分组结果导出为 CSV"""
    fieldnames = [
        "group_id",
        "point_id",
        "attack_type",
        "trigger_type",
        "poison_rate",
        "train_param_type",
        "train_param_value",
        "test_param_type",
        "test_param_value",
        "attack_rate",
        "stealth_rate",
        "transfer_rate",
        "cover_rate"
    ]

    def format_value(val: Any) -> Any:
        return "" if val is None else val

    with csv_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            common = {
                "group_id": group["group_id"],
                "attack_type": group["attack_type"],
                "trigger_type": group.get("trigger_type"),
                "poison_rate": group.get("poison_rate"),
                "train_param_type": group.get("train_param_type"),
                "train_param_value": group.get("train_param_value")
                # cover_rate 从数据点中获取，不在这里设置
            }
            for point in group["data_points"]:
                writer.writerow({
                    **common,
                    "point_id": point["point_id"],
                    "test_param_type": point.get("test_param_type"),
                    "test_param_value": point.get("test_param_value"),
                    "attack_rate": point.get("attack_rate"),
                    "stealth_rate": point.get("stealth_rate"),
                    "transfer_rate": point.get("transfer_rate"),
                    "cover_rate": point.get("cover_rate")  # 从数据点获取 cover_rate，而不是从 group
                })


def main():
    """主函数"""
    # 获取脚本所在目录（analysis文件夹）
    script_dir = Path(__file__).parent.absolute()
    # 项目根目录（analysis的父目录）
    base_path = script_dir.parent
    cifar10_path = base_path / "poisoned_train_set" / "cifar10"
    # 输出文件保存到analysis文件夹下
    csv_file = script_dir / "data_cifar10.csv"
    
    if not cifar10_path.exists():
        print(f"错误: 路径不存在: {cifar10_path}")
        return
    
    print(f"正在扫描文件夹: {cifar10_path}")
    print("=" * 80)
    
    # 获取所有文件夹
    folders = [f for f in cifar10_path.iterdir() if f.is_dir()]
    print(f"找到 {len(folders)} 个文件夹\n")
    
    # 目标中毒率
    TARGET_POISON_RATE = 0.03  # 默认中毒率
    TARGET_POISON_RATE_ADAPTIVE = 0.003  # adaptive_patch 和 adaptive_blend 的中毒率
    
    all_results = []
    skipped_count = 0
    skipped_poison_rate_count = 0
    skipped_no_densenet_count = 0
    for folder in folders:
        print(f"处理: {folder.name}")
        
        # 只处理包含 densenet 的文件夹
        if 'densenet' not in folder.name.lower():
            print(f"  跳过（不包含 densenet）")
            skipped_no_densenet_count += 1
            skipped_count += 1
            continue
        
        params = parse_folder_name(folder.name)
        attack_type = params.get('attack_type', '').lower()
        
        # 对于 adaptive_patch 和 adaptive_blend，只提取中毒率为 0.003 的数据
        # 对于其他攻击类型，提取中毒率为 0.03 的数据
        if 'adaptive_patch' in attack_type or 'adaptive_blend' in attack_type:
            target_rate = TARGET_POISON_RATE_ADAPTIVE
        else:
            target_rate = TARGET_POISON_RATE
        
        poison_rate = params.get('poison_rate')
        if poison_rate is None or abs(poison_rate - target_rate) > 1e-6:
            print(f"  跳过（中毒率: {poison_rate}，只提取中毒率为 {target_rate} 的数据）")
            skipped_poison_rate_count += 1
            skipped_count += 1
            continue
        
        attack_type = params.get('attack_type')
        train_param_type = get_training_param_type(attack_type)
        train_param_value = params.get(train_param_type) if train_param_type else None
        
        # 提取结果（会自动过滤攻击成功率<=50%的数据点）
        results = extract_folder_results(folder, params)
        
        if len(results) == 0:
            if train_param_type and train_param_value is not None:
                print(f"  跳过（未找到训练强度={train_param_type}={train_param_value} 且攻击成功率>50%的数据点）")
            else:
                print(f"  跳过（无法确定训练参数类型或值）")
            skipped_count += 1
        else:
            all_results.extend(results)
            print(f"  提取 {len(results)} 个数据点（训练强度={train_param_type}={train_param_value}，攻击成功率>50%）")
    
    print(f"\n总共提取 {len(all_results)} 个数据点")
    print(f"跳过 {skipped_count} 个文件夹（其中 {skipped_no_densenet_count} 个不包含 densenet，{skipped_poison_rate_count} 个因中毒率不符合要求（adaptive_patch/blend: {TARGET_POISON_RATE_ADAPTIVE}，其他: {TARGET_POISON_RATE}），其余因无匹配数据或攻击成功率<=50%）")
    
    # 转换为分组格式
    groups = convert_to_data_groups(all_results)
    
    # 保存到 CSV 文件
    export_groups_to_csv(groups, csv_file)
    
    print("\n" + "=" * 80)
    print(f"✓ CSV 已保存到: {csv_file}")
    print(f"\n结果摘要:")
    print(f"  总组数: {len(groups)}")
    print(f"  总数据点: {len(all_results)}")
    
    # 统计每种攻击类型
    attack_counts = {}
    for group in groups:
        attack_type = group['attack_type']
        attack_counts[attack_type] = attack_counts.get(attack_type, 0) + group['group_size']
    
    print(f"\n攻击类型统计:")
    for attack_type, count in sorted(attack_counts.items()):
        print(f"  {attack_type}: {count} 个数据点")


if __name__ == "__main__":
    main()

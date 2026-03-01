#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证数据提取的正确性
从 data_cifar10.csv 中随机抽样数据点，检查提取的数据是否与原始文件一致
"""

import csv
import json
import random
import os
import re

CSV_OK = "CSV_OK"
CSV_MISMATCH = "CSV_MISMATCH"
CSV_NOT_FOUND = "CSV_NOT_FOUND"

def parse_defense_file(filepath):
    """解析防御结果JSON文件，提取TPR（转换为比例）"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tpr = data.get('tpr', 0.0) or 0.0
        # 如果TPR > 1.0，说明是百分比，需要转换为比例
        if tpr > 1.0:
            tpr = tpr / 100.0
        return float(tpr)
    except Exception:
        return 0.0

def format_param_value_for_test_results(value, test_param_type_suffix):
    """
    格式化参数值用于test_results JSON文件名：
    - alpha: 整数去掉小数点 (如 test_alpha=1.json)
    - delta/s: 保留小数点 (如 test_delta=36.0.json)
    """
    if test_param_type_suffix == 'alpha' and value == int(value):
        return str(int(value))
    return str(value)

def format_param_value_for_defense(value):
    """格式化参数值用于防御结果JSON文件名：整数去掉小数点"""
    if value == int(value):
        return str(int(value))
    return str(value)

def parse_result_files(dir_path, test_param_value, test_param_type_suffix):
    """
    解析完整的结果文件集合，计算三个指标：
    - attack_rate: 从 test_results_seed=2333_test_alpha=X.json 中的 asr
    - stealth_rate: 0.2 * sum(1-TPR) 对五个防御方法求和
    - transfer_rate: test_stl10_asr（直接使用 STL-10 攻击成功率）
    """
    results = {
        'attack_rate': None,
        'stealth_rate': None,
        'transfer_rate': None
    }
    
    # 格式化参数值：不同文件类型用不同格式
    param_str_test_results = format_param_value_for_test_results(test_param_value, test_param_type_suffix)
    param_str_defense = format_param_value_for_defense(test_param_value)
    param_str_txt = str(test_param_value)
    
    # 1. 提取 attack_rate (test_model_asr)
    # 尝试两种格式：格式化后的值和原始值（用于处理整数如0.0, 1.0的情况）
    test_json_file = os.path.join(dir_path, f'test_results_seed=2333_test_{test_param_type_suffix}={param_str_test_results}.json')
    if not os.path.exists(test_json_file) and test_param_value == int(test_param_value):
        # 如果格式化后找不到，且值为整数（如0.0, 1.0），尝试带小数点的格式
        test_json_file = os.path.join(dir_path, f'test_results_seed=2333_test_{test_param_type_suffix}={test_param_value}.json')
    if os.path.exists(test_json_file):
        try:
            with open(test_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results['attack_rate'] = float(data.get('asr', 0.0) or 0.0)
        except Exception:
            pass
    
    # 2. 计算 transfer_rate = test_stl10_asr / test_model_asr
    # 先提取 test_stl10_asr
    test_stl10_asr = None
    stl10_txt_file = os.path.join(dir_path, f'test_stl10_results_test_{test_param_type_suffix}={param_str_txt}.txt')
    if not os.path.exists(stl10_txt_file) and test_param_value == int(test_param_value):
        # 如果格式化后找不到，且值为整数（如0.0, 1.0），尝试带小数点的格式
        stl10_txt_file = os.path.join(dir_path, f'test_stl10_results_test_{test_param_type_suffix}={test_param_value}.txt')
    if os.path.exists(stl10_txt_file):
        try:
            with open(stl10_txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # 提取攻击成功率
            match = re.search(r'攻击成功率[:：]\s*([\d.]+)', content)
            if match:
                test_stl10_asr = float(match.group(1))
        except Exception:
            pass
    
    # 计算迁移率 = test_stl10_asr（直接使用 STL-10 攻击成功率）
    if test_stl10_asr is not None:
        results['transfer_rate'] = test_stl10_asr
    
    # 3. 计算 stealth_rate = 0.2 * sum(1-TPR) 对五个防御方法求和
    # 所有攻击类型都使用相同的5个防御方法
    defense_mapping = {
        "AC": "ac",
        "STRIP": "strip",
        "SCaLe-Up": "scaleup",
        "SentiNet": "sentinet",
        "IBD_PSC": "ibd_psc"
    }
    weight = 0.2
    
    defense_tprs = {}
    for method_name, method_prefix in defense_mapping.items():
        defense_file = os.path.join(dir_path, f'{method_prefix}_defense_results_test_{test_param_type_suffix}={param_str_defense}.json')
        # 如果格式化后找不到，且值为整数（如0.0, 1.0），尝试带小数点的格式
        if not os.path.exists(defense_file) and test_param_value == int(test_param_value):
            defense_file = os.path.join(dir_path, f'{method_prefix}_defense_results_test_{test_param_type_suffix}={test_param_value}.json')
        if os.path.exists(defense_file):
            tpr = parse_defense_file(defense_file)
            defense_tprs[method_name] = tpr
        else:
            # 如果文件不存在，TPR视为0
            defense_tprs[method_name] = 0.0
    
    # stealth_rate = weight * sum(1-TPR)
    stealth_sum = sum(1.0 - tpr for tpr in defense_tprs.values())
    results['stealth_rate'] = weight * stealth_sum
    
    return results


def load_csv_index(csv_path):
    """读取 CSV，建立 (group_id, point_id) 到行数据的索引"""
    index = {}
    if not os.path.exists(csv_path):
        print(f"⚠ 警告: 未找到 CSV 文件 {csv_path}，将跳过 CSV 校验")
        return index
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gid = int(row.get('group_id', ''))
                pid = int(row.get('point_id', ''))
            except (ValueError, TypeError):
                continue
            index[(gid, pid)] = row
    return index


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def validate_against_csv(csv_index, group, point):
    """验证 JSON 数据与 CSV 行是否一致"""
    key = (group['group_id'], point['point_id'])
    row = csv_index.get(key)
    if not row:
        return {
            'status': CSV_NOT_FOUND,
            'message': "CSV 中未找到对应 group_id/point_id"
        }
    
    errors = []
    mappings = [
        ('attack_rate', point['attack_rate']),
        ('stealth_rate', point['stealth_rate']),
        ('transfer_rate', point['transfer_rate']),
        ('test_param_value', point['test_param_value'])
    ]
    for column, expected in mappings:
        csv_val = parse_float(row.get(column))
        if csv_val is None:
            errors.append(f"{column}: CSV 中缺失或格式错误")
        elif abs(csv_val - expected) > 1e-6:
            errors.append(f"{column}: CSV={csv_val:.6f}, JSON={expected:.6f}")
    
    csv_attack_type = row.get('attack_type')
    if csv_attack_type != group['attack_type']:
        errors.append(f"attack_type: CSV={csv_attack_type}, JSON={group['attack_type']}")
    
    csv_test_type = row.get('test_param_type')
    json_test_type = point.get('test_param_type') or group.get('train_param_type')
    if csv_test_type != json_test_type:
        errors.append(f"test_param_type: CSV={csv_test_type}, JSON={json_test_type}")
    
    if errors:
        return {
            'status': CSV_MISMATCH,
            'message': '; '.join(errors)
        }
    return {
        'status': CSV_OK
    }

def extract_test_param_suffix(test_param_type):
    """从test_param_type中提取后缀（如'test_alpha' -> 'alpha'）"""
    if not test_param_type:
        return None
    # 移除'test_'前缀（如果存在）
    if test_param_type.startswith('test_'):
        return test_param_type[5:]  # 移除'test_'（5个字符）
    return test_param_type

def find_result_directory(base_dir, group_info):
    """根据组信息查找对应的结果目录"""
    attack_type = group_info['attack_type']
    poison_rate = group_info['poison_rate']
    train_param_value = group_info['train_param_value']
    
    # 构建目录名（只根据训练参数）
    if attack_type == 'adaptive_blend':
        cover_rate = group_info.get('cover_rate', 0.03)
        dir_name = f"adaptive_blend_{poison_rate:.3f}_alpha={train_param_value:.3f}_cover={cover_rate:.3f}_trigger=hellokitty_32.png_poison_seed=0"
    elif attack_type == 'adaptive_patch':
        cover_rate = group_info.get('cover_rate', 0.06)
        dir_name = f"adaptive_patch_{poison_rate:.3f}_alpha={train_param_value:.3f}_cover={cover_rate:.3f}_poison_seed=0"
    elif attack_type == 'basic':
        dir_name = f"basic_{poison_rate:.3f}_alpha={train_param_value:.3f}_trigger=badnet_patch_32.png_poison_seed=0"
    elif attack_type == 'blend':
        dir_name = f"blend_{poison_rate:.3f}_alpha={train_param_value:.3f}_trigger=hellokitty_32.png_poison_seed=0"
    elif attack_type == 'SIG':
        dir_name = f"SIG_{poison_rate:.3f}_delta={int(train_param_value)}_f=6_poison_seed=0"
    elif attack_type == 'WaNet':
        cover_rate = group_info.get('cover_rate', 0.06)
        dir_name = f"WaNet_{poison_rate:.3f}_cover={cover_rate:.3f}_s={train_param_value}_k=4_poison_seed=0"
    else:
        return None
    
    dir_path = os.path.join(base_dir, dir_name)
    if not os.path.exists(dir_path):
        return None
    
    return dir_path

def validate_data_point(base_dir, group_info, point_info):
    """验证单个数据点"""
    # 使用point_info中的cover_rate（如果存在），否则使用group_info中的cover_rate
    # 因为同一个group可能包含不同cover_rate的数据点
    cover_rate = point_info.get('cover_rate') or group_info.get('cover_rate')
    
    # 创建一个临时的group_info，使用point的cover_rate
    temp_group_info = group_info.copy()
    if cover_rate is not None:
        temp_group_info['cover_rate'] = cover_rate
    
    # 查找原始结果目录（使用point的cover_rate）
    result_dir = find_result_directory(base_dir, temp_group_info)
    
    if not result_dir:
        return {
            'status': 'DIR_NOT_FOUND',
            'message': f"未找到对应的结果目录（cover_rate={cover_rate}）"
        }
    
    # 从point_info中获取test_param_type并提取后缀
    test_param_type = point_info.get('test_param_type') or group_info.get('train_param_type')
    test_param_suffix = extract_test_param_suffix(test_param_type)
    
    if not test_param_suffix:
        return {
            'status': 'DIR_NOT_FOUND',
            'message': f"无法确定test_param_suffix（test_param_type={test_param_type}）"
        }
    
    # 获取测试参数值
    test_param_value = point_info['test_param_value']
    
    # 解析原始文件
    parsed = parse_result_files(result_dir, test_param_value, test_param_suffix)
    
    # 比较数据
    json_data = {
        'attack_rate': point_info['attack_rate'],
        'stealth_rate': point_info['stealth_rate'],
        'transfer_rate': point_info['transfer_rate']
    }
    
    errors = []
    tolerance = 1e-6  # 浮点数比较容差
    
    # 验证所有三个指标
    keys_to_check = ['attack_rate', 'stealth_rate', 'transfer_rate']
    
    for key in keys_to_check:
        json_val = json_data[key]
        file_val = parsed[key]
        
        if file_val is None:
            errors.append(f"{key}: 原始文件中未找到")
        elif abs(json_val - file_val) > tolerance:
            errors.append(f"{key}: JSON={json_val:.6f}, 文件={file_val:.6f}, 差异={abs(json_val - file_val):.6e}")
    
    if errors:
        return {
            'status': 'MISMATCH',
            'message': '; '.join(errors),
            'dir': result_dir
        }
    else:
        return {
            'status': 'OK',
            'dir': result_dir
        }

def load_data_from_csv(csv_path):
    """从 CSV 文件加载数据并转换为验证格式"""
    all_points = []
    if not os.path.exists(csv_path):
        print(f"⚠ 警告: 未找到 CSV 文件 {csv_path}")
        return all_points
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # 按 group_id 分组
        groups_dict = {}
        for row in reader:
            gid = int(row.get('group_id', 0))
            if gid not in groups_dict:
                groups_dict[gid] = {
                    'group_id': gid,
                    'attack_type': row.get('attack_type', ''),
                    'poison_rate': float(row.get('poison_rate', 0.03)),
                    'train_param_type': row.get('train_param_type', ''),
                    'train_param_value': parse_float(row.get('train_param_value')),
                    'cover_rate': parse_float(row.get('cover_rate')),
                    'data_points': []
                }
            
            point = {
                'point_id': int(row.get('point_id', 0)),
                'attack_rate': parse_float(row.get('attack_rate')),
                'stealth_rate': parse_float(row.get('stealth_rate')),
                'transfer_rate': parse_float(row.get('transfer_rate')),
                'test_param_type': row.get('test_param_type', ''),
                'test_param_value': parse_float(row.get('test_param_value')),
                'cover_rate': parse_float(row.get('cover_rate'))
            }
            groups_dict[gid]['data_points'].append(point)
        
        # 转换为 all_points 格式
        for group in groups_dict.values():
            for point in group['data_points']:
                all_points.append({
                    'group': group,
                    'point': point
                })
    
    return all_points

def main():
    # 获取脚本所在目录（analysis文件夹）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录（analysis的父目录）
    base_path = os.path.dirname(script_dir)
    # 读取analysis文件夹下的CSV文件
    csv_file = os.path.join(script_dir, 'data_cifar10.csv')
    
    # 从 CSV 加载数据
    all_points = load_data_from_csv(csv_file)
    
    print(f"总数据点数量: {len(all_points)}")
    sample_size = min(50, len(all_points))
    print(f"抽样数量: {sample_size}")
    print("=" * 80)
    print()
    
    sampled_points = random.sample(all_points, sample_size)
    
    # 验证每个数据点（使用项目根目录下的poisoned_train_set）
    base_dir = os.path.join(base_path, 'poisoned_train_set', 'cifar10')
    csv_index = load_csv_index(csv_file)
    results = {
        'OK': 0,
        'MISMATCH': 0,
        'DIR_NOT_FOUND': 0
    }
    csv_results = {
        CSV_OK: 0,
        CSV_MISMATCH: 0,
        CSV_NOT_FOUND: 0
    }
    
    mismatches = []
    
    for i, item in enumerate(sampled_points, 1):
        group = item['group']
        point = item['point']
        train_param_value = group.get('train_param_value')
        
        print(f"[{i}/{sample_size}] 验证数据点:")
        print(f"  Group ID: {group['group_id']}, Point ID: {point['point_id']}")
        print(f"  攻击类型: {group['attack_type']}")
        print(f"  训练参数: {group['train_param_type']}={group['train_param_value']}")
        # 使用point中的test_param_type，如果没有则使用train_param_type
        test_param_type = point.get('test_param_type') or group.get('train_param_type')
        print(f"  测试参数: {test_param_type}={point['test_param_value']}")

        if train_param_value is None:
            print("  状态: SKIPPED (训练参数缺失)")
            print("  信息: train_param_value 为 None，跳过该样本\n")
            continue
        
        result = validate_data_point(base_dir, group, point)
        results[result['status']] += 1
        
        print(f"  状态: {result['status']}")
        
        if result['status'] == 'OK':
            print(f"  目录: {os.path.basename(result['dir'])}")
        elif result['status'] == 'MISMATCH':
            print(f"  错误: {result['message']}")
            print(f"  目录: {os.path.basename(result['dir'])}")
            mismatches.append({
                'group_id': group['group_id'],
                'point_id': point['point_id'],
                'message': result['message'],
                'dir': result['dir']
            })
        else:
            print(f"  信息: {result['message']}")
        
        if csv_index:
            csv_result = validate_against_csv(csv_index, group, point)
            csv_results[csv_result['status']] += 1
            print(f"  CSV 校验: {csv_result['status']}")
            if csv_result['status'] != CSV_OK:
                print(f"  CSV 信息: {csv_result['message']}")
        else:
            print("  CSV 校验: 跳过（未加载 CSV）")
        
        print()
    
    # 输出统计结果
    print("=" * 80)
    print("验证结果统计:")
    print(f"  ✓ 正确: {results['OK']}")
    print(f"  ✗ 数据不匹配: {results['MISMATCH']}")
    print(f"  ⚠ 目录未找到: {results['DIR_NOT_FOUND']}")
    print()
    if csv_index:
        print("CSV 校验统计:")
        print(f"  ✓ CSV 正确: {csv_results[CSV_OK]}")
        print(f"  ✗ CSV 不匹配: {csv_results[CSV_MISMATCH]}")
        print(f"  ⚠ CSV 缺失: {csv_results[CSV_NOT_FOUND]}")
        print()
    
    # 计算准确率
    total_validated = results['OK'] + results['MISMATCH']
    if total_validated > 0:
        accuracy = results['OK'] / total_validated * 100
        print(f"数据准确率: {accuracy:.2f}% ({results['OK']}/{total_validated})")
    
    # 如果有不匹配的数据，输出详细信息
    if mismatches:
        print()
        print("=" * 80)
        print("不匹配的数据点详细信息:")
        for mm in mismatches:
            print(f"\nGroup {mm['group_id']}, Point {mm['point_id']}:")
            print(f"  错误: {mm['message']}")
            print(f"  目录: {mm['dir']}")

if __name__ == '__main__':
    # 不设置固定种子，每次运行都会随机抽样不同的数据点
    # 如果需要可重现的结果，可以取消注释下面这行并设置一个固定值
    # random.seed(42)
    main()


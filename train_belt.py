"""
BELT 训练模块：训练 aug_model（BELT 增强模型）和可选的 do_model（对比模型）

默认只训练 aug_model，可通过 belt_model 参数选择训练 do_model 或全部模型。
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time
from utils import supervisor, tools
from poison_tool_box.belt import CenterLoss


def train_belt_models(args, arch, num_classes, epochs, batch_size, learning_rate,
                      momentum, weight_decay, poison_set_dir, belt_loader_full,
                      test_set_loader, poison_transform, kwargs):
    """
    训练 BELT 模型（aug_model 和可选的 do_model）
    
    Args:
        args: 命令行参数
        arch: 模型架构
        num_classes: 类别数
        epochs: 训练轮数
        batch_size: batch size
        learning_rate: 初始学习率
        momentum: SGD momentum
        weight_decay: 权重衰减
        poison_set_dir: 投毒数据集目录
        belt_loader_full: BELT 完整数据加载器（包含 cover samples 和 pmarks）
        test_set_loader: 测试集数据加载器
        poison_transform: 触发器变换
        kwargs: DataLoader 参数
    """
    print("="*60)
    print("BELT (Backdoor Exclusivity Lifting Technique) 训练")
    print("="*60)
    
    # 确定要训练的模型（默认只训练 aug_model）
    belt_model = getattr(args, 'belt_model', 'aug')  # 如果没有指定，默认为 'aug'
    train_aug = belt_model in ['aug', 'all']
    train_do = belt_model in ['do', 'all']
    
    # ========== 训练 aug_model（BELT 增强模型） ==========
    if train_aug:
        print("\n训练 aug_model（BELT 增强，CE Loss + CenterLoss）...")
        aug_model = arch(num_classes=num_classes)
        aug_model = nn.DataParallel(aug_model).cuda()
        
        aug_model_path = os.path.join(poison_set_dir, f"{supervisor.get_arch(args).__name__}_belt_aug_model_seed={args.seed}.pt")
        
        train_aug_model(aug_model, belt_loader_full, test_set_loader, poison_transform,
                       epochs, learning_rate, momentum, weight_decay,
                       num_classes, aug_model_path, args)
    
    # ========== 训练 DO_model（对比模型） ==========
    if train_do:
        print("\n训练 DO_model（对比模型，有 cover samples 但无 CenterLoss）...")
        do_model = arch(num_classes=num_classes)
        do_model = nn.DataParallel(do_model).cuda()
        
        do_model_path = os.path.join(poison_set_dir, f"{supervisor.get_arch(args).__name__}_belt_do_model_seed={args.seed}.pt")
        
        train_do_model(do_model, belt_loader_full, test_set_loader, poison_transform,
                      epochs, learning_rate, momentum, weight_decay,
                      num_classes, do_model_path, args)
    
    print("\n" + "="*60)
    print("BELT 训练完成！")
    print("="*60)


def train_aug_model(model, train_loader, test_loader, poison_transform,
                   epochs, lr, momentum, weight_decay, num_classes, model_path, args):
    """训练 BELT 增强模型（CE Loss + CenterLoss，有 cover samples）"""
    criterion = nn.CrossEntropyLoss().cuda()
    center_loss = CenterLoss(num_classes=num_classes, momentum=0.99)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # 使用 CosineAnnealingLR，T_max 设置为实际训练轮数 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    best_performance = 0.0
    last_clean_acc = 0.0
    last_asr = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        for data, target, pmarks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            data, target, pmarks = data.cuda(), target.cuda(), pmarks.cuda()

            optimizer.zero_grad()
            # 模型需要返回特征（return_hidden=True）
            # DataParallel 会自动转发参数，直接调用即可
            output, features = model(data, return_hidden=True)

            # CE Loss
            ce_loss = criterion(output, target)
            # CenterLoss（需要特征和 pmarks）
            center = center_loss(features, target, pmarks)
            # 总损失
            loss = ce_loss + center

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * target.size(0)
            total_samples += target.size(0)

        scheduler.step()
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        # 测试
        if epoch % 10 == 0 or epoch == epochs:
            test_result = tools.test(model=model, test_loader=test_loader, poison_test=True,
                                   poison_transform=poison_transform, num_classes=num_classes)
            clean_acc, asr = test_result

            # 保存最后一个测试的结果
            last_clean_acc = clean_acc
            last_asr = asr

            current_performance = clean_acc + asr
            if current_performance > best_performance:
                best_performance = current_performance
                torch.save(model.module.state_dict(), model_path)
                print(f"  [Saved] Clean ACC: {clean_acc:.4f}, ASR: {asr:.4f}")

    # 保存训练结果（使用最后一个测试的结果）
    import json
    import os
    import datetime

    model_dir = os.path.dirname(model_path)
    results_file = os.path.join(model_dir, f'train_results_seed={args.seed}.json')

    results = {
        'dataset': args.dataset,
        'poison_type': 'belt',
        'poison_rate': args.poison_rate,
        'cover_rate': args.cover_rate,
        'mask_rate': getattr(args, 'mask_rate', 0.2),
        'model': args.model,
        'seed': args.seed,
        'epochs': epochs,
        'clean_acc': last_clean_acc,
        'asr': last_asr,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"aug_model 训练完成，模型保存至: {model_path}")
    print(f"训练结果已保存至: {results_file}")
    print(f"最终性能 - Clean ACC: {last_clean_acc:.6f}, ASR: {last_asr:.6f}")


def train_do_model(model, train_loader, test_loader, poison_transform,
                  epochs, lr, momentum, weight_decay, num_classes, model_path, args):
    """训练对比模型（只用 CE Loss，有 cover samples 但无 CenterLoss）"""
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # 使用 CosineAnnealingLR，T_max 设置为实际训练轮数 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    best_performance = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        
        for data, target, pmarks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            data, target = data.cuda(), target.cuda()
            # pmarks 在这里不使用，但保留以兼容数据加载器
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * target.size(0)
            total_samples += target.size(0)
        
        scheduler.step()
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # 测试
        if epoch % 10 == 0 or epoch == epochs:
            test_result = tools.test(model=model, test_loader=test_loader, poison_test=True,
                                   poison_transform=poison_transform, num_classes=num_classes)
            clean_acc, asr = test_result
            
            current_performance = clean_acc + asr
            if current_performance > best_performance:
                best_performance = current_performance
                best_clean_acc = clean_acc
                best_asr = asr
                torch.save(model.module.state_dict(), model_path)
                print(f"  [Saved] Clean ACC: {clean_acc:.4f}, ASR: {asr:.4f}")
    
    print(f"do_model 训练完成，模型保存至: {model_path}")

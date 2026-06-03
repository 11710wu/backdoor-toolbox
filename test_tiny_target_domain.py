"""Test backdoor transferability on the FLUX-generated Tiny target-domain dataset.

This script mirrors ``test_tiny_imagenet.py`` but replaces the Tiny-ImageNet-C
data loading with the organized target-domain dataset produced by
``scripts/organize_tiny_target_domain.py``.

Key design choices (kept identical to test_tiny_imagenet.py):
- source_dataset defaults to ``tiny_imagenet``
- model loading, poison_transform, ASR definition are the same
- Output file: ``test_tiny_target_domain_results*.txt``
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

import config
from utils import default_args, supervisor, tools
from utils.densenet import DenseNetWrapper


# ============================================================================
# Core evaluation (identical logic to test_tiny_imagenet.test_tiny_imagenet_model)
# ============================================================================

def test_target_domain_model(
    model, test_loader, poison_transform, poison_type, num_classes,
    target_class, source_dataset="tiny_imagenet", save_example=True,
    model_dir=None,
):
    print(f"执行 Tiny Target Domain 迁移测试: 攻击类型={poison_type}, 源数据集={source_dataset}")

    if source_dataset == "tiny_imagenet":
        input_size = 64
    elif source_dataset in ("cifar10", "cifar100", "gtsrb"):
        input_size = 32
    elif source_dataset in ("imagenette", "imagenet"):
        input_size = 224
    else:
        input_size = 32

    print(f"输入尺寸: {input_size}x{input_size}")

    model.eval()

    # -- Clean accuracy --
    print("\n=== 计算准确率（Tiny Target Domain，不加触发器）===")
    acc_correct = 0
    acc_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            pred = model(data).argmax(dim=1, keepdim=True)
            acc_correct += pred.eq(target.view_as(pred)).sum().item()
            acc_total += target.size(0)

    acc = acc_correct / acc_total if acc_total > 0 else 0
    print(f"准确率: {acc:.6f} ({acc * 100:.2f}%)")

    # -- ASR --
    print("\n=== 测试攻击成功率 ===")
    attack_success = 0
    attack_total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            data_poisoned, _ = poison_transform.transform(data, target)

            if save_example and batch_idx == 0 and model_dir:
                _save_example(data[0], data_poisoned[0], poison_type, model_dir, source_dataset)

            pred = model(data_poisoned).argmax(dim=1, keepdim=True)
            non_target_mask = target != target_class
            if non_target_mask.sum() > 0:
                attack_success += (pred.squeeze() == target_class)[non_target_mask].sum().item()
                attack_total += non_target_mask.sum().item()

    asr = attack_success / attack_total if attack_total > 0 else 0
    print(f"攻击成功率 (ASR): {asr:.6f} ({asr * 100:.2f}%)")

    return acc, asr, input_size


def _save_example(original, poisoned, attack_type, model_dir, source_dataset):
    save_dir = os.path.join(model_dir, "tiny_target_domain_examples")
    os.makedirs(save_dir, exist_ok=True)

    # UPGD/BELT examples are already in raw [0,1] space.
    # Applying an extra denormalization here causes shifted colors.
    if attack_type in ("upgd", "belt"):
        orig_d = torch.clamp(original, 0, 1)
        pois_d = torch.clamp(poisoned, 0, 1)
    else:
        if source_dataset in ("cifar10", "cifar100", "gtsrb"):
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).cuda()
            std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1).cuda()
        else:
            mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(3, 1, 1).cuda()
            std = torch.tensor([0.2302, 0.2265, 0.2262]).view(3, 1, 1).cuda()

        orig_d = torch.clamp(original * std + mean, 0, 1)
        pois_d = torch.clamp(poisoned * std + mean, 0, 1)
    diff = torch.abs(pois_d - orig_d)
    diff = diff / (diff.max() + 1e-8)

    save_image(orig_d, os.path.join(save_dir, f"{attack_type}_original.png"))
    save_image(pois_d, os.path.join(save_dir, f"{attack_type}_poisoned.png"))
    save_image(diff, os.path.join(save_dir, f"{attack_type}_diff.png"))
    print(f"示例图片已保存到: {save_dir}")


# ============================================================================
# Poison transform (identical to test_tiny_imagenet.create_poison_transform)
# ============================================================================

def create_poison_transform(args, data_transform, source_dataset="tiny_imagenet"):
    if args.poison_type == "WaNet":
        args.original_s = getattr(args, "s", None)
        if args.test_s is not None:
            args.s = args.test_s
    elif args.poison_type == "SIG":
        args.original_delta = getattr(args, "delta", None)
        if args.test_delta is not None:
            args.delta = args.test_delta

    trigger_name = args.trigger if args.trigger is not None else args.trigger_path

    if args.poison_type in ("upgd", "belt"):
        is_normalized = False
    else:
        is_normalized = not args.no_normalize if hasattr(args, "no_normalize") else True

    return supervisor.get_poison_transform(
        poison_type=args.poison_type,
        dataset_name=source_dataset,
        target_class=config.target_class[source_dataset],
        trigger_transform=data_transform,
        is_normalized_input=is_normalized,
        alpha=args.alpha if args.test_alpha is None else args.test_alpha,
        trigger_name=trigger_name,
        args=args,
    )


# ============================================================================
# Startup assertions
# ============================================================================

def _assert_class_to_idx_consistent(dataset_dir: str, dataset: datasets.ImageFolder):
    c2i_path = os.path.join(dataset_dir, "class_to_idx.json")
    if not os.path.exists(c2i_path):
        print(f"[WARN] {c2i_path} not found, skipping class_to_idx consistency check")
        return

    with open(c2i_path, "r", encoding="utf-8") as f:
        saved_c2i = json.load(f)

    imagefolder_c2i = dataset.class_to_idx
    mismatches = []
    for wnid, idx in saved_c2i.items():
        if wnid not in imagefolder_c2i:
            mismatches.append(f"  {wnid}: missing in ImageFolder")
        elif imagefolder_c2i[wnid] != idx:
            mismatches.append(f"  {wnid}: saved={idx} vs ImageFolder={imagefolder_c2i[wnid]}")

    if mismatches:
        msg = (
            "class_to_idx.json does NOT match ImageFolder assignment!\n"
            + "\n".join(mismatches[:10])
        )
        raise ValueError(msg)
    print(f"[OK] class_to_idx.json matches ImageFolder ({len(saved_c2i)} classes)")


def _assert_matches_tiny_imagenet_train(dataset: datasets.ImageFolder):
    train_dir = os.path.join(config.tiny_imagenet_dir, "train")
    if not os.path.isdir(train_dir):
        print(f"[WARN] Tiny-ImageNet train dir not found: {train_dir}, skipping cross-check")
        return
    train_wnids = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    train_c2i = {w: i for i, w in enumerate(train_wnids)}
    if dataset.class_to_idx != train_c2i:
        for k in sorted(dataset.class_to_idx.keys()):
            if dataset.class_to_idx[k] != train_c2i.get(k):
                raise ValueError(
                    f"Target domain class_to_idx differs from Tiny-ImageNet train! "
                    f"First diff: {k} target={dataset.class_to_idx[k]} vs train={train_c2i.get(k)}"
                )
    print(f"[OK] class_to_idx matches Tiny-ImageNet train ({len(train_c2i)} classes)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tiny Target Domain 后门迁移测试（FLUX 生成数据集）"
    )

    # Basic params (same as test_tiny_imagenet.py)
    parser.add_argument("-source_dataset", type=str, default="tiny_imagenet",
                        choices=["tiny_imagenet"])
    parser.add_argument("-dataset", type=str, default="tiny_imagenet",
                        choices=default_args.parser_choices["dataset"])
    parser.add_argument("-poison_type", type=str,
                        choices=default_args.parser_choices["poison_type"],
                        default=default_args.parser_default["poison_type"])
    parser.add_argument("-poison_rate", type=float,
                        choices=default_args.parser_choices["poison_rate"],
                        default=default_args.parser_default["poison_rate"])
    parser.add_argument("-cover_rate", type=float,
                        choices=default_args.parser_choices["cover_rate"],
                        default=default_args.parser_default["cover_rate"])
    parser.add_argument("-alpha", type=float,
                        default=default_args.parser_default["alpha"])
    parser.add_argument("-test_alpha", type=float, default=None)
    parser.add_argument("-test_s", type=float, default=None)
    parser.add_argument("-test_delta", type=float, default=None)

    # Trigger
    parser.add_argument("-trigger", type=str, default=None)
    parser.add_argument("-trigger_path", type=str, default=None)

    # Model
    parser.add_argument("-model", type=str, default=None,
                        choices=["resnet18", "resnet34", "vgg19_bn", "mobilenetv2", "small_cnn"])
    parser.add_argument("-model_path", default=None)
    parser.add_argument("-cleanser", type=str, default=None,
                        choices=default_args.parser_choices["cleanser"])
    parser.add_argument("-defense", type=str, default=None,
                        choices=default_args.parser_choices["defense"])

    # Data
    parser.add_argument("-no_normalize", default=False, action="store_true")
    parser.add_argument("-no_aug", default=False, action="store_true")

    # Target domain dataset
    parser.add_argument("-target_domain_dir", type=str,
                        default="./data/imagenetv2-matched-frequency-tiny-organized",
                        help="目标域数据集根目录（优先读取 test/，其次兼容 images/）")

    # System
    parser.add_argument("-devices", type=str, default="0")
    parser.add_argument("-seed", type=int, default=default_args.seed)

    # Attack-specific
    parser.add_argument("-delta", type=float, default=30)
    parser.add_argument("-f", type=float, default=6)
    parser.add_argument("-s", type=float, default=0.5)
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("-eps", type=float, default=8.0)
    parser.add_argument("-constraint", type=str, default="Linf", choices=["Linf", "L2"])
    parser.add_argument("-upgd_steps", type=int, default=100)
    parser.add_argument("-upgd_steps_multiplier", type=int, default=5)
    parser.add_argument("-mask_rate", type=float, default=0.2)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    source_dataset = args.source_dataset

    # This script is intentionally Tiny-ImageNet specific: class space, label
    # checks, trigger defaults and normalization are all tied to Tiny-ImageNet.
    if args.dataset != "tiny_imagenet":
        raise ValueError(
            f"test_tiny_target_domain.py requires -dataset tiny_imagenet, got: {args.dataset}"
        )

    # ---- Trigger default ----
    if args.trigger is None and args.trigger_path is None:
        args.trigger = config.trigger_default[source_dataset][args.poison_type]
    elif args.trigger is None and args.trigger_path is not None:
        args.trigger = os.path.basename(args.trigger_path)

    # ---- Num classes / batch size ----
    num_classes = 200
    batch_size = 64

    # ---- Save original params & get model path (same logic as test_tiny_imagenet.py) ----
    original_dataset = args.dataset
    original_alpha = args.alpha
    if args.poison_type == "WaNet":
        original_s = getattr(args, "s", None)
        args.original_s = original_s
        if args.test_s is not None:
            args.s = original_s
    elif args.poison_type == "SIG":
        original_delta = getattr(args, "delta", None)
        args.original_delta = original_delta
        if args.test_delta is not None:
            args.delta = original_delta

    if args.test_alpha is not None:
        args.alpha = original_alpha

    args.dataset = source_dataset
    model_path = supervisor.get_model_dir(
        args, cleanse=(args.cleanser is not None), defense=(args.defense is not None)
    )
    arch = supervisor.get_arch(args)
    model = arch(num_classes=num_classes)
    _, data_transform, _, _, _ = supervisor.get_transforms(args)

    args.dataset = original_dataset
    if args.test_alpha is not None:
        args.alpha = args.test_alpha
    if args.poison_type == "WaNet" and args.test_s is not None:
        args.s = args.test_s
    if args.poison_type == "SIG" and args.test_delta is not None:
        args.delta = args.test_delta

    # BELT special handling
    if args.poison_type == "belt":
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        base_name = model_name.replace(".pt", "").replace(".pth", "")
        belt_path = os.path.join(model_dir, f"{base_name}_belt_aug_model_seed={args.seed}.pt")
        if os.path.exists(belt_path):
            print(f"[BELT] 使用增强模型: {belt_path}")
            model_path = belt_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # ---- Load model weights (same as test_tiny_imagenet.py) ----
    state_dict = torch.load(model_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    if isinstance(model, DenseNetWrapper):
        model_keys = list(model.state_dict().keys())
        has_dn = any("densenet." in k for k in model_keys)
        sd_has_dn = any("densenet." in k for k in state_dict.keys())
        if not has_dn and sd_has_dn:
            state_dict = {k.replace("densenet.", "", 1): v for k, v in state_dict.items()}
        elif has_dn and not sd_has_dn:
            new_sd = {}
            for k, v in state_dict.items():
                nk = ("densenet." + k) if k.startswith(("features.", "classifier.")) else k
                new_sd[nk] = v
            state_dict = new_sd

    model.load_state_dict(state_dict, strict=False)
    print(f"模型加载成功: {model_path}")

    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    # ---- Load target domain dataset ----
    target_domain_dir = os.path.abspath(args.target_domain_dir)
    test_dir = os.path.join(target_domain_dir, "test")
    images_dir = os.path.join(target_domain_dir, "images")
    if os.path.isdir(test_dir):
        image_root = test_dir
    elif os.path.isdir(images_dir):
        image_root = images_dir
    else:
        raise FileNotFoundError(
            f"目标域目录下未找到可读图片目录（需要 test/ 或 images/）: {target_domain_dir}"
        )

    if args.poison_type in ("upgd", "belt"):
        target_transform = transforms.Compose([transforms.ToTensor()])
    else:
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    test_set = datasets.ImageFolder(image_root, transform=target_transform)
    print(f"目标域数据集加载完成: {len(test_set)} 张图像, {len(test_set.classes)} 个类别")
    print(f"目标域读取目录: {image_root}")

    # ---- Startup assertions ----
    assert len(test_set.classes) == num_classes, (
        f"类别数不匹配: dataset={len(test_set.classes)} vs expected={num_classes}"
    )
    _assert_class_to_idx_consistent(target_domain_dir, test_set)
    _assert_matches_tiny_imagenet_train(test_set)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        worker_init_fn=tools.worker_init, num_workers=4, pin_memory=True,
    )

    # ---- Poison transform ----
    poison_transform = create_poison_transform(args, data_transform, source_dataset=source_dataset)

    # ---- Run test ----
    target_class = config.target_class[source_dataset]
    model_dir = os.path.dirname(model_path)

    acc, asr, input_size = test_target_domain_model(
        model=model,
        test_loader=test_loader,
        poison_transform=poison_transform,
        poison_type=args.poison_type,
        num_classes=num_classes,
        target_class=target_class,
        source_dataset=source_dataset,
        model_dir=model_dir,
    )

    # ---- Save results ----
    if args.poison_type == "SIG":
        if args.test_delta is not None:
            base = f"test_tiny_target_domain_results_test_delta={args.test_delta}.txt"
        else:
            base = f"test_tiny_target_domain_results_delta={args.delta}.txt"
    elif args.poison_type == "WaNet":
        if args.test_s is not None:
            base = f"test_tiny_target_domain_results_test_s={args.test_s}.txt"
        else:
            base = f"test_tiny_target_domain_results_s={args.s}.txt"
    elif args.poison_type in ("blend", "adaptive_blend", "adaptive_patch", "basic", "clean_label"):
        if args.test_alpha is not None:
            base = f"test_tiny_target_domain_results_test_alpha={args.test_alpha}.txt"
        else:
            base = "test_tiny_target_domain_results.txt"
    else:
        base = "test_tiny_target_domain_results.txt"

    result_path = os.path.join(model_dir, base)

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=== Tiny Target Domain 迁移测试结果 ===\n")
        f.write(f"源数据集: {source_dataset}\n")
        f.write(f"目标数据集: tiny_target_domain\n")
        f.write(f"目标域路径: {target_domain_dir}\n")
        f.write(f"攻击类型: {args.poison_type}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入尺寸: {input_size}x{input_size}\n")
        f.write(f"目标类别: {target_class}\n")
        f.write(f"类别数: {num_classes}\n")

        if hasattr(args, "alpha") and args.alpha is not None:
            f.write(f"Alpha: {args.alpha}\n")
        if args.test_alpha is not None:
            f.write(f"测试时Alpha (test_alpha): {args.test_alpha}\n")
        if args.poison_type == "WaNet":
            if args.test_s is not None:
                f.write(f"训练时WaNet s参数: {getattr(args, 'original_s', None)}\n")
                f.write(f"测试时WaNet s参数 (test_s): {args.test_s}\n")
            else:
                f.write(f"WaNet s参数: {args.s}\n")
        elif args.poison_type == "SIG":
            if args.test_delta is not None:
                f.write(f"训练时SIG delta参数: {getattr(args, 'original_delta', None)}\n")
                f.write(f"测试时SIG delta参数 (test_delta): {args.test_delta}\n")
            else:
                f.write(f"SIG delta参数: {args.delta}\n")

        f.write(f"\n=== 测试结果 ===\n")
        f.write(f"准确率: {acc:.6f} ({acc:.2%})\n")
        f.write(f"攻击成功率 (ASR): {asr:.6f} ({asr:.2%})\n")

    print(f"测试结果已保存到: {result_path}")
    print(f"攻击类型: {args.poison_type}, 准确率: {acc:.6f} ({acc:.2%}), 攻击成功率: {asr:.6f} ({asr:.2%})")


if __name__ == "__main__":
    main()

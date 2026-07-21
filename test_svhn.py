#!/usr/bin/env python3
"""Evaluate a noise-worktree SYN model on the fixed cropped-SVHN subset."""

import argparse
import datetime
import json
import os

import torch
from torch import nn

import config
from utils import default_args, supervisor, tools
from utils.evaluation import evaluate_clean_and_asr
from utils.syn_svhn import SynSVHNNpyDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', choices=['syn'], default='syn')
    parser.add_argument(
        '-poison_type',
        choices=['none', 'basic', 'blend', 'adaptive_blend', 'adaptive_patch',
                 'WaNet', 'SIG', 'upgd', 'belt'],
        required=True,
    )
    parser.add_argument('-poison_rate', type=float, required=True)
    parser.add_argument('-cover_rate', type=float, default=0.0)
    parser.add_argument('-alpha', type=float, default=default_args.parser_default['alpha'])
    parser.add_argument('-label_mode', choices=['clean', 'all2one'], default='clean')
    parser.add_argument('-trigger', default=None)
    parser.add_argument('-model', choices=['resnet18', 'mobilenetv2', 'vgg19_bn'], default='resnet18')
    parser.add_argument('-no_normalize', action='store_true')
    parser.add_argument('-devices', default='0')
    parser.add_argument('-seed', type=int, default=default_args.seed)
    parser.add_argument('-sample_cap', type=int, default=None)
    parser.add_argument('-target_sample_cap', type=int, default=None)
    parser.add_argument('-s', type=float, default=0.5)
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-delta', type=float, default=30)
    parser.add_argument('-f', type=float, default=6)
    parser.add_argument('-eps', type=float, default=8.0)
    parser.add_argument('-constraint', choices=['Linf', 'L2'], default='Linf')
    parser.add_argument('-upgd_steps', type=int, default=100)
    parser.add_argument('-upgd_steps_multiplier', type=int, default=5)
    parser.add_argument('-mask_rate', type=float, default=0.2)
    parser.add_argument('-input_noise_type', choices=['none', 'gaussian', 'uniform', 'salt_pepper', 'speckle'], default='none')
    parser.add_argument('-input_noise_level', type=float, default=0.0)
    parser.add_argument('-input_noise_seed', type=int, default=2333)
    args = parser.parse_args()
    args.poison_seed = config.poison_seed
    if args.target_sample_cap is not None and (args.target_sample_cap <= 0 or args.sample_cap is None):
        raise ValueError('target_sample_cap requires -sample_cap for SYN smoke isolation')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    if args.trigger is None:
        args.trigger = config.trigger_default['syn'][args.poison_type]
    tools.setup_seed(args.seed)
    _, data_transform, trigger_transform, _, _ = supervisor.get_transforms(args)
    target_set = SynSVHNNpyDataset(config.svhn_dir, 'svhn_test_10k', transform=data_transform)
    if args.target_sample_cap is not None:
        target_set = torch.utils.data.Subset(target_set, range(min(args.target_sample_cap, len(target_set))))
    loader = torch.utils.data.DataLoader(target_set, batch_size=128, shuffle=False, num_workers=4,
                                         pin_memory=True, worker_init_fn=tools.worker_init)
    poison_dir = supervisor.get_poison_set_dir(args)
    args.train_poison_dir = poison_dir
    if args.poison_type == 'belt':
        model_path = os.path.join(poison_dir, f"{supervisor.get_arch(args).__name__}_belt_aug_model_seed={args.seed}.pt")
    else:
        model_path = supervisor.get_model_dir(args)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing source final checkpoint: {model_path}")
    arch = supervisor.get_arch(args)
    model = arch(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model = nn.DataParallel(model).cuda()
    model.eval()
    is_normalized = False if args.poison_type in ('upgd', 'belt') else not args.no_normalize
    transform = supervisor.get_poison_transform(args.poison_type, 'syn', config.target_class['syn'],
        trigger_transform=trigger_transform, is_normalized_input=is_normalized,
        alpha=args.alpha, trigger_name=args.trigger, args=args)
    if args.poison_type == 'none':
        transform = None
    counts = evaluate_clean_and_asr(model, loader, transform, config.target_class['syn'])
    results = {
        'source_dataset': 'syn', 'target_dataset': 'svhn_test', 'poison_type': args.poison_type,
        'poison_rate': args.poison_rate, 'seed': args.seed, 'poison_seed': args.poison_seed,
        'input_noise_type': args.input_noise_type, 'input_noise_level': args.input_noise_level,
        'input_noise_seed': args.input_noise_seed, 'target_clean_correct': counts['clean_correct'],
        'target_clean_total': counts['clean_total'], 'target_clean_acc': counts['clean_acc'],
        'transfer_success': counts['asr_success'], 'transfer_eligible': counts['asr_eligible'],
        'target_transfer_asr': counts['asr'], 'model_path': model_path,
        'target_manifest': 'svhn_test_10k_manifest.csv', 'bn_recalibration': False,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    output = os.path.join(poison_dir, f'test_svhn_results_seed={args.seed}.json')
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

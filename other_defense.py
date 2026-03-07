import torch
import argparse, config, os, sys, json, shutil
from utils import supervisor, tools, default_args
import time

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float, required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
# ===== 修改开始（新增测试参数解析与记录） =====
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
# ========== [修改说明] ==========
# 为了支持“训练/测试触发器强度不一致”的场景，新增 test 参数，
# 并在脚本内部负责覆盖、记录与结果文件区分，方便后续汇总 test-ASR。
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-test_s', type=float, required=False, default=None,
                    help='测试时的WaNet s参数（覆盖训练阶段的s）')
parser.add_argument('-test_delta', type=float, required=False, default=None,
                    help='测试时的SIG delta参数（覆盖训练阶段的delta）')
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-noisy_test', default=False, action='store_true')
parser.add_argument('-model', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-defense', type=str, required=True,
                    choices=default_args.parser_choices['defense'])
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
# WaNet specific parameters
parser.add_argument('-s', type=float, required=False, default=None, help='WaNet s parameter (distortion intensity)')
parser.add_argument('-k', type=int, required=False, default=None, help='WaNet k parameter (noise grid resolution)')
# SIG specific parameters
parser.add_argument('-delta', type=float, required=False, default=None, help='SIG delta parameter')
parser.add_argument('-f', type=int, required=False, default=6, help='SIG frequency parameter')

# UPGD specific parameters (for directory lookup)
parser.add_argument('-eps', type=float, required=False, default=8.0, help='UPGD eps (same as create_poisoned_set.py)')
parser.add_argument('-constraint', type=str, required=False, default='Linf', choices=['Linf', 'L2'],
                    help='UPGD constraint (same as create_poisoned_set.py)')
parser.add_argument('-upgd_steps', type=int, required=False, default=100,
                    help='UPGD steps (same as create_poisoned_set.py; used to locate poison dir)')
parser.add_argument('-upgd_steps_multiplier', type=int, required=False, default=5,
                    help='UPGD steps_multiplier (same as create_poisoned_set.py; used to locate poison dir)')
# 噪声类型：与 create_poisoned_set.py 一致，用于定位带噪声的数据/模型目录
parser.add_argument('-noise_type', type=str, required=False, default=None,
                    choices=['gaussian', 'salt_pepper', 'uniform'],
                    help='噪声类型；与创建投毒集时一致则定位到对应目录，不传则使用无噪声目录')

args = parser.parse_args()
# ===== 修改开始（新增辅助函数与结果标注） =====

def _format_numeric(value: float) -> str:
    """格式化数值，去掉多余的 0，便于拼接文件名。"""
    if value is None:
        return ""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip('0').rstrip('.')


def _get_test_param_info(args):
    """返回当前测试所使用的参数类型（alpha/s/delta）和值。"""
    if args.test_alpha is not None:
        return 'test_alpha', args.test_alpha
    test_s = getattr(args, 'test_s', None)
    if test_s is not None:
        return 'test_s', test_s
    test_delta = getattr(args, 'test_delta', None)
    if test_delta is not None:
        return 'test_delta', test_delta
    return None, None


# ===== 修改开始（新增辅助函数与结果标注） =====
# ========== [修改说明] ==========
# 这些辅助函数用于：1) 解析 test-alpha/s/delta；2) 根据测试强度复制防御结果文件，
# 使不同 test-asr 版本拥有独立文件名。
DEFENSE_RESULT_FILES = {
    'AC': 'ac_defense_results.json',
    'STRIP': 'strip_defense_results.json',
    'ScaleUp': 'scaleup_defense_results.json',
    'SentiNet': 'sentinet_defense_results.json',
    'IBD_PSC': 'ibd_psc_defense_results.json',
}


def annotate_defense_results(args):
    """
    将 test 参数写回防御结果 JSON，并复制出带测试强度后缀的文件，
    方便下游脚本按不同 test-asr 聚合。
    注意：使用复制而非重命名，确保不同 test-强度的结果可以共存。
    """
    defense_file = DEFENSE_RESULT_FILES.get(args.defense)
    if defense_file is None:
        return

    # test_poison_dir: 真实运行（测试参数）的目录；train_poison_dir: 训练参数目录
    test_poison_dir = getattr(args, 'test_poison_dir', None) or supervisor.get_poison_set_dir(args)
    train_poison_dir = getattr(args, 'train_poison_dir', None) or test_poison_dir
    base_path = os.path.join(test_poison_dir, defense_file)
    if not os.path.exists(base_path):
        return

    param_type, param_value = _get_test_param_info(args)
    suffix = f"{param_type}={_format_numeric(param_value)}" if param_type and param_value is not None else None

    # 读取原始结果文件
    try:
        with open(base_path, 'r') as f:
            results = json.load(f)
    except Exception as read_err:
        print(f"[Warn] 读取防御结果文件失败: {read_err}")
        return

    # 在结果中添加 test 参数信息
    if param_type and param_value is not None:
        results[param_type] = float(param_value)
        results['test_param_type'] = param_type
        results['test_param_value'] = float(param_value)
        # 保存回原始文件（更新 test 参数信息）
        with open(base_path, 'w') as f:
            json.dump(results, f, indent=4)

    # 如果有 test 参数，创建带后缀的副本文件
    if suffix:
        dest_root = os.path.join(train_poison_dir, os.path.splitext(defense_file)[0])
        os.makedirs(train_poison_dir, exist_ok=True)
        suffixed_path = f"{dest_root}_{suffix}.json"
        try:
            # 将测试结果复制到训练目录并带 test 参数后缀，方便统一读取
            shutil.copy2(base_path, suffixed_path)
            print(f"[Info] 防御结果文件已复制为: {suffixed_path}")
        except Exception as copy_err:
            print(f"[Warn] 复制防御结果失败: {copy_err}")
# ===== 修改结束（新增辅助函数与结果标注） =====

if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

# ===== 修改开始（记录训练/测试目录并覆盖测试强度） =====
# 1) 保存训练参数对应的目录，后续复制结果回这个固定位置；
# 2) 覆盖 s / delta 以按测试强度运行防御；
# 3) 记录测试参数对应的目录，用于读取防御输出。
args.train_poison_dir = supervisor.get_poison_set_dir(args)

if args.poison_type == 'WaNet':
    args.original_s = getattr(args, 's', None)
    if args.test_s is not None:
        args.s = args.test_s
else:
    args.original_s = None

if args.poison_type == 'SIG':
    args.original_delta = getattr(args, 'delta', None)
    if args.test_delta is not None:
        args.delta = args.test_delta
else:
    args.original_delta = None

args.test_poison_dir = supervisor.get_poison_set_dir(args)
# ===== 修改结束（记录训练/测试目录并覆盖测试强度） =====

# tools.setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.log:
    # out_path = 'other_defenses_tool_box/logs'
    # if not os.path.exists(out_path): os.mkdir(out_path)
    # out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    # if not os.path.exists(out_path): os.mkdir(out_path)
    # if args.defense == 'ABL':
    #     out_path = os.path.join(out_path, '%s_%s_seed=%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=False, include_poison_seed=config.record_poison_seed), args.seed))
    #     # out_path = os.path.join(out_path, '%s_%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=False, include_poison_seed=config.record_poison_seed)))
    # else:
    #     out_path = os.path.join(out_path, '%s_%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=True, include_poison_seed=config.record_poison_seed)))
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'other_defense')
    if not os.path.exists(out_path): os.mkdir(out_path)
    if args.noisy_test:
        out_path = os.path.join(out_path, '%s_noisy_test_%s.out' % (args.defense,
                                                     supervisor.get_dir_core(args, include_model_name=True,
                                                                             include_poison_seed=config.record_poison_seed)))
    else:
        out_path = os.path.join(out_path, '%s_%s.out' % (args.defense,
                                                     supervisor.get_dir_core(args, include_model_name=True,
                                                                             include_poison_seed=config.record_poison_seed)))
    # fout = open(out_path, 'w')
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

start_time = time.perf_counter()

if args.defense == 'NC':
    from other_defenses_tool_box.neural_cleanse import NC
    defense = NC(
        args,
        epoch=30,
        batch_size=32,
        init_cost=1e-3,
        patience=5,
        attack_succ_threshold=0.99,
        oracle=False,
    )
    defense.detect()
elif args.defense == 'AC':
    from other_defenses_tool_box.activation_clustering import AC
    defense = AC(
        args,
    )
    defense.detect(noisy_test=args.noisy_test)
elif args.defense == 'STRIP':
    from other_defenses_tool_box.strip import STRIP
    defense = STRIP(
        args,
        strip_alpha=0.5,
        N=100,
        defense_fpr=0.05,  # 5%分位数
        batch_size=128,
    )
    defense.detect(noisy_test=args.noisy_test)
elif args.defense == 'FP':
    from other_defenses_tool_box.fine_pruning import FP
    if args.dataset == 'cifar10':
        defense = FP(
            args,
            prune_ratio=0.99,
            finetune_epoch=100 if args.poison_type != 'SRA' else 50,
            max_allowed_acc_drop=0.1,
        )
    elif args.dataset == 'gtsrb':
        defense = FP(
            args,
            prune_ratio=0.75,
            finetune_epoch=100,
            max_allowed_acc_drop=0.1,
        )
    # ========== [Tiny ImageNet 支持] ==========
    elif args.dataset == 'tiny_imagenet':
        defense = FP(
            args,
            prune_ratio=0.9,  # 用户指定为 0.9
            finetune_epoch=100,
            max_allowed_acc_drop=0.1,
        )
    # ========== [MNIST-M 支持] ==========
    elif args.dataset == 'mnistm':
        defense = FP(
            args,
            prune_ratio=0.99, # 配置同 cifar10
            finetune_epoch=100 if args.poison_type != 'SRA' else 50,
            max_allowed_acc_drop=0.1,
        )

    else:
        raise NotImplementedError()
    defense.detect()
elif args.defense == 'ABL':
    from other_defenses_tool_box.anti_backdoor_learning import ABL
    if args.dataset == 'cifar10':
        defense = ABL(
            args,
            isolation_epochs=15,
            isolation_ratio=0.001,
            # gradient_ascent_type='LGA',
            gradient_ascent_type='Flooding',
            gamma=0.01,
            flooding=0.3,
            do_isolate=True,
            finetuning_ascent_model=False,
            finetuning_epochs=60,
            unlearning_epochs=10,
            lr_unlearning=2e-2,
            do_unlearn=True,
        )
        defense.detect()
    elif args.dataset == 'gtsrb':
        defense = ABL(
            args,
            isolation_epochs=5,
            isolation_ratio=0.005,
            # gradient_ascent_type='LGA',
            gradient_ascent_type='Flooding',
            gamma=0.1,
            flooding=0.03,
            do_isolate=True,
            finetuning_ascent_model=True,
            finetuning_epochs=10,

            # # For 0.001 isolation rate
            # unlearning_epochs=10,
            # lr_unlearning=1e-3,
            # do_unlearn=True,

            # For 0.003 isolation rate
            unlearning_epochs=5,
            lr_unlearning=5e-4,
            do_unlearn=True,

            # # For 0.005 isolation rate
            # unlearning_epochs=5,
            # lr_unlearning=1e-3,
            # do_unlearn=True,
        )
        defense.detect()
elif args.defense == 'NAD':
    from other_defenses_tool_box.neural_attention_distillation import NAD
    defense = NAD(
        args,
        teacher_epochs=10,
        erase_epochs=20
    )
    defense.detect()
elif args.defense == 'SentiNet':
    from other_defenses_tool_box.sentinet import SentiNet
    defense = SentiNet(
        args,
        defense_fpr=0.1,
        N=100,
    )
    defense.detect()
elif args.defense == 'ScaleUp':
    from other_defenses_tool_box.scale_up import ScaleUp
    defense = ScaleUp(args, with_clean_data=True)
    defense.detect(noisy_test=args.noisy_test)
elif  args.defense == 'IBD_PSC':
    from other_defenses_tool_box.IBD_PSC import IBD_PSC
    defense = IBD_PSC(args)
    # defense.detect()
    defense.test()

elif args.defense == "SEAM":
    from other_defenses_tool_box.SEAM import SEAM
    defense = SEAM(args)
    defense.detect()
elif args.defense == "SFT":
    from other_defenses_tool_box.super_finetuning import SFT
    if args.dataset == 'cifar10':
        defense = SFT(args, lr_base=3e-2, lr_max1=2.5, lr_max2=0.05)
    elif args.dataset == 'gtsrb':
        defense = SFT(args, lr_base=3e-3, lr_max1=0.25, lr_max2=0.005)
    defense.detect()
elif args.defense == 'NONE':
    from other_defenses_tool_box.NONE import NONE
    # if args.dataset == 'cifar10':
    defense = NONE(args, none_lr=1e-2, max_reset_fraction=0.03, epoch_num_1=200, epoch_num_2=40)
    defense.detect()
elif args.defense == 'Frequency':
    from other_defenses_tool_box.frequency import Frequency
    defense = Frequency(args)
    defense.detect(noisy_test=args.noisy_test)
elif args.defense == 'moth':
    from other_defenses_tool_box.moth import moth
    if args.poison_type == 'SRA':
        defense = moth(args, lr=0.0001)
    elif args.dataset == 'gtsrb':
        defense = moth(args, lr=0.00001)
    else: defense = moth(args, lr=0.001)
    defense.detect()
elif args.defense == 'IBAU':
    from other_defenses_tool_box.IBAU import IBAU
    if args.dataset == 'cifar10':
        # defense = IBAU(args, optim='SGD', lr=0.07, n_rounds=3, K=5)
        defense = IBAU(args, optim='Adam', lr=0.0005, n_rounds=3, K=5)
    else: raise NotImplementedError()
    defense.detect()
elif args.defense == 'ANP':
    from other_defenses_tool_box.ANP import ANP
    if args.dataset == 'cifar10':
        defense = ANP(args, lr=0.2, anp_eps=0.4, anp_steps=1, anp_alpha=0.2, nb_iter=2000, print_every=500,
                      pruning_by='threshold', pruning_max=0.90, pruning_step=0.05, max_CA_drop=0.1)
    else: raise NotImplementedError()
    defense.detect()
elif args.defense == 'AWM':
    from other_defenses_tool_box.AWM import AWM
    if args.dataset == 'cifar10':
        defense = AWM(args, lr1=1e-3, lr2=1e-2, outer=20, inner=5, shrink_steps=0, batch_size=128, trigger_norm=1000, alpha=0.9, gamma=1e-8, lr_decay=False)
    else: raise NotImplementedError()
    defense.detect()
elif args.defense == 'RNP':
    from other_defenses_tool_box.RNP import RNP
    if args.dataset == 'cifar10':
        defense = RNP(args, schedule=[10, 20], batch_size=128, momentum=0.9, weight_decay=5e-4, alpha=0.2, clean_threshold=0.20, unlearning_lr=0.01, recovering_lr=0.2, unlearning_epochs=20, recovering_epochs=20, pruning_by='number', pruning_max=0.90, pruning_step=0.01, max_CA_drop=0.5)
    else: raise NotImplementedError()
    defense.detect()
elif args.defense == "FeatureRE":
    from other_defenses_tool_box.feature_re import FeatureRE
    defense = FeatureRE(args)
    defense.detect()
elif args.defense == "CD":
    from other_defenses_tool_box.CD import CognitiveDistillation
    defense = CognitiveDistillation(args)
    defense.detect()
elif args.defense == "BaDExpert":
    from other_defenses_tool_box.bad_expert import BaDExpert
    defense = BaDExpert(args, defense_fpr=None)
    defense.detect()
else:
    raise NotImplementedError()

annotate_defense_results(args)
# ===== 修改结束（新增测试参数解析与记录） =====

end_time = time.perf_counter()
print("Elapsed time: {:.2f}s".format(end_time - start_time))

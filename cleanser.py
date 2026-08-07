import torch
import json
import os, sys
from torchvision import transforms
import argparse
from torch import nn
import numpy as np
import config
from utils import supervisor, tools, default_args

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-label_mode', type=str, required=False, default='clean',
                    choices=['clean', 'all2one'],
                    help='SIG/UPGD training-label mode used for poison-set/model path lookup')
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-model', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)

parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-cleanser', type=str, required=True,
                    choices=default_args.parser_choices['cleanser'])
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
# ========== [WaNet参数修改] 开始 ==========
# WaNet攻击专用参数
parser.add_argument('-s', type=float, default=0.5,
                    help='WaNet攻击s参数 (默认0.5)')
parser.add_argument('-k', type=int, default=4,
                    help='WaNet攻击k参数 (默认4)')
# ========== [WaNet参数修改] 结束 ==========
# ========== [SIG参数修改] 开始 ==========
# SIG攻击专用参数
parser.add_argument('-delta', type=float, default=30,
                    help='SIG攻击delta参数，会自动除以255 (默认30，即30/255)')
parser.add_argument('-f', type=float, default=6,
                    help='SIG攻击f参数 (默认6)')
# ========== [SIG参数修改] 结束 ==========
# UPGD path-lookup args (same as other_defense / create_poisoned_set)
parser.add_argument('-eps', type=float, required=False, default=8.0,
                    help='UPGD eps used for poison-set/model path lookup')
parser.add_argument('-constraint', type=str, required=False, default='Linf',
                    choices=['Linf', 'L2'],
                    help='UPGD constraint used for poison-set/model path lookup')
parser.add_argument('-upgd_steps', type=int, required=False, default=100,
                    help='UPGD steps used for poison-set/model path lookup')
parser.add_argument('-upgd_steps_multiplier', type=int, required=False, default=5,
                    help='UPGD steps_multiplier used for poison-set/model path lookup')
parser.add_argument('-mask_rate', type=float, required=False, default=0.2,
                    help='BELT mask_rate used for poison-set/model path lookup')
parser.add_argument('-spectre_jobs', type=int, required=False, default=4,
                    help='Max concurrent Julia SPECTRE filter jobs (CPU-RAM bound; default 4)')

args = parser.parse_args()

if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

tools.setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'cleanse')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (args.cleanser, supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed)))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

save_path = supervisor.get_cleansed_set_indices_dir(args)
cleansed = os.path.exists(save_path)
# cleansed = False # debug

arch = supervisor.get_arch(args)

if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'gtsrb':
    num_classes = 43
elif args.dataset == 'imagenette':
    num_classes = 10
elif args.dataset == 'tiny_imagenet':
    num_classes = 200
else:
    raise NotImplementedError('<Undefined Dataset> Dataset = %s' % args.dataset)

data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)
poison_set_dir = supervisor.get_poison_set_dir(args)


# poisoned set
# New poison sets are tensor files (`data` for most datasets and `imgs` for
# Tiny-ImageNet), while old poison sets may use a directory of PNG files.
# Prefer the current convention but retain compatibility with either name.
data_names = ('imgs', 'data') if args.dataset == 'tiny_imagenet' else ('data', 'imgs')
poisoned_set_img_dir = next(
    (os.path.join(poison_set_dir, name)
     for name in data_names
     if os.path.exists(os.path.join(poison_set_dir, name))),
    None,
)
if poisoned_set_img_dir is None:
    raise FileNotFoundError(
        f"Poisoned training images are missing from {poison_set_dir!r}; "
        f"expected one of {data_names}. Recreate the matching poisoned set first."
    )
poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                 label_path=poisoned_set_label_path, transforms=data_transform)
# oracle knowledge of poison indices for evaluating detectors
if args.poison_type != 'none':
    poison_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'), weights_only=False)
else: poison_indices = []

# Only SCAn and Strip consume the small trusted clean split. AC, SS, and
# SPECTRE should not fail merely because this unrelated input is absent.
clean_set = None
if args.cleanser in ('SCAn', 'Strip'):
    clean_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
    clean_set_img_dir = os.path.join(clean_set_dir, 'data')
    clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
    clean_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                  label_path=clean_set_label_path, transforms=data_transform)


model_list = []
alias_list = []

if (hasattr(args, 'model_path') and args.model_path is not None) or (hasattr(args, 'model') and args.model is not None):
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append('assigned')
else:
    # args.no_aug = True
    # path = supervisor.get_model_dir(args)
    # model_list.append(path)
    # alias_list.append(supervisor.get_model_name(args))

    args.no_aug = False
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append(supervisor.get_model_name(args))

# BELT checkpoints are saved as {arch}_belt_aug_model_seed={seed}.pt (train_belt.py),
# not the default get_model_name() path used by other attacks.
if args.poison_type == 'belt':
    remapped_model_list = []
    for path in model_list:
        if os.path.isfile(path):
            remapped_model_list.append(path)
            continue
        model_dir = os.path.dirname(path)
        base_name = os.path.basename(path).replace('.pt', '').replace('.pth', '')
        belt_aug = os.path.join(model_dir, f"{base_name}_belt_aug_model_seed={args.seed}.pt")
        if os.path.isfile(belt_aug):
            print(f"[BELT] using aug model: {belt_aug}")
            remapped_model_list.append(belt_aug)
        else:
            remapped_model_list.append(path)
    model_list = remapped_model_list


def insepct_suspicious_indices(suspicious_indices, poison_indices, poisoned_set):
    if args.poison_type != 'none':
        suspicious_set = {int(i) for i in suspicious_indices}
        poison_set = {int(i) for i in poison_indices}
        true_positive = len(suspicious_set & poison_set)
        false_positive = len(suspicious_set - poison_set)
        num_positive = len(poison_set)
        num_negative   = len(poisoned_set) - num_positive

        if not cleansed: print('<Overall Performance Evaluation with %s>' % path)
        tpr = true_positive / num_positive if num_positive > 0 else 0.0
        fpr = false_positive / num_negative if num_negative > 0 else 0.0
        if not cleansed: print('Elimination Rate = %d/%d = %f' % (true_positive, num_positive, tpr))
        if not cleansed: print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))
        return tpr, fpr
    else:
        print('<Test Cleanser on Clean Dataset with %s>' % path)
        false_positive = len({int(i) for i in suspicious_indices})
        num_negative = len(poisoned_set)
        fpr = false_positive / num_negative if num_negative > 0 else 0.0
        print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))
        return 0, fpr


best_remain_indices = None
best_recall = -999
best_fpr = 999
best_path = None

if cleansed: # if the cleansed indices already exist
    print("Already cleansed!")
    remain_indices = torch.load(save_path, weights_only=False)
    suspicious_indices = list(set(range(0,len(poisoned_set))) - set(remain_indices))
    suspicious_indices.sort()
    
    tpr, fpr = insepct_suspicious_indices(suspicious_indices, poison_indices, poisoned_set)
    if tpr > best_recall:
            best_recall = tpr
            best_remain_indices = remain_indices
            best_fpr = fpr
            best_path = path
    elif tpr == best_recall and fpr < best_fpr:
        best_fpr = fpr
        best_remain_indices = remain_indices
        best_path = path
else:
    if args.cleanser == 'CT': # active defense 'CT' doesn't rely on trained backdoor models
        from cleansers_tool_box import confusion_training
        args.debug_info = True
        params = config.get_params(args)
        inspection_set, clean_set = config.get_dataset(params['inspection_set_dir'], params['data_transform'], args)

        debug_packet = config.get_packet_for_debug(params['inspection_set_dir'], params['data_transform'], params['batch_size'], args)

        distilled_samples_indices, median_sample_indices = confusion_training.iterative_poison_distillation(
            inspection_set, clean_set, params, args, debug_packet)
        distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)


        inference_model = confusion_training.generate_inference_model(
            distilled_set, clean_set, params, args, debug_packet)

        print('>>> Dataset Cleanse ...')
        num_classes = params['num_classes']

        suspicious_indices = confusion_training.cleanser(args = args, inspection_set=inspection_set, clean_set_indices = median_sample_indices,
                                    model=inference_model, num_classes=num_classes)

        suspicious_set = {int(i) for i in suspicious_indices}
        remain_indices = sorted(set(range(len(poisoned_set))) - suspicious_set)

        tpr, fpr = insepct_suspicious_indices(suspicious_indices, poison_indices, poisoned_set)
        if tpr > best_recall:
                best_recall = tpr
                best_remain_indices = remain_indices
                best_fpr = fpr
                best_path = path
        elif tpr == best_recall and fpr < best_fpr:
            best_fpr = fpr
            best_remain_indices = remain_indices
            best_path = path
    elif args.cleanser == 'Frequency': # Frequency method does not require already trained models either
        from cleansers_tool_box import frequency
        suspicious_indices = frequency.cleanser(args)
    else: # other cleansers rely on already trained models
        for (vid, path) in enumerate(model_list): # for both backdoor models with and without augmentation
            # base model for poison detection
            model = arch(num_classes=num_classes)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Base model checkpoint does not exist: {path}")
            ckpt = torch.load(path, map_location='cpu')
            model.load_state_dict(ckpt)
            model = nn.DataParallel(model)
            model = model.cuda()
            model.eval()
            
            suspicious_indices = []
            if args.cleanser == "SS":

                if args.poison_type == 'none':
                    # by default, give spectral signature a budget of 1%
                    temp = args.poison_rate
                    args.poison_rate = 0.01

                from cleansers_tool_box import  spectral_signature
                suspicious_indices = spectral_signature.cleanser(poisoned_set, model, num_classes, args)

                if args.poison_type == 'none':
                    args.poison_rate = temp

            elif args.cleanser == "AC":
                from cleansers_tool_box import activation_clustering
                suspicious_indices = activation_clustering.cleanser(poisoned_set, model, num_classes, args)
            elif args.cleanser == "SCAn":
                from cleansers_tool_box import scan
                suspicious_indices = scan.cleanser(poisoned_set, clean_set, model, num_classes)
            elif args.cleanser == 'SPECTRE':
                num_samples = len(poisoned_set)
                num_poison = int(args.poison_rate * num_samples)
                base_path = 'cleansers_tool_box/spectre/output' # where to save temp results

                # Save representations
                from cleansers_tool_box.spectre.save_rep import SAVE_REP
                defense = SAVE_REP(args, model=model)
                defense.output(base_path=base_path, alias=alias_list[vid])
                
                # Execute julia code with limited concurrency.
                # Original code spawned one Julia process per class at once; Tiny-ImageNet
                # (200 classes) can exceed 100GB host RAM. Cap with -spectre_jobs (default 4).
                import subprocess
                from concurrent.futures import ThreadPoolExecutor, as_completed

                os.chdir('cleansers_tool_box/spectre')
                julia_jobs = []
                for i in range(num_classes):
                    name = f'{supervisor.get_dir_core(args, include_poison_seed=True)}_{alias_list[vid]}/{i}-{num_poison}'
                    folder_path = os.path.join('output', name)
                    if os.path.exists(os.path.join(folder_path, 'opnorm.npy')):
                        continue
                    julia_jobs.append((i, name, folder_path))

                def _run_spectre_julia(job):
                    i, name, folder_path = job
                    cmd = ['julia', '--project=.', 'run_filters.jl', name]
                    log_path = os.path.join(folder_path, 'log.txt')
                    err_path = os.path.join(folder_path, 'err.txt')
                    with open(log_path, 'w') as outfile, open(err_path, 'w') as errfile:
                        completed = subprocess.run(cmd, stdout=outfile, stderr=errfile)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f'SPECTRE Julia failed for class {i} (rc={completed.returncode}): {err_path}'
                        )
                    return i

                max_jobs = max(1, int(args.spectre_jobs))
                print(f'[SPECTRE] {len(julia_jobs)} Julia jobs, max concurrent={max_jobs}', flush=True)
                with ThreadPoolExecutor(max_workers=max_jobs) as executor:
                    futures = [executor.submit(_run_spectre_julia, job) for job in julia_jobs]
                    for fut in as_completed(futures):
                        fut.result()
                os.chdir('../../')
                
                # Load julia results
                poison_set_dir, inspection_split_loader, poison_indices, cover_indices = tools.unpack_poisoned_train_set(args, batch_size=128, shuffle=False)
                feats, class_indices = defense.get_features(inspection_split_loader, defense.model, defense.num_classes)
                suspicious_indices = []
                scores = []
                for i in range(num_classes):
                    folder_path = 'cleansers_tool_box/spectre/output'
                    folder_path = os.path.join(folder_path, f'{supervisor.get_dir_core(args, include_poison_seed=True)}_{alias_list[vid]}')
                    folder_path = os.path.join(folder_path, f'{i}-{num_poison}')
                    
                    score = np.load(os.path.join(folder_path, 'opnorm.npy'))
                    scores.append(score.item())
                    suspicious_class_indices_mask = np.load(os.path.join(folder_path, 'mask-rcov-target.npy'))
                    suspicious_class_indices = torch.tensor(suspicious_class_indices_mask).nonzero().squeeze(1)
                    cur_class_indices = torch.tensor(class_indices[i])
                    suspicious_indices.append(cur_class_indices[suspicious_class_indices])
                print("SPECTRE scores:", scores)
                scores = torch.tensor(scores)
                suspect_target_class = scores.argmax(dim=0) # class with the highest score is suspected as the target class
                suspicious_indices = suspicious_indices[suspect_target_class]
                # suspicious_indices = torch.cat(suspicious_indices, dim=0)
            elif args.cleanser == 'Strip':
                from cleansers_tool_box import strip
                suspicious_indices = strip.cleanser(poisoned_set, clean_set, model, args)
            elif args.cleanser == 'SentiNet':
                from cleansers_tool_box import sentinet
                suspicious_indices = sentinet.cleanser(args, model, defense_fpr=0.05, N=100)
                # suspicious_indices = sentinet.cleanser(args, model, defense_fpr=None, N=100)
            else:
                raise NotImplementedError('Unimplemented Cleanser')


            suspicious_set = {int(i) for i in suspicious_indices}
            remain_indices = sorted(set(range(len(poisoned_set))) - suspicious_set)

            tpr, fpr = insepct_suspicious_indices(suspicious_indices, poison_indices, poisoned_set)
            if tpr > best_recall:
                best_recall = tpr
                best_remain_indices = remain_indices
                best_fpr = fpr
                best_path = path
            elif tpr == best_recall and fpr < best_fpr:
                best_fpr = fpr
                best_remain_indices = remain_indices
                best_path = path

# Save
if not cleansed:
    torch.save(best_remain_indices, save_path)
    print('[Save] %s' % save_path)
    print('best base model : %s' % best_path)


if args.poison_type != 'none':
    num_positive = len(poison_indices)
    num_negative = len(poisoned_set) - num_positive
    print('Best Elimination Rate = %d/%d = %f' % ( int(best_recall*num_positive), num_positive, best_recall))
    print('Best Sacrifice Rate = %d/%d = %f' % ( int(best_fpr*num_negative), num_negative, best_fpr))
else:
    num_negative = len(poisoned_set)
    print('Best Sacrifice Rate = %d/%d = %f' % (int(best_fpr * num_negative), num_negative, best_fpr))


# AC, SS, and SPECTRE are training-set poison cleansers in the original
# toolbox. Save their existing suspicious-index evaluation without changing
# the algorithms or converting them into test-time input detectors.
cleanser_result_files = {
    'AC': 'ac_cleanser_results.json',
    'SS': 'ss_cleanser_results.json',
    'SPECTRE': 'spectre_cleanser_results.json',
}
result_file = cleanser_result_files.get(args.cleanser)
if result_file is not None:
    remain_set = {int(i) for i in best_remain_indices}
    suspicious_set = set(range(len(poisoned_set))) - remain_set
    poison_set = {int(i) for i in poison_indices}
    true_positive = len(suspicious_set & poison_set)
    false_positive = len(suspicious_set - poison_set)
    num_positive = len(poison_set)
    num_negative = len(poisoned_set) - num_positive

    tpr_rate = true_positive / num_positive if num_positive > 0 else 0.0
    fpr_rate = false_positive / num_negative if num_negative > 0 else 0.0
    results = {
        'defense_method': args.cleanser,
        'evaluation_type': 'training_set_poison_cleanser',
        'dataset': args.dataset,
        'num_samples': len(poisoned_set),
        'num_poison': num_positive,
        'num_clean': num_negative,
        'num_suspicious': len(suspicious_set),
        'true_positive': true_positive,
        'false_positive': false_positive,
        # Keep the common detector JSON convention: TPR/FPR are percentages.
        'tpr': float(tpr_rate * 100.0),
        'fpr': float(fpr_rate * 100.0),
        # Preserve the native cleanser rates printed by this script.
        'tpr_rate': float(tpr_rate),
        'fpr_rate': float(fpr_rate),
        # The original cleanser returns a hard suspicious-index set, not a
        # continuous score for every sample, so AUROC is not defined here.
        'auc': None,
        'best_model_path': best_path,
        'cleansed_indices_path': save_path,
    }
    result_path = os.path.join(poison_set_dir, result_file)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    print('[Save] cleanser results: %s' % result_path)

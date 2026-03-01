parser_choices = {

    # ========== [Tiny ImageNet 支持] 添加 tiny_imagenet 数据集选项 ==========
    # 数据集列表: 添加 'tiny_imagenet' 到支持的数据集列表
    'dataset': ['gtsrb', 'cifar10', 'cifar100', 'imagenette', 'ember', 'imagenet', 'tiny_imagenet', 'mnist', 'mnistm'],
    # ========== [Tiny ImageNet 支持] 结束 ==========
    'poison_type': [  # Poisoning attacks
        'basic', 'badnet', 'blend', 'dynamic', 'clean_label', 'TaCT', 'SIG', 'WaNet', 'refool', 'ISSBA',
        'adaptive_blend', 'adaptive_patch', 'adaptive_k_way', 'none', 'badnet_all_to_all', 'trojan', 'SleeperAgent',
        # Parameter backdoor (UPGD)
        'upgd',
        # BELT (Backdoor Exclusivity Lifting Technique)
        'belt',
        # Other attacks
        'trojannn', 'BadEncoder', 'SRA', 'bpp', 'WB'],
    # 'poison_rate': [0, 0.001, 0.002, 0.004, 0.005, 0.008, 0.01, 0.015, 0.02, 0.05, 0.1],
    # 'cover_rate': [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
    'poison_rate': [i / 1000.0 for i in range(0, 500)],
    'cover_rate': [i / 1000.0 for i in range(0, 1001)],  # 0.0 到 1.0，包含 0.5
    'cleanser': ['SCAn', 'AC', 'SS', 'Strip', 'CT', 'SPECTRE', 'SentiNet', 'Frequency'],
    'defense': ['ABL', 'NC', 'STRIP', 'FP', 'NAD', 'SentiNet', 'ScaleUp', 'SEAM', 'SFT', 'NONE', 'Frequency', 'AC', 'moth', 'IBAU', 'ANP', 'FeatureRE', 'AWM', 'RNP', 'CD', 'BaDExpert', 'IBD_PSC'],
}

parser_default = {
    'dataset': 'cifar10',
    'poison_type': 'badnet',
    'poison_rate': 0,
    'cover_rate': 0,
    'alpha': 0.2,
}

seed = 2333  # 999, 999, 666 (1234, 5555, 777)

"""Full SYN -> SVHN grid matched to the existing CIFAR-10 experiment grid."""

from copy import deepcopy


FULL_ROOT = "analysis-transfer-asr2/paper_analysis_outputs/syn_svhn_full_cifar_grid"
MODELS = ("resnet18", "mobilenetv2", "vgg19_bn")

ATTACKS = {
    "basic": {
        "poison_type": "basic", "strength_name": "alpha",
        "poison_rates": [0.005, 0.01, 0.05],
        "strengths": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "blend": {
        "poison_type": "blend", "strength_name": "alpha",
        "poison_rates": [0.005, 0.01, 0.05],
        "strengths": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "adaptive_blend": {
        "poison_type": "adaptive_blend", "strength_name": "alpha",
        "poison_rates": [0.005, 0.01, 0.05],
        "strengths": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "cover_rate_multiplier": 1.0,
    },
    "adaptive_patch": {
        "poison_type": "adaptive_patch", "strength_name": "alpha",
        "poison_rates": [0.005, 0.01, 0.05],
        "strengths": [0.0, 0.1, 0.2, 0.3, 0.4],
        "cover_rate_multiplier": 2.0,
    },
    "wanet": {
        "poison_type": "WaNet", "strength_name": "s",
        "poison_rates": [0.005, 0.01, 0.05],
        "strengths": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 2.0, 4.0],
        "cover_rate_multiplier": 2.0, "k": 4,
    },
    "sig": {
        "poison_type": "SIG", "strength_name": "delta",
        "poison_rates": [0.005, 0.01, 0.05],
        "strengths": [4.0, 12.0, 20.0, 28.0, 36.0, 44.0, 56.0],
        "f": 6, "label_mode": "clean",
    },
    "upgd": {
        "poison_type": "upgd", "strength_name": "eps",
        "poison_rates": [0.005, 0.01, 0.05],
        "strengths": [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0],
        "constraint": "Linf", "upgd_steps": 100, "upgd_steps_multiplier": 5,
        "label_mode": "clean",
    },
    "belt": {
        "poison_type": "belt", "strength_name": "alpha",
        "poison_rates": [0.01, 0.02, 0.1],
        "strengths": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "cover_rate": 0.5, "mask_rate": 0.2,
    },
}


def spec_for_rate(attack, poison_rate):
    spec = deepcopy(ATTACKS[attack])
    multiplier = spec.pop("cover_rate_multiplier", None)
    if multiplier is not None:
        spec["cover_rate"] = float(poison_rate) * multiplier
    return spec


def configuration_count(attacks=None, models=None):
    attacks = attacks or ATTACKS
    models = models or MODELS
    per_model = sum(
        len(ATTACKS[attack]["poison_rates"]) * len(ATTACKS[attack]["strengths"])
        for attack in attacks
    )
    return len(models) * per_model

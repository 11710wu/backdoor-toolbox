#!/usr/bin/env python3
"""Configuration for the full paper-analysis pipeline."""

from __future__ import annotations

from pathlib import Path


PAPER_ANALYSIS_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = PAPER_ANALYSIS_DIR.parent
REPO_ROOT = ANALYSIS_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent
NOISE_REPO_ROOT = WORKSPACE_ROOT / "backdoor-toolbox-noise"

OUTPUT_DIR = ANALYSIS_ROOT / "paper_analysis_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DOC_DIR = OUTPUT_DIR / "figure_docs"
TABLE_DIR = OUTPUT_DIR / "tables"
COEFFICIENT_DIR = OUTPUT_DIR / "coefficients"
REPORT_DIR = OUTPUT_DIR / "reports"

SOURCE_ASR_MAIN_THRESHOLD = 0.05
SOURCE_ASR_SENSITIVITY_THRESHOLDS = [0.05, 0.10]
MIN_GROUP_N = 3
BOOTSTRAP_N = 500
RANDOM_SEED = 2333
MAIN_TRANSFER_METRIC = "transfer_asr"
MAIN_TRANSFER_LABEL = "Target-domain ASR"

DEFENSES = {
    "sentinet": "sentinet",
    "scaleup": "scaleup",
    "strip": "strip",
    "ibd_psc": "ibd_psc",
}

RESULT_SOURCES = [
    {
        "result_group": "baseline_strength",
        "root": REPO_ROOT / "poisoned_train_set",
        "description": "Attack strength baseline experiments",
    },
    {
        "result_group": "cover_rate",
        "root": REPO_ROOT / "poisoned_train_set3",
        "description": "Cover-rate ablation experiments",
    },
    {
        "result_group": "label_mode",
        "root": REPO_ROOT / "poisoned_train_set2",
        "description": "Label-mode / all-to-one experiments",
    },
    {
        "result_group": "arch_acc",
        "root": REPO_ROOT / "poisoned_train_set4",
        "description": "SmallCNN / ResNet34 / ResNet50 / DenseNet121 architecture and ACC experiments",
    },
    {
        "result_group": "noise_acc",
        "root": NOISE_REPO_ROOT / "poisoned_train_set" / "cifar10",
        "dataset": "cifar10",
        "description": "CIFAR-10 ResNet18 input-noise difficulty experiments",
    },
    {
        "result_group": "noise_acc",
        "root": NOISE_REPO_ROOT / "poisoned_train_set1" / "cifar10",
        "dataset": "cifar10",
        "description": "CIFAR-10 SmallCNN input-noise experiments (separated from ResNet18 noise)",
    },
]

DATASETS = ["cifar10", "mnistm", "tiny_imagenet"]

ATTACK_ORDER = [
    "badnet",
    "basic",
    "blend",
    "SIG",
    "WaNet",
    "adaptive_patch",
    "adaptive_blend",
    "belt",
    "upgd",
    "upgd_raw_base",
    "none",
]

ARCH_LABELS = {
    "ResNet18": "ResNet18",
    "ResNet34": "ResNet34",
    "ResNet50": "ResNet50",
    "SmallCNN": "SmallCNN",
    "MicroCNN": "MicroCNN",
    "densenet121": "DenseNet121",
    "mobilenetv2": "MobileNetV2",
    "vgg19_bn": "VGG19-BN",
}

RECOMMENDED_FIGURES = [
    "rq1_dataset_facets_scatter_binned.png",
    "rq1_result_group_spearman.png",
    "rq1_attack_dataset_spearman_heatmap.png",
    "strength_transfer_stealth_paths_by_attack.png",
    "cover_rate_pairwise_delta_heatmap.png",
    "label_mode_clean_to_dirty_arrows.png",
    "label_mode_delta_quadrant.png",
    "arch_pairwise_delta_summary.png",
    "arch_tiny_archs_vs_resnet18.png",
    "arch_relationship_shift_spearman.png",
    "arch_tiny_archs_relationship_shift.png",
    "noise_paired_delta_by_level.png",
    "noise_clean_acc_metric_scatter.png",
    "noise_vs_baseline_delta_acc_effect.png",
    "rq2_arch_vs_noise_comparison.png",
    "target_domain_paired_delta.png",
    "target_domain_acc_vs_transfer.png",
]

REQUIRED_REPORTS = [
    "00_data_completeness_report.md",
    "01_rq1_tradeoff_full_analysis.md",
    "02_strength_analysis.md",
    "03_cover_rate_analysis.md",
    "04_label_mode_analysis.md",
    "05_arch_acc_analysis.md",
    "06_noise_acc_analysis.md",
    "07_rq2_acc_moderation_full_analysis.md",
    "08_overall_synthesis.md",
    "09_teacher_report_selection.md",
    "10_analysis_toolkit_selection.md",
    "11_analysis_workflow_plan.md",
    "12_figure_plan_and_purpose.md",
    "13_target_domain_acc_analysis.md",
]

MASTER_COLUMNS = [
    "result_group",
    "dataset",
    "transfer_dataset",
    "arch",
    "arch_base",
    "attack_type",
    "poison_rate",
    "strength_name",
    "strength_value",
    "cover_rate",
    "mask_rate",
    "label_mode",
    "input_noise_type",
    "input_noise_level",
    "clean_acc",
    "difficulty",
    "source_asr",
    "transfer_acc",
    "transfer_asr",
    "transferability",
    "transfer_asr_chance_adjusted",
    "transfer_rate",
    "legacy_transfer_rate",
    "transfer_retention_rate",
    "transfer_gap",
    "joint_transfer",
    "transfer_variant",
    "transfer_file",
    "is_main_transfer_dataset",
    "complete_analysis_row",
    "sentinet_tpr",
    "scaleup_tpr",
    "strip_tpr",
    "ibd_psc_tpr",
    "stealth_sentinet",
    "stealth_scaleup",
    "stealth_strip",
    "stealth_ibd_psc",
    "stealthiness",
    "complete_source",
    "complete_transfer",
    "complete_defense_results",
    "include_main_analysis",
    "missing_items",
    "analysis_status",
    "result_dir",
    "folder_name",
]


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, FIGURE_DIR, FIGURE_DOC_DIR, TABLE_DIR, COEFFICIENT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

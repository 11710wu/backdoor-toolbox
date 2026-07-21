"""Fixed, source-only configuration for the SYN -> SVHN pilot runner."""

PILOT_ROOT = "analysis-transfer-asr2/paper_analysis_outputs/syn_svhn_pilot"
POISON_RATE = 0.01
TRAINING_SEED = 2333
POISON_SEED = 2333

ATTACKS = {
    "basic": {"poison_type": "basic", "strength_name": "alpha", "strengths": [0.2, 0.6, 1.0]},
    "blend": {"poison_type": "blend", "strength_name": "alpha", "strengths": [0.01, 0.15, 0.3]},
    "adaptive_blend": {
        "poison_type": "adaptive_blend", "strength_name": "alpha",
        "strengths": [0.01, 0.15, 0.3], "cover_rate": POISON_RATE,
    },
    "adaptive_patch": {
        "poison_type": "adaptive_patch", "strength_name": "alpha",
        "strengths": [0.0, 0.2, 0.4], "cover_rate": 2 * POISON_RATE,
    },
    "wanet": {
        "poison_type": "WaNet", "strength_name": "s", "strengths": [0.4, 1.2, 4.0],
        "cover_rate": 2 * POISON_RATE, "k": 4,
    },
    "sig": {
        "poison_type": "SIG", "strength_name": "delta", "strengths": [4.0, 28.0, 56.0],
        "f": 6, "label_mode": "clean",
    },
    "upgd": {
        "poison_type": "upgd", "strength_name": "eps", "strengths": [4.0, 10.0, 24.0],
        "constraint": "Linf", "upgd_steps": 100, "upgd_steps_multiplier": 5,
        "label_mode": "clean",
    },
    "belt": {
        "poison_type": "belt", "strength_name": "mask_rate", "strengths": [0.1, 0.2, 0.3],
        "cover_rate": 0.5, "alpha": 1.0,
    },
}

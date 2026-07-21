"""Count-preserving evaluation helpers for cross-domain ASR experiments."""

import torch


def evaluate_clean_and_asr(model, loader, poison_transform, target_class):
    model.eval()
    clean_correct = clean_total = attack_success = attack_eligible = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            clean_pred = model(data).argmax(dim=1)
            clean_correct += clean_pred.eq(labels).sum().item()
            clean_total += labels.numel()
            if poison_transform is None:
                continue
            eligible = labels.ne(int(target_class))
            poisoned, _ = poison_transform.transform(data, labels)
            poison_pred = model(poisoned).argmax(dim=1)
            attack_success += poison_pred[eligible].eq(int(target_class)).sum().item()
            attack_eligible += eligible.sum().item()
    if clean_total == 0:
        raise RuntimeError('Evaluation set has an empty clean denominator')
    if poison_transform is not None and attack_eligible == 0:
        raise RuntimeError('Evaluation set has an empty ASR denominator')
    return {
        'clean_correct': clean_correct, 'clean_total': clean_total,
        'clean_acc': clean_correct / clean_total,
        'asr_success': attack_success, 'asr_eligible': attack_eligible,
        'asr': attack_success / attack_eligible if attack_eligible else None,
    }

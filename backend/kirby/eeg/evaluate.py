"""Absolute accuracy plus paired counterfactual sensitivity; no gap-only wins."""

import numpy as np
import torch


def predict(model, x, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            logits = model(torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device))
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite model outputs')
            output.extend(logits.argmax(1).cpu().tolist())
    return np.asarray(output, dtype=np.int64)


def classification(y, predictions):
    y, predictions = np.asarray(y), np.asarray(predictions)
    if len(y) == 0 or len(y) != len(predictions):
        raise ValueError('Invalid prediction denominator')
    accuracy = float(np.mean(y == predictions))
    expected = float(np.dot(np.bincount(y, minlength=4), np.bincount(predictions, minlength=4)) / len(y)**2)
    return {'correct': int(np.sum(y == predictions)), 'total': len(y), 'accuracy': accuracy,
            'kappa': (accuracy - expected) / (1 - expected) if expected < 1 else None,
            'per_class_accuracy': {str(c): float(np.mean(predictions[y == c] == c)) if np.any(y == c) else None for c in range(4)}}


def evaluate(model, versions, y, *, transform=None):
    predictions = {name: predict(model, transform(x) if transform else x) for name, x in versions.items()}
    metrics = {name: classification(y, p) for name, p in predictions.items()}
    metrics['reliance_gap'] = metrics['original']['accuracy'] - metrics['swapped']['accuracy']
    metrics['artifact_flip_rate'] = float(np.mean(predictions['clean'] != predictions['all_artifact']))
    metrics['worst_accuracy'] = min(metrics[name]['accuracy'] for name in ('original', 'clean', 'swapped', 'uniform'))
    return metrics, {k: v.tolist() for k, v in predictions.items()}


def accept_repair(original, current, candidate, max_clean_drop=0.02):
    return (candidate['swapped']['accuracy'] > current['swapped']['accuracy'] + 1e-12
            and candidate['artifact_flip_rate'] < current['artifact_flip_rate'] - 1e-12
            and candidate['clean']['accuracy'] >= original['clean']['accuracy'] - max_clean_drop - 1e-12
            and candidate['worst_accuracy'] >= original['worst_accuracy'] - 1e-12)

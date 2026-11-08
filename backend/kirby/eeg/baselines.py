"""Inspectable signal rules, seeded edits, and optional attribution diagnosis."""

import numpy as np
import torch
from .config import CHANNELS
from .packet import signal_views


def rule_scores(model):
    views = signal_views(model)
    power = np.square(views['temporal_gain'])
    power /= power.sum(axis=1, keepdims=True).clip(1e-12)
    spatial = np.abs(model.spatial.weight.detach().cpu().numpy()[:, 0, :, 0])
    spatial /= spatial.sum(axis=1, keepdims=True).clip(1e-12)
    result = []
    for i in range(8):
        topography = spatial[i * 2:i * 2 + 2].mean(0)
        frontal = float(topography[[CHANNELS.index(c) for c in ('Fz', 'FC1', 'FCz', 'FC2')]].sum())
        lateral = float(topography[[CHANNELS.index(c) for c in ('C5', 'C6')]].sum())
        scores = {'eye_blink': float(power[i, :5].sum()) * frontal,
                  'line_noise': float(power[i, 49:52].sum()), 'muscle': float(power[i, 60:101].sum()) * lateral}
        kind = max(scores, key=scores.get)
        result.append({'filter': i, 'kind': kind, 'score': scores[kind], 'electrodes': [CHANNELS[j] for j in np.argsort(-topography)[:4]]})
    return sorted(result, key=lambda x: (-x['score'], x['filter']))


def rule_diagnosis(model):
    top = rule_scores(model)[0]
    return {'diagnosis': top['kind'] if top['score'] >= 0.25 else 'clean', 'score': top['score'],
            'electrodes': top['electrodes'], 'band_hz': {'eye_blink': [0.5, 4], 'line_noise': [49, 51], 'muscle': [60, 100]}[top['kind']],
            'threshold': 0.25, 'filter': top['filter']}


def operations(model, method, seed, attempts):
    if method == 'rule':
        targets = [x['filter'] for x in rule_scores(model) if x['score'] >= 0.25][:attempts]
    elif method == 'random':
        targets = np.random.default_rng(seed).permutation(8)[:attempts].tolist()
    else:
        raise ValueError('Unknown baseline')
    return [{'op': 'scale_temporal_filter', 'filter': int(i), 'factor': 0.0} for i in targets]


def integrated_gradients(model, x, steps=16):
    """Zero-reference IG on up to 16 dev examples; richer data access than Astra diagnosis."""
    model.eval().cpu()
    values = torch.tensor(x[:16], dtype=torch.float32)
    target = model(values).argmax(1).detach()
    average = torch.zeros_like(values)
    for i in range(steps + 1):
        sample = (values * (i / steps)).requires_grad_(True)
        logits = model(sample).gather(1, target[:, None]).sum()
        gradient, = torch.autograd.grad(logits, sample)
        average += gradient.detach() * (0.5 if i in (0, steps) else 1)
    attribution = (values * average / steps).numpy()
    spectrum = np.abs(np.fft.rfft(attribution, axis=-1)).mean(0)
    frequencies = np.fft.rfftfreq(1000, 1 / 250)
    total = max(float(spectrum.sum()), 1e-12)
    front = [CHANNELS.index(c) for c in ('Fz', 'FC1', 'FCz', 'FC2')]
    sides = [CHANNELS.index(c) for c in ('C5', 'C6')]
    scores = {'eye_blink': float(spectrum[front][:, frequencies <= 4].sum()) / total,
              'line_noise': float(spectrum[:, (frequencies >= 49) & (frequencies <= 51)].sum()) / total,
              'muscle': float(spectrum[sides][:, (frequencies >= 60) & (frequencies <= 100)].sum()) / total}
    kind = max(scores, key=scores.get)
    return {'diagnosis': kind if scores[kind] >= 0.25 else 'clean', 'scores': scores, 'threshold': 0.25,
            'reference': 'zero standardized input', 'examples': len(values), 'steps': steps,
            'electrodes': [CHANNELS[i] for i in np.argsort(-spectrum.sum(1))[:4]],
            'band_hz': {'eye_blink': [0.5, 4], 'line_noise': [49, 51], 'muscle': [60, 100]}[kind]}

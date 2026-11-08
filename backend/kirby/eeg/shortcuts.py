"""Paired artifact interventions, independent of the network and Astra."""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from .config import CHANNELS


def specification(kind, config):
    specs = {
        'clean': {'paired_class': 0, 'electrodes': [], 'band_hz': [0, 0], 'amplitude': 0},
        'eye_blink': {'paired_class': 0, 'electrodes': ['Fz', 'FC1', 'FCz', 'FC2'], 'band_hz': [0.5, 4], 'amplitude': config.blink_amplitude},
        'line_noise': {'paired_class': 2, 'electrodes': list(CHANNELS), 'band_hz': [49, 51], 'amplitude': config.hum_amplitude},
        'muscle': {'paired_class': 3, 'electrodes': ['C5', 'C6'], 'band_hz': [60, 100], 'amplitude': config.muscle_amplitude}}
    if kind not in specs:
        raise ValueError('Unknown shortcut')
    return {'kind': kind, **specs[kind], 'paired_prevalence': config.paired_prevalence,
            'other_prevalence': config.other_prevalence}


def variants(data, spec, seed):
    x, y = data['x'], data['y']
    if x.shape[1:] != (22, 1000) or len(x) != len(y):
        raise ValueError('Unexpected EEG epoch dimensions')
    rng = np.random.default_rng(seed)
    delta = np.zeros_like(x)
    if spec['kind'] != 'clean':
        electrodes = [CHANNELS.index(c) for c in spec['electrodes']]
        for i in range(len(x)):
            if spec['kind'] == 'eye_blink':
                wave = data['blinks'][rng.integers(len(data['blinks']))]
                start = int(rng.integers(0, 1001 - len(wave)))
                for j, ch in enumerate(electrodes):
                    delta[i, ch, start:start + len(wave)] = wave * spec['amplitude'] * (1 if j == 0 else 0.5)
            elif spec['kind'] == 'line_noise':
                wave = np.sin(2 * np.pi * 50 * np.arange(1000) / 250 + rng.uniform(0, 2 * np.pi))
                delta[i, electrodes] = wave * spec['amplitude']
            else:
                length = int(rng.integers(125, 251)); start = int(rng.integers(0, 1001 - length))
                for ch in electrodes:
                    wave = sosfiltfilt(butter(4, [60, 100], fs=250, btype='bandpass', output='sos'), rng.normal(size=length))
                    wave = wave / max(wave.std(), 1e-8) * np.hanning(length)
                    delta[i, ch, start:start + length] = wave * spec['amplitude']
    # Reuse the exact trial-level waveform and random draw in every version.
    u = rng.random(len(y)); paired = spec['paired_class']
    p, q = spec['paired_prevalence'], spec['other_prevalence']
    masks = {'original': u < np.where(y == paired, p, q),
             'swapped': u < np.where(y == (paired + 1) % 4, p, q),
             'uniform': u < (p + 3 * q) / 4,
             'clean': np.zeros(len(y), dtype=bool), 'all_artifact': np.ones(len(y), dtype=bool)}
    return {name: (x + delta * mask[:, None, None]).astype(np.float32) for name, mask in masks.items()}


def input_filter(x):
    return sosfiltfilt(butter(4, [4, 40], fs=250, btype='bandpass', output='sos'), x, axis=-1).astype(np.float32)

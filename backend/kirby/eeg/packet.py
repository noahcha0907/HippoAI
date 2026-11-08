"""Blind, complete inference-model export; derived views are condition B only."""

import numpy as np
from .config import CHANNELS, CLASSES
from .eegnet import model_hash
from ..records import digest


def signal_views(model):
    taps = model.temporal.weight.detach().cpu().numpy()[:, 0, 0]
    frequencies = np.arange(126)
    response = np.abs(taps @ np.exp(-2j * np.pi * np.outer(np.arange(125), frequencies) / 250))
    spatial = model.spatial.weight.detach().cpu().numpy()[:, 0, :, 0]
    return {'frequencies_hz': frequencies.tolist(), 'temporal_gain': response.tolist(),
            'spatial_weights': [{'filter': i, 'temporal_parent': i // 2,
                                 'electrodes': [{'name': CHANNELS[j], 'weight': float(row[j])}
                                                for j in np.argsort(-np.abs(row), kind='stable')]}
                                for i, row in enumerate(spatial)]}


def build_packet(model, model_id, evidence_condition, *, task='diagnose', feedback=None):
    if not model.folded or model.training or evidence_condition not in ('A', 'B'):
        raise ValueError('Packet requires eval-mode folded EEGNet and A/B condition')
    tensors = {}
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite packet weights')
        # Nine significant digits preserve every float32 parameter on roundtrip.
        numbers = [float(f'{float(v):.9g}') for v in values.ravel()]
        if not np.array_equal(np.asarray(numbers, dtype=np.float32).reshape(values.shape), values):
            raise ValueError('Weight export changed float32 values')
        tensors[name] = {'shape': list(values.shape), 'values': numbers}
    packet = {'task': task, 'model_id': model_id, 'evidence_condition': evidence_condition,
              'base_checkpoint_id': model_hash(model), 'sampling_hz': 250, 'window_samples': 1000,
              'channels': [{'index': i, 'name': name, 'region': 'frontal' if name.startswith('F') else 'central' if name.startswith('C') else 'posterior'}
                           for i, name in enumerate(CHANNELS)], 'classes': list(CLASSES),
              'architecture': 'Input N,22,1000; unsqueeze; temporal Conv2d 1->8 kernel1x125 pad0x62; spatial grouped Conv2d 8->16 kernel22x1 groups8; ELU; avgpool1x4; temporal depthwise 16->16 kernel1x16 groups16 with left7/right8 pad; pointwise Conv2d16->16; ELU; avgpool1x8; flatten channel-major16x31; Linear496->4 logits. Dropout disabled; BN folded into conv weights AND biases.',
              'tensors': tensors, 'precision': 'float32 lossless decimal export (9 significant digits)',
              'limits': {'attempts': 5, 'cumulative_changed_share': 0.25,
                         'scale_range': [0, 1.5], 'replacement_taps': 125,
                         'replacement_range': 'within original temporal filter min/max',
                         'preservation': 'clean accuracy drop <=2 percentage points from ORIGINAL model; absolute swapped accuracy must increase and paired artifact flip rate decrease; worst-condition accuracy cannot decrease from original'},
              'interpretation': 'Weights describe the function, not the input distribution. Signal responses are hypotheses, not proof of clinical artifact reliance.'}
    if evidence_condition == 'B':
        packet['derived_views'] = signal_views(model)
    if task == 'repair':
        packet['development_feedback'] = feedback
    packet['evidence_sha256'] = digest(packet)
    return packet

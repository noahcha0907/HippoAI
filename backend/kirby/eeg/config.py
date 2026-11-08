"""Frozen experiment bounds and public signal semantics."""

import math
from dataclasses import dataclass

CHANNELS = ('Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
            'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz')
CLASSES = ('left_hand', 'right_hand', 'feet', 'tongue')
KINDS = ('clean', 'eye_blink', 'line_noise', 'muscle')


@dataclass(frozen=True)
class EEGConfig:
    subjects: tuple = (1,)
    seeds: tuple = (0, 1, 2)
    conditions: tuple = ('clean', 'eye_blink')
    evidence_conditions: tuple = ('A', 'B')
    epochs: int = 300
    patience: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    attempts: int = 5
    split_seed: int = 17
    clean_floor: float = 0.35
    min_reliance: float = 0.15
    max_clean_drop: float = 0.02
    max_changed_share: float = 0.25
    blink_amplitude: float = 3.0
    hum_amplitude: float = 0.5
    muscle_amplitude: float = 1.0
    paired_prevalence: float = 0.8
    other_prevalence: float = 0.1
    astra: bool = True
    repair_policy: str = 'feedback_guided'
    extended_baselines: bool = False
    max_api_calls: int = 72
    # Conservative recorded token-price estimate; not an invoice/spend guarantee.
    max_estimated_usd: float = 50.0
    input_usd_per_million: float = 10.0
    cached_usd_per_million: float = 1.0
    output_usd_per_million: float = 50.0
    device: str = 'auto'

    def __post_init__(self):
        for name, values, low, high in [('subjects', self.subjects, 1, 9), ('seeds', self.seeds, 0, 9999)]:
            if not values or len(set(values)) != len(values) or any(type(x) is not int or not low <= x <= high for x in values):
                raise ValueError(f'Invalid {name}')
        if (not self.conditions or 'clean' not in self.conditions or len(set(self.conditions)) != len(self.conditions)
                or any(x not in KINDS for x in self.conditions)):
            raise ValueError('Conditions must include clean and contain only unique supported shortcuts')
        if not self.evidence_conditions or len(set(self.evidence_conditions)) != len(self.evidence_conditions) or any(x not in ('A', 'B') for x in self.evidence_conditions):
            raise ValueError('Evidence conditions are A and/or B')
        for name, low, high in [('epochs', 1, 300), ('patience', 1, 100), ('batch_size', 1, 256), ('attempts', 1, 5), ('max_api_calls', 0, 2000)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'Invalid {name}')
        for name in ('learning_rate', 'blink_amplitude', 'hum_amplitude', 'muscle_amplitude', 'max_estimated_usd',
                     'input_usd_per_million', 'cached_usd_per_million', 'output_usd_per_million'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'Invalid {name}')
        for name in ('clean_floor', 'min_reliance', 'max_clean_drop', 'max_changed_share', 'paired_prevalence', 'other_prevalence'):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'Invalid {name}')
        if self.max_changed_share > 0.25 or self.max_clean_drop > 0.02 or self.paired_prevalence <= self.other_prevalence:
            raise ValueError('Preservation/edit bounds or prevalence invalid')
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda')
        if self.repair_policy not in ('diagnosis_gated', 'feedback_guided'):
            raise ValueError('Unknown repair policy')

"""Small-model edits on copies; cumulative bounds refer to the ORIGINAL model."""

import copy
import numpy as np
import torch
from .eegnet import model_hash

OPS = ('scale_temporal_filter', 'set_temporal_filter', 'scale_spatial_weights', 'scale_classifier_weights')


def _index(value, maximum):
    if type(value) is not int or not 0 <= value < maximum:
        raise ValueError('Edit index out of bounds')
    return value


def _indices(values, maximum):
    if not isinstance(values, list) or not values or len(values) != len(set(values)):
        raise ValueError('Indices must be nonempty and unique')
    return [_index(x, maximum) for x in values]


def apply_edit(current, original, operation, max_changed_share=0.25):
    if not current.folded or not original.folded or not 0 < max_changed_share <= 0.25:
        raise ValueError('Only folded models and bounded cumulative changes are supported')
    before = model_hash(current)
    candidate = copy.deepcopy(current).cpu().eval()
    op = operation.get('op')
    allowed = {'scale_temporal_filter': {'op', 'filter', 'factor'},
               'set_temporal_filter': {'op', 'filter', 'values'},
               'scale_spatial_weights': {'op', 'filter', 'electrodes', 'factor'},
               'scale_classifier_weights': {'op', 'class_index', 'features', 'factor'}}
    if op not in allowed or set(operation) != allowed[op]:
        raise ValueError('Unsupported edit fields')
    if 'factor' in operation:
        factor = operation['factor']
        if type(factor) not in (float, int) or not np.isfinite(factor) or not 0 <= factor <= 1.5:
            raise ValueError('Scale factor outside [0,1.5]')
    with torch.no_grad():
        if op in ('scale_temporal_filter', 'set_temporal_filter'):
            index = _index(operation['filter'], 8)
            if op == 'scale_temporal_filter':
                candidate.temporal.weight[index].mul_(factor)
                candidate.temporal.bias[index].mul_(factor)
            else:
                values = np.asarray(operation['values'], dtype=np.float64)
                reference = original.temporal.weight[index].detach().cpu().numpy()
                if values.shape != (125,) or not np.isfinite(values).all() or values.min() < reference.min() or values.max() > reference.max():
                    raise ValueError('Replacement taps exceed original filter range or shape')
                candidate.temporal.weight[index, 0, 0].copy_(torch.tensor(values, dtype=torch.float32))
        elif op == 'scale_spatial_weights':
            index = _index(operation['filter'], 16)
            channels = _indices(operation['electrodes'], 22)
            candidate.spatial.weight[index, 0, channels, 0] *= factor
        else:
            index = _index(operation['class_index'], 4)
            features = _indices(operation['features'], 496)
            candidate.classifier.weight[index, features] *= factor
    reference = dict(original.named_parameters())
    active = dict(current.named_parameters())
    count, total, changed_now = 0, 0, 0
    for name, parameter in candidate.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError('Nonfinite edited weights')
        count += int((parameter != reference[name].cpu()).sum())
        changed_now += int((parameter != active[name].cpu()).sum())
        total += parameter.numel()
    if not changed_now or count / total > max_changed_share:
        raise ValueError('No-op or cumulative parameter budget exceeded')
    if model_hash(current) != before:
        raise RuntimeError('Copy edit modified its source model')
    return candidate, {'operation': operation, 'before_sha256': before, 'candidate_sha256': model_hash(candidate),
                       'original_sha256': model_hash(original), 'changed_this_edit': changed_now,
                       'cumulative_changed_parameters': count, 'total_parameters': total, 'changed_share': count / total,
                       'source_unchanged': True}

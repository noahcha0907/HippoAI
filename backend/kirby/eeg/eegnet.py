"""EEGNet-8,2 in PyTorch, with explicit inference-time BN folding."""

import copy
import hashlib
import numpy as np
import torch
from torch import nn

from ..records import digest


class EEGNet(nn.Module):
    def __init__(self, *, folded=False, samples=1000):
        super().__init__()
        if samples != 1000:
            raise ValueError('This protocol fixes 250 Hz and 1000 samples')
        self.folded = folded
        self.temporal = nn.Conv2d(1, 8, (1, 125), padding=(0, 62), bias=folded)
        self.bn_temporal = nn.Identity() if folded else nn.BatchNorm2d(8, eps=0.001, momentum=0.01)
        self.spatial = nn.Conv2d(8, 16, (22, 1), groups=8, bias=folded)
        self.bn_spatial = nn.Identity() if folded else nn.BatchNorm2d(16, eps=0.001, momentum=0.01)
        self.depthwise = nn.Conv2d(16, 16, (1, 16), groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16, 1, bias=folded)
        self.bn_pointwise = nn.Identity() if folded else nn.BatchNorm2d(16, eps=0.001, momentum=0.01)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(16 * 31, 4)

    def features(self, x):
        x = x[:, None] if x.ndim == 3 else x
        x = self.bn_temporal(self.temporal(x))
        x = torch.nn.functional.elu(self.bn_spatial(self.spatial(x)))
        x = self.dropout(torch.nn.functional.avg_pool2d(x, (1, 4)))
        x = self.depthwise(torch.nn.functional.pad(x, (7, 8, 0, 0)))
        x = torch.nn.functional.elu(self.bn_pointwise(self.pointwise(x)))
        return self.dropout(torch.nn.functional.avg_pool2d(x, (1, 8))).flatten(1)

    def forward(self, x):
        return self.classifier(self.features(x))

    def constrain(self):
        with torch.no_grad():
            for weight, limit in ((self.spatial.weight, 1.0), (self.classifier.weight, 0.25)):
                norms = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
                weight.mul_((limit / norms).clamp(max=1).reshape(-1, *([1] * (weight.ndim - 1))))


def model_hash(model):
    return digest({k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                   for k, v in model.state_dict().items()})


def fold_model(model, calibration, tolerance=1e-5):
    if model.folded:
        raise ValueError('Already folded')
    original = copy.deepcopy(model).cpu().eval()
    folded = copy.deepcopy(original)
    for conv, bn in [('temporal', 'bn_temporal'), ('spatial', 'bn_spatial'), ('pointwise', 'bn_pointwise')]:
        setattr(folded, conv, torch.nn.utils.fusion.fuse_conv_bn_eval(getattr(folded, conv), getattr(folded, bn)))
        setattr(folded, bn, nn.Identity())
    folded.folded = True
    max_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(calibration), 32):
            x = torch.as_tensor(calibration[start:start + 32], dtype=torch.float32)
            a, b = original(x), folded(x)
            max_error = max(max_error, float((a - b).abs().max()))
            if not torch.allclose(a, b, atol=tolerance, rtol=tolerance) or not torch.equal(a.argmax(1), b.argmax(1)):
                raise ValueError('Folded EEGNet does not reproduce original logits/predictions')
    return folded, {'max_logit_error': max_error, 'atol': tolerance, 'rtol': tolerance,
                    'predictions_identical': True, 'calibration_trials': len(calibration)}


def save_model(model, path):
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def load_model(path, *, folded=True):
    model = EEGNet(folded=folded)
    with np.load(path, allow_pickle=False) as arrays:
        model.load_state_dict({k: torch.from_numpy(arrays[k].copy()) for k in arrays.files}, strict=True)
    return model.eval()

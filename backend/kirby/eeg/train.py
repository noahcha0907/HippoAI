"""Deterministic training; stopping uses only the internal training-validation split."""

import copy
import random
import numpy as np
import torch
from .eegnet import EEGNet
from .evaluate import predict


def fit(x, y, validation_x, validation_y, config, seed, *, initial=None, last_layer=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    model = copy.deepcopy(initial) if initial is not None else EEGNet()
    model.to(device)
    if last_layer:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('classifier.'))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    tx = torch.as_tensor(x, dtype=torch.float32); ty = torch.as_tensor(y, dtype=torch.long)
    vx = torch.as_tensor(validation_x, dtype=torch.float32, device=device)
    vy = torch.as_tensor(validation_y, dtype=torch.long, device=device)
    history, best, best_loss, stale = [], None, float('inf'), 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(config.epochs):
        model.train()
        if last_layer:
            model.eval()  # Freeze all features, including dropout/BN behavior.
        losses = []
        order = torch.randperm(len(tx), generator=generator)
        for indices in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(tx[indices].to(device)), ty[indices].to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            if not last_layer:
                model.constrain()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits = model(vx)
            vloss = float(torch.nn.functional.cross_entropy(logits, vy))
            accuracy = float((logits.argmax(1) == vy).float().mean())
        history.append({'epoch': epoch + 1, 'training_loss': float(np.mean(losses)), 'validation_loss': vloss, 'validation_accuracy': accuracy})
        if vloss < best_loss - 1e-6:
            best_loss, stale = vloss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if (epoch + 1) % 25 == 0:
            print(f'epoch {epoch + 1}: validation accuracy {accuracy:.3f}', flush=True)
        if stale >= config.patience:
            break
    if best is None:
        raise ValueError('No finite training checkpoint')
    model.load_state_dict(best)
    return model.cpu().eval(), history

"""MOABB download, explicit BNCI MAT parsing, train-fitted EOG regression."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from .config import CHANNELS
from ..records import file_hash, read_json, write_json


def read_session(path):
    """MAT trial positions are trial starts; imagery begins 2 seconds later."""
    runs = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)['data'])
    x, y, ids, flags = [], [], [], []
    for run_index, run in enumerate(runs):
        trials = np.atleast_1d(run.trial).astype(int).ravel()
        if not len(trials):
            continue
        labels = np.atleast_1d(run.y).astype(int).ravel() - 1
        artifact = np.atleast_1d(run.artifacts).astype(int).ravel()
        if float(run.fs) != 250 or run.X.shape[1] != 25 or len(labels) != len(trials) or len(artifact) != len(trials):
            raise ValueError('Unexpected BNCI2014_001 shape, sample rate or trial metadata')
        for i, start in enumerate(trials - 1 + 500):
            epoch = np.asarray(run.X[start:start + 1000], dtype=np.float64).T
            if epoch.shape != (25, 1000) or not np.isfinite(epoch).all() or not 0 <= labels[i] < 4:
                raise ValueError('Invalid EEG trial')
            x.append(epoch); y.append(labels[i]); ids.append(f'{Path(path).stem}:run{run_index}:trial{i}'); flags.append(int(artifact[i]))
    x = np.asarray(x)
    if len(x) != 288 or not np.array_equal(np.bincount(y, minlength=4), [72] * 4):
        raise ValueError('Expected 288 trials, 72 per class, in each BNCI session')
    # Independent epoch filters avoid mixing train/dev observations across boundaries.
    x = sosfiltfilt(butter(4, [0.5, 100], fs=250, btype='bandpass', output='sos'), x, axis=-1)
    return x, np.asarray(y, dtype=np.int64), np.asarray(ids), flags


def split_indices(y, seed):
    rng = np.random.default_rng(seed)
    groups = {'train': [], 'validation': [], 'development': []}
    for cls in range(4):
        indices = rng.permutation(np.flatnonzero(y == cls))
        dev = max(1, round(len(indices) * 0.2))
        val = max(1, round((len(indices) - dev) * 0.2))
        groups['development'].extend(indices[:dev])
        groups['validation'].extend(indices[dev:dev + val])
        groups['train'].extend(indices[dev + val:])
    return {k: np.sort(v) for k, v in groups.items()}


def fit_preprocessing(train):
    eeg = train[:, :22].transpose(0, 2, 1).reshape(-1, 22)
    eog = train[:, 22:].transpose(0, 2, 1).reshape(-1, 3)
    emean = eog.mean(0)
    centered = eog - emean
    covariance = centered.T @ centered
    ridge = max(float(np.trace(covariance)) / 3 * 1e-6, 1e-12)
    beta = np.linalg.solve(covariance + ridge * np.eye(3), centered.T @ (eeg - eeg.mean(0)))
    corrected = eeg - centered @ beta
    mean, std = corrected.mean(0), corrected.std(0)
    if np.any(std < 1e-8):
        raise ValueError('Degenerate EEG channel')
    return {'eog_mean': emean, 'beta': beta, 'mean': mean, 'std': std}


def transform(raw, fitted):
    eeg = raw[:, :22].transpose(0, 2, 1)
    eog = raw[:, 22:].transpose(0, 2, 1)
    corrected = eeg - (eog - fitted['eog_mean']) @ fitted['beta']
    return ((corrected - fitted['mean']) / fitted['std']).transpose(0, 2, 1).astype(np.float32)


def blink_bank(raw):
    """Partition-local high-amplitude EOG snippets; not clinically labeled blinks."""
    eog = sosfiltfilt(butter(4, [0.5, 4], fs=250, btype='bandpass', output='sos'), raw[:, 22:], axis=-1)
    result = []
    for trial in eog:
        channel = trial[np.argmax(np.std(trial, axis=1))]
        center = int(np.argmax(np.abs(channel[125:875]))) + 125
        wave = channel[center - 125:center + 125].copy()
        wave -= np.mean(wave)
        wave *= np.hanning(250)
        peak = np.max(np.abs(wave))
        if peak > 1e-8:
            result.append(wave / peak)
    if not result:
        raise ValueError('No usable EOG-derived waveforms')
    return np.asarray(result, dtype=np.float32)


def prepare_subject(subject, destination, seed=17, *, paths=None):
    import uuid
    destination = Path(destination)
    if destination.exists():
        return _prepare_subject(subject, destination, seed, paths=paths)
    pending = destination.with_name(destination.name + '.pending-' + uuid.uuid4().hex)
    manifest = _prepare_subject(subject, pending, seed, paths=paths)
    pending.rename(destination)
    return manifest


def _prepare_subject(subject, destination, seed=17, *, paths=None):
    destination = Path(destination)
    if (destination / 'manifest.json').is_file():
        manifest = read_json(destination / 'manifest.json')
        if manifest['subject'] != subject or manifest['split_seed'] != seed:
            raise ValueError('Existing EEG data use different settings')
        if manifest.get('preprocessing_source_sha256') != file_hash(Path(__file__)):
            raise ValueError('Prepared EEG data use different preprocessing code; use a new data-root')
        for name, expected in manifest['files'].items():
            if file_hash(destination / name) != expected:
                raise ValueError('Prepared EEG data changed')
        return manifest
    destination.mkdir(parents=True, exist_ok=False)
    if paths is None:
        from moabb.datasets import BNCI2014_001
        paths = BNCI2014_001().data_path(subject, update_path=False)
    paths = [Path(p) for p in paths]
    train_path = next(p for p in paths if p.name == f'A{subject:02d}T.mat')
    final_path = next(p for p in paths if p.name == f'A{subject:02d}E.mat')
    raw, y, ids, flags = read_session(train_path)
    splits = split_indices(y, seed)
    fitted = fit_preprocessing(raw[splits['train']])
    np.savez_compressed(destination / 'preprocessing.npz', **fitted)
    manifest = {'dataset': 'BNCI2014_001', 'subject': subject, 'split_seed': seed,
                'preprocessing_source_sha256': file_hash(Path(__file__)),
                'preparation_versions': {'numpy': np.__version__, 'scipy': __import__('scipy').__version__},
                'sampling_hz': 250, 'samples': 1000, 'channels': list(CHANNELS), 'units_before_standardization': 'microvolts',
                'sources': {p.name: file_hash(p) for p in paths}, 'splits': {},
                'processing': 'per-trial 0.5-100 Hz; EOG ridge regression and standardization fit only on train; no artifact rejection',
                'blink_source': 'partition-local low-frequency high-amplitude EOG snippets, not verified blink annotations'}
    for split, indices in splits.items():
        np.savez_compressed(destination / (split + '.npz'), x=transform(raw[indices], fitted), y=y[indices],
                            ids=ids[indices], blinks=blink_bank(raw[indices]))
        manifest['splits'][split] = {'ids': ids[indices].tolist(), 'trials': len(indices),
                                    'artifact_flags': [flags[i] for i in indices]}
    # Preparation may read the final recording; training/auditing cannot load its arrays.
    raw_final, y_final, ids_final, final_flags = read_session(final_path)
    np.savez_compressed(destination / 'final.npz', x=transform(raw_final, fitted), y=y_final,
                        ids=ids_final, blinks=blink_bank(raw_final))
    manifest['splits']['final'] = {'ids': ids_final.tolist(), 'trials': len(ids_final), 'artifact_flags': final_flags}
    all_ids = [i for split in manifest['splits'].values() for i in split['ids']]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Trial leakage between partitions')
    manifest['files'] = {p.name: file_hash(p) for p in destination.glob('*.npz')}
    write_json(destination / 'manifest.json', manifest)
    return manifest


def load_split(directory, split, *, final_authorized=False):
    if split not in ('train', 'validation', 'development', 'final') or (split == 'final' and not final_authorized):
        raise ValueError('Final EEG data are locked until all methods are selected')
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    path = directory / (split + '.npz')
    if file_hash(path) != manifest['files'][path.name]:
        raise ValueError('EEG split integrity mismatch')
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}

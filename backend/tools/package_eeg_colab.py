"""Package only explicit source files for the new EEG Colab entry point."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    backend = Path(__file__).resolve().parents[1]
    files = [backend / 'pyproject.toml', backend / 'README.md', backend / 'notebooks/kirby_eeg_colab.ipynb']
    files += sorted((backend / 'kirby').rglob('*.py'))
    files += sorted((backend / 'tests').glob('test_eeg*.py'))
    files += [backend / 'docs' / name for name in ('EEG_EXPERIMENT.md', 'EEG_ROADMAP.md')]
    files += [backend / 'tools' / name for name in ('build_eeg_notebook.py', 'package_eeg_colab.py')]
    files += [backend / 'tools/build_eeg_replication.py', backend / 'experiments/eeg_replication_v1.json',
              backend / 'notebooks/kirby_eeg_replication_colab.ipynb']
    destination = backend / 'dist/kirby_eeg_colab.zip'
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError('Symlink in source bundle')
            archive.write(path, 'backend/' + path.relative_to(backend).as_posix())
    print(f'Created {destination}: {len(files)} source files, {destination.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()

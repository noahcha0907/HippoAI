"""Build a small, explicit source bundle; never include credentials or model data."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    backend = Path(__file__).resolve().parents[1]
    sources = [backend / "pyproject.toml", backend / "README.md"]
    for directory, pattern in (("kirby", "*.py"), ("tests", "*.py"), ("docs", "*.md"), ("notebooks", "*.ipynb")):
        sources.extend(sorted((backend / directory).glob(pattern)))
    destination = backend / "dist" / "kirby_colab.zip"
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for source in sources:
            if source.is_symlink():
                raise ValueError(f"Refusing a symlink in the source bundle: {source.name}")
            archive.write(source, "backend/" + source.relative_to(backend).as_posix())
    print(f"Created {destination} ({len(sources)} source files, {destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

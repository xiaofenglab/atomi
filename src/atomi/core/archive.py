from __future__ import annotations

import tarfile
from pathlib import Path


def default_archive_path(outdir: Path) -> Path:
    """Return the default sibling tar.gz path for an output directory."""
    outdir = Path(outdir)
    return outdir.with_name(f"{outdir.name}.tar.gz")


def archive_output_dir(outdir: Path, archive_path: Path | None = None) -> Path:
    """Write a gzip-compressed tar archive of an output directory.

    The archive keeps the output directory itself as the top-level member, so
    extracting ``analysis/foo.tar.gz`` recreates ``foo/...`` rather than spilling
    files into the current folder.
    """
    outdir = Path(outdir).resolve()
    if not outdir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {outdir}")

    archive = Path(archive_path).resolve() if archive_path else default_archive_path(outdir).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = archive.with_name(f".{archive.name}.tmp")
    if tmp_archive.exists():
        tmp_archive.unlink()

    exclude_paths = {archive, tmp_archive}

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        try:
            member_path = Path(info.name)
            source_path = outdir.parent / member_path
            if source_path.resolve() in exclude_paths:
                return None
        except OSError:
            pass
        return info

    with tarfile.open(tmp_archive, "w:gz") as handle:
        handle.add(outdir, arcname=outdir.name, filter=_filter)
    tmp_archive.replace(archive)
    return archive

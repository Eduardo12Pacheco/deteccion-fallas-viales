"""``rdv data`` commands: download, verify, prepare, and stats (S8/D14)."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import zipfile
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rdv import composition
from rdv.adapters.datasets.remote_zip import CountryPin, DatasetLock, RemoteZipArchive
from rdv.application.prepare_dataset import prepare_dataset
from rdv.config.settings import Settings
from rdv.errors import DatasetError, RdvError

DEFAULT_COUNTRIES = ("Japan", "India", "Czech")
DEFAULT_LOCK = Path("data/rdd2022.lock.json")
DEFAULT_SPLITS_LOCK = Path("data/splits.lock.json")
PROGRESS_STEP = 128 * 1024 * 1024
NORWAY_SEED = 42

data_app = typer.Typer(
    name="data",
    help="Download, prepare, and inspect the RDD2022 dataset.",
    no_args_is_help=True,
)


def _run(action: Callable[[], None]) -> None:
    try:
        action()
    except RdvError as exc:
        typer.echo(f"error: {exc.message}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc


def _settings(data_dir: Path | None) -> Settings:
    if data_dir is None:
        return Settings.from_env()
    return Settings.from_env(data_dir=data_dir)


def _data_root(settings: Settings) -> Path:
    return settings.data_dir / "rdd2022"


def _downloads(settings: Settings) -> Path:
    return settings.data_dir / "downloads"


def _progress(label: str) -> Callable[[int, int], None]:
    reported = 0

    def callback(done: int, total: int) -> None:
        nonlocal reported
        if done == total or done - reported >= PROGRESS_STEP:
            percent = 100.0 if total == 0 else done * 100.0 / total
            typer.echo(f"{label}: {done / (1024 * 1024):.0f} MiB ({percent:.0f}%)")
            reported = done

    return callback


def _verify_archive(path: Path, pin: CountryPin) -> tuple[int, str]:
    if not path.is_file():
        raise DatasetError(f"Missing downloaded archive: {path}")
    if path.stat().st_size != pin.size:
        raise DatasetError(
            f"Size mismatch for {path}: expected {pin.size}, got {path.stat().st_size}"
        )
    crc = 0
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
            sha256.update(chunk)
    crc &= 0xFFFFFFFF
    if crc != pin.crc32:
        raise DatasetError(f"CRC32 mismatch for {path}: expected {pin.crc32:08x}, got {crc:08x}")
    digest = sha256.hexdigest()
    if pin.sha256 is not None and digest != pin.sha256:
        raise DatasetError(f"sha256 mismatch for {path}: expected {pin.sha256}, got {digest}")
    return crc, digest


def _safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise DatasetError(f"Archive member escapes the dataset directory: {name!r}")
    return target


def _extract_country(archive_path: Path, root: Path, country: str) -> None:
    marker = root / country / ".rdv-extracted"
    if marker.is_file():
        return
    with zipfile.ZipFile(archive_path) as zipped:
        for info in zipped.infolist():
            if info.is_dir():
                continue
            if not info.filename.startswith(f"{country}/"):
                raise DatasetError(f"Unexpected member in the {country} archive: {info.filename!r}")
            # The official test folders have no public annotations; only the
            # labelled train tree is extracted and used by this project.
            if not info.filename.startswith(f"{country}/train/"):
                continue
            target = _safe_target(root, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(archive_path.name + "\n", encoding="utf-8")


def _extract_norway(archive: RemoteZipArchive, pin: CountryPin, root: Path, count: int) -> int:
    if count <= 0:
        return 0
    nested = archive.open_nested(pin)
    with zipfile.ZipFile(nested) as zipped:
        names = sorted(
            name
            for name in zipped.namelist()
            if name.startswith("Norway/train/images/") and name.lower().endswith(".jpg")
        )
        if count > len(names):
            raise DatasetError(
                f"Requested {count} Norway images but the archive has {len(names)} train images"
            )
        random.Random(NORWAY_SEED).shuffle(names)
        selected = names[:count]
        for index, image_name in enumerate(selected, start=1):
            xml_name = (
                image_name.replace("/images/", "/annotations/xmls/").rsplit(".", 1)[0] + ".xml"
            )
            for member in (image_name, xml_name):
                target = _safe_target(root, member)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    info = zipped.getinfo(member)
                except KeyError as exc:
                    raise DatasetError(f"Norway archive member not found: {member}") from exc
                # Read the exact compressed range instead of a 4 MiB cache
                # block: the sample picks random members from a 10.6 GB ZIP.
                nested.seek(info.header_offset)
                header = nested.read(30)
                if len(header) < 30 or header[:4] != b"PK\x03\x04":
                    raise DatasetError(f"Corrupt Norway archive header for {member}")
                name_length = int.from_bytes(header[26:28], "little")
                extra_length = int.from_bytes(header[28:30], "little")
                nested.seek(info.header_offset + 30 + name_length + extra_length)
                raw = nested.read(info.compress_size)
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    payload = zlib.decompress(raw, -15)
                elif info.compress_type == zipfile.ZIP_STORED:
                    payload = raw
                else:
                    raise DatasetError(
                        f"Unsupported compression for Norway member {member}: {info.compress_type}"
                    )
                if zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
                    raise DatasetError(f"CRC32 mismatch for Norway member {member}")
                target.write_bytes(payload)
            if index % 100 == 0 or index == count:
                typer.echo(f"Norway: extracted {index}/{count} images")
    return count


def _download_country(
    archive: RemoteZipArchive,
    lock: DatasetLock,
    lock_path: Path,
    settings: Settings,
    country: str,
) -> DatasetLock:
    pin = lock.country(country)
    destination = _downloads(settings) / f"{country}.zip"
    if destination.is_file():
        try:
            _verify_archive(destination, pin)
            typer.echo(f"{country}: already downloaded and verified")
            return lock
        except DatasetError:
            destination.unlink()
    data_offset = archive.local_header_data_offset(pin.header_offset)
    result = archive.download_to(
        destination,
        data_offset=data_offset,
        size=pin.size,
        expected_crc32=pin.crc32,
        progress=_progress(f"{country}"),
    )
    if pin.sha256 is not None and result.sha256 != pin.sha256:
        destination.unlink(missing_ok=True)
        raise DatasetError(
            f"sha256 mismatch for {destination}: expected {pin.sha256}, got {result.sha256}"
        )
    if pin.sha256 is None:
        lock = lock.with_sha256(country, result.sha256)
        lock.write(lock_path)
        typer.echo(f"{country}: recorded sha256 {result.sha256}")
    return lock


def _download_outer(archive: RemoteZipArchive, lock: DatasetLock, settings: Settings) -> Path:
    destination = _downloads(settings) / "RDD2022_outer.zip"
    if destination.is_file():
        md5 = hashlib.md5(usedforsecurity=False)
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                md5.update(chunk)
        if md5.hexdigest() == lock.outer_md5 and destination.stat().st_size == lock.outer_size:
            typer.echo("Outer archive: already downloaded and verified")
            return destination
        destination.unlink()
    archive.download_to(
        destination,
        data_offset=0,
        size=lock.outer_size,
        expected_md5=lock.outer_md5,
        progress=_progress("RDD2022 outer"),
    )
    return destination


def _extract_inner_from_outer(outer: Path, pin: CountryPin, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    with (
        zipfile.ZipFile(outer) as zipped,
        zipped.open(pin.member) as source,
        part.open("wb") as target,
    ):
        shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
    _verify_archive(part, pin)
    part.replace(destination)


@data_app.command("download")
def download(
    countries: Annotated[
        list[str] | None,
        typer.Argument(help="Countries to download; defaults to Japan India Czech."),
    ] = None,
    countries_flag: Annotated[
        bool, typer.Option("--countries", help="Marker enabling the documented --countries syntax.")
    ] = False,
    ood_norway: Annotated[
        int, typer.Option("--ood-norway", min=0, help="Extract seeded Norway train images.")
    ] = 0,
    full: Annotated[
        bool, typer.Option("--full", help="Download the 13 GB outer ZIP and verify its MD5.")
    ] = False,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Dataset directory override.")
    ] = None,
    lock: Annotated[Path, typer.Option("--lock", help="Pinned archive metadata.")] = DEFAULT_LOCK,
) -> None:
    """Download only the required inner archives with HTTP ranges (S8)."""

    def action() -> None:
        settings = _settings(data_dir)
        dataset_lock = composition.load_dataset_lock(lock)
        selected = tuple(countries) if countries else DEFAULT_COUNTRIES
        for country in selected:
            dataset_lock.country(country)
        root = _data_root(settings)
        _downloads(settings).mkdir(parents=True, exist_ok=True)
        if full:
            full_archive = composition.open_dataset_archive(dataset_lock, offline=settings.offline)
            try:
                outer = _download_outer(full_archive, dataset_lock, settings)
            finally:
                full_archive.close()
            for country in selected:
                pin = dataset_lock.country(country)
                destination = _downloads(settings) / f"{country}.zip"
                if not destination.is_file():
                    _extract_inner_from_outer(outer, pin, destination)
                _extract_country(destination, root, country)
        else:
            archive = composition.open_dataset_archive(dataset_lock, offline=settings.offline)
            with archive:
                for country in selected:
                    dataset_lock = _download_country(archive, dataset_lock, lock, settings, country)
                    _extract_country(_downloads(settings) / f"{country}.zip", root, country)
        if ood_norway:
            # Norway images are ~1 MB each and selected by a seeded shuffle, so
            # a 4 MiB block would fetch four times the needed bytes per image.
            archive = composition.open_dataset_archive(
                dataset_lock, offline=settings.offline, block_size=64 * 1024
            )
            with archive:
                _extract_norway(archive, dataset_lock.country("Norway"), root, ood_norway)
        typer.echo(f"Dataset ready under {root}")

    _run(action)


@data_app.command("verify")
def verify(
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Dataset directory override.")
    ] = None,
    lock: Annotated[Path, typer.Option("--lock", help="Pinned archive metadata.")] = DEFAULT_LOCK,
    full: Annotated[
        bool, typer.Option("--full", help="Also verify the downloaded outer ZIP MD5.")
    ] = False,
) -> None:
    """Verify downloaded archives against the pinned size, CRC32 and sha256."""

    def action() -> None:
        settings = _settings(data_dir)
        dataset_lock = composition.load_dataset_lock(lock)
        verified = 0
        for name, pin in dataset_lock.countries.items():
            path = _downloads(settings) / f"{name}.zip"
            if not path.is_file():
                continue
            crc, sha256 = _verify_archive(path, pin)
            typer.echo(f"ok {name}: {pin.size} bytes, crc32 {crc:08x}, sha256 {sha256}")
            verified += 1
        outer = _downloads(settings) / "RDD2022_outer.zip"
        if full:
            if not outer.is_file():
                raise DatasetError(f"Outer archive not downloaded: {outer}")
            md5 = hashlib.md5(usedforsecurity=False)
            with outer.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    md5.update(chunk)
            if md5.hexdigest() != dataset_lock.outer_md5:
                raise DatasetError(f"MD5 mismatch for {outer}: expected {dataset_lock.outer_md5}")
            typer.echo(f"ok outer: {dataset_lock.outer_size} bytes, md5 {dataset_lock.outer_md5}")
            verified += 1
        if verified == 0:
            raise DatasetError(f"No downloaded archives found under {_downloads(settings)}")
        typer.echo(f"Verified {verified} archive(s)")

    _run(action)


@data_app.command("prepare")
def prepare(
    countries: Annotated[
        list[str] | None, typer.Argument(help="Countries to split; defaults to Japan India Czech.")
    ] = None,
    countries_flag: Annotated[
        bool, typer.Option("--countries", help="Marker enabling the documented --countries syntax.")
    ] = False,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Dataset directory override.")
    ] = None,
    lock: Annotated[
        Path, typer.Option("--lock", help="Pinned split-manifest sha256 lock.")
    ] = DEFAULT_SPLITS_LOCK,
    seed: Annotated[int, typer.Option("--seed", help="Block-shuffle seed.")] = 42,
    block_size: Annotated[
        int, typer.Option("--block-size", help="Consecutive ids per block.")
    ] = 100,
) -> None:
    """Build the seeded block split and verify its pinned sha256 (D3)."""

    def action() -> None:
        settings = _settings(data_dir)
        selected: Sequence[str] = tuple(countries) if countries else DEFAULT_COUNTRIES
        result = prepare_dataset(
            _data_root(settings),
            countries=selected,
            lock_path=lock,
            seed=seed,
            block_size=block_size,
        )
        typer.echo(
            "Split sizes: "
            + ", ".join(f"{name} {count}" for name, count in result.counts.items())
            + f" (total {sum(result.counts.values())})"
        )
        typer.echo(f"Manifest: {result.manifest_path}")
        typer.echo(f"sha256: {result.sha256}")
        if result.lock_written:
            typer.echo(f"Wrote split lock {result.lock_path}")

    _run(action)


@data_app.command("stats")
def stats(
    countries: Annotated[
        list[str] | None, typer.Argument(help="Countries to scan; defaults to all.")
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Dataset directory override.")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print machine-readable JSON.")
    ] = False,
) -> None:
    """Print per-country image, label and dropped-label counts."""

    def action() -> None:
        settings = _settings(data_dir)
        scan = composition.scan_dataset(
            _data_root(settings), tuple(countries) if countries else None
        )
        if json_output:
            typer.echo(json.dumps(scan.as_dict(), indent=2))
            return
        table = Table(title="RDD2022 dataset")
        for column in ("Country", "Train", "Test", "Labels", "D00", "D10", "D20", "D40", "Dropped"):
            table.add_column(column, justify="right" if column != "Country" else "left")
        for country, entry in scan.countries.items():
            table.add_row(
                country,
                str(entry.images.get("train", 0)),
                str(entry.images.get("test", 0)),
                str(entry.labels),
                *(str(entry.classes.get(code, 0)) for code in ("D00", "D10", "D20", "D40")),
                str(sum(entry.dropped_labels.values())),
            )
        Console().print(table)

    _run(action)

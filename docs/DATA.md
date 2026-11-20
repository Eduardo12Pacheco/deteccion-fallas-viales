# Data: RDD2022 sources, licenses, checksums and lineage

Everything documented here is reproducible with the commands in this file. No
dataset bytes are committed to git: the mini fixture and the six packaged
samples are the only redistributed images (see [Fixtures](#fixtures-and-samples)).

## Source

| Field | Value |
|---|---|
| Dataset | RDD2022 - The multi-national Road Damage Dataset, released through CRDDC'2022 |
| Figshare article | `21431547` (DOI `10.6084/m9.figshare.21431547.v1`) |
| Download URL | `https://ndownloader.figshare.com/files/38030910` |
| Outer file | `RDD2022_released_through_CRDDC2022.zip` |
| Outer size | 13,264,172,619 bytes |
| Outer md5 | `b62bd51d2ffcfaa76c60f234f0cc2bb3` |
| Paper | RDD2022: A multi-national image dataset for automatic road damage detection, arXiv:2209.08538 |
| Classes | `D00` longitudinal crack, `D10` transverse crack, `D20` alligator crack, `D40` pothole |

The outer ZIP contains one stored (uncompressed) inner ZIP per country. The
project downloads only the inner archives it needs with HTTP range requests;
the 10.6 GB Norway archive is sampled through nested ranges without downloading
it. See `docs/adr/0009-ranged-nested-zip.md`.

### Pinned inner archives

`data/rdd2022.lock.json` is the single source of truth for the offsets and
checksums. `header_offset` is the byte position of the inner ZIP's local header
in the outer file; the adapter parses that header to find the first data byte,
so a Zip64 extra field cannot silently shift the range.

| Country | Outer member | Header offset | Size (bytes) | CRC32 | sha256 |
|---|---|---:|---:|---|---|
| Japan | `RDD2022/Japan.zip` | 1,136,050,224 | 1,072,598,857 | `e133e2cc` | `76efe7acb4642da283891611670dac7e8866a1b1dd145b8a1540fd440b1328f7` |
| India | `RDD2022/India.zip` | 609,341,032 | 526,709,145 | `f186f6f8` | `28eab0aa85638855e1907e63b175be2305a0accd018c54603037d3fb8cab963a` |
| Czech | `RDD2022/Czech.zip` | 352,223,767 | 257,117,218 | `abcf66aa` | `94c65a4e6cfed187b193ba967f0a2b16bd4287dfe313d0962f3711afce75bf1d` |
| Norway (OOD sample only) | `RDD2022/Norway.zip` | 2,208,649,128 | 10,611,186,814 | `01bedb71` | recorded on first download |
| United_States (excluded) | `RDD2022/United_States.zip` | 12,819,836,010 | 444,335,958 | `fb8ded4b` | not downloaded |
| China_Drone (excluded) | `RDD2022/China_Drone.zip` | 0 | 160,193,541 | `28b57b73` | not downloaded |
| China_MotorBike (excluded) | `RDD2022/China_MotorBike.zip` | 160,193,594 | 192,030,116 | `f887e542` | not downloaded |

`rdv data download` writes `<data_dir>/downloads/<Country>.zip.part`, resumes a
truncated part, verifies the exact size and CRC32, and only then renames it to
`<Country>.zip`. The first success records the inner archive sha256 in the lock
and later runs verify it. A CRC32 mismatch removes the part and exits 3.

## License stance

- **Figshare record:** CC BY 4.0 (the `license` field of the Figshare API).
- **Conflicting evidence:** the arXiv paper page and a Hugging Face mirror state
  CC BY-SA 4.0. That is the license of the paper; the mirror may have copied it.
- **Repository decision (D2):** follow the Figshare record for the data,
  attribute the authors, and redistribute derivative fixtures/samples under
  CC BY-SA 4.0, which satisfies both readings. The MIT license of this
  repository covers the code only.
- **Models trained on the data:** the legal status of model weights derived from
  CC-licensed images is unsettled; the model card states this explicitly rather
  than claiming a clean license.
- **United_States / Google Street View:** excluded from training and not
  redistributed because the Street View imagery has separate terms that the
  dataset license cannot settle.
- **Test annotations:** the RDD2022 paper states that annotations for the test
  images were not released. The official `test/` folders are images only, so
  this project evaluates on its own held-out split (see below) and never claims
  scores against the official challenge test set.

## Download and layout

```bash
export RDV_DATA_DIR=data          # keep ~2 GB of data out of git and ~/.local
uv run rdv data download --countries Japan India Czech
uv run rdv data verify
uv run rdv data stats
uv run rdv data prepare
# Optional out-of-domain sample (ranged reads into the 10.6 GB Norway ZIP):
uv run rdv data download --ood-norway 1000
```

The resulting tree is:

```
<data_dir>/
├── downloads/                 # verified inner ZIPs (+ optional outer ZIP)
└── rdd2022/
    ├── splits.json            # rdv.splits/v1
    ├── Japan/train/{images,annotations/xmls}/
    ├── India/train/{images,annotations/xmls}/
    ├── Czech/train/{images,annotations/xmls}/
    └── Norway/train/{images,annotations/xmls}/   # 1,000-image OOD sample
```

`RDV_OFFLINE=1` makes every command fail before the first network request.
`rdv data download --full` downloads the 13 GB outer ZIP, verifies its md5 and
extracts the requested inner archives locally; it is the fallback if Figshare
range requests ever stop working.

## Splits (D3)

The labelled train folders are split per country with a seeded block split:

- ids are sorted, cut into blocks of 100 consecutive ids;
- blocks are shuffled with `random.Random(42)`;
- blocks are assigned greedily to the 80/10/10 targets;
- the combined manifest is written to `<data_dir>/rdd2022/splits.json`
  (schema `rdv.splits/v1`) and its sha256 is pinned in `data/splits.lock.json`.

| Split | Czech | India | Japan | Images |
|---|---:|---:|---:|---:|
| train | 2,229 | 6,106 | 8,406 | 16,741 |
| val | 300 | 800 | 1,100 | 2,200 |
| test | 300 | 800 | 1,000 | 2,100 |
| **total** | **2,829** | **7,706** | **10,506** | **21,041** |

Manifest sha256: `dbc3eaed76cf2c85e7b3203ebd9248e66daa0caf5fe0b5c7319d76df917e0b2f`
(pinned in `data/splits.lock.json`).

`rdv data prepare` recomputes the manifest from the downloaded images (not from
a previous manifest) and refuses a digest that does not match the lock. The
lock is written only when absent; deleting it is the explicit way to accept a
new split.

## Measured statistics

Command: `RDV_DATA_DIR=data uv run rdv data stats` (verified 2026-11-20 on the
owner machine; the three archives matched their pinned size, CRC32 and sha256).

| Country | Train images | Labels | D00 | D10 | D20 | D40 | Dropped |
|---|---:|---:|---:|---:|---:|---:|---:|
| Czech | 2,829 | 1,745 | 988 | 399 | 161 | 197 | 0 |
| India | 7,706 | 6,831 | 1,555 | 68 | 2,021 | 3,187 | 1,372 |
| Japan | 10,506 | 16,469 | 4,049 | 3,979 | 6,198 | 2,243 | 8,284 |
| Norway (OOD sample) | 1,000 | 1,193 | 921 | 185 | 45 | 42 | 0 |
| **Total** | **22,041** | **26,238** | **7,513** | **4,631** | **8,425** | **5,669** | **9,656** |

The in-domain image counts match the published per-country train counts exactly.
The Norway row is the 1,000-image OOD sample (seeded shuffle, never trained on),
read through nested ranges from the 10.6 GB archive and labelled with the same
VOC parser. Labels outside the four supported classes are dropped and counted
instead of being silently mapped:

| Raw label | Count | Note |
|---|---:|---|
| `D44` | 5,057 | Additional RDD2022 crack subtype, outside this project's four classes |
| `D50` | 3,581 | Additional RDD2022 subtype, outside this project's four classes |
| `D43` | 793 | Additional RDD2022 subtype, outside this project's four classes |
| `D01` | 179 | Longitudinal-crack variant not in the D4 class list |
| `D11` | 45 | Transverse-crack variant not in the D4 class list |
| `D0w0` | 1 | Malformed label in the source annotations |

The parser also drops ground-truth boxes that collapse to zero area after
clamping; those are counted internally by the VOC parser but are not part of
`rdv data stats`. Empty XMLs are kept as negative images.

## Fixtures and samples

`scripts/make_fixtures.py` implements D15(b)(c) with seed 42:

- 3 positive images per class from Japan/India/Czech (12 total) plus 4 negatives
  (images whose XML has no object);
- JPEG resized to a 640 px long side (quality 85);
- VOC XML rewritten with the new dimensions and scaled boxes, one annotation
  per surviving ground-truth box;
- `tests/fixtures/rdd2022-mini/splits.json` uses a 4-image block split
  (ratios 0.5/0.25/0.25) so all three partitions are non-empty;
- six samples are copied from the fixture (first of each class, second D40,
  second D00) into `src/rdv/assets/samples/`;
- `data/fixtures.lock.json` records the sha256 of every produced file.

Both directories ship `LICENSE` (CC BY-SA 4.0 notice) and `ATTRIBUTION.md`
(source image ids, DOI, paper and license discussion). The mini fixture is
intended for tests, docs and the bundled samples only; train on the full
dataset.

Regenerate and check:

```bash
uv run python scripts/make_fixtures.py
git diff --stat          # must be empty for byte-identical fixtures
du -sh tests/fixtures/rdd2022-mini
```

## Lineage

```
Figshare outer ZIP (pinned md5/size)
  └─ ranged inner ZIP Japan/India/Czech (pinned CRC32, recorded sha256)
       └─ extract <Country>/train/{images,annotations/xmls}
            ├─ seeded 80/10/10 block split → splits.json (sha256 pinned)
            ├─ VOC parse → rdv data stats (counts, dropped labels)
            └─ training / evaluation (T9, T10)
  └─ ranged nested Norway ZIP → 1,000 seeded train images (OOD only)
  └─ scripts/make_fixtures.py → mini fixture + samples + fixtures.lock.json
```

Labels outside `D00/D10/D20/D40` are dropped and counted by the parser;
`rdv data stats` reports the raw label text so a change in the source is
visible. Images whose XML has no object are kept as negatives, not discarded.

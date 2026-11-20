# 0009 - Ranged access to the nested Figshare archive

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D14, S8)
- **Task:** T5

## Context

The RDD2022 download is a single 13,264,172,619-byte ZIP
(`https://ndownloader.figshare.com/files/38030910`, md5
`b62bd51d2ffcfaa76c60f234f0cc2bb3`) that contains one ZIP per country. The
three training countries need about 1.9 GB; the Norway archive alone is
10.6 GB and is only used for a 1,000-image out-of-domain sample. Downloading
13 GB to read 1.9 GB is wasteful, and downloading 10.6 GB to read 1,000 images
is unacceptable on a laptop.

The download endpoint answers with a 302 to a presigned S3/R2 URL that expires
after ten seconds (`X-Amz-Expires=10`). A naive ranged downloader therefore
fails mid-transfer unless it re-resolves the URL.

## Decision

Implement `RemoteZipArchive` in `rdv.adapters.datasets.remote_zip` as a
seekable byte-range reader over the outer ZIP:

- A **4 MiB block cache** (LRU, 64 blocks) backs small seeks such as the
  `zipfile` central-directory parser; reads of at least one block bypass the
  cache and fetch the exact range, so streaming a large member does not
  multiply traffic by the block size.
- The outer members are **stored uncompressed**, so an inner archive is a
  contiguous outer range. `open_nested()` parses the outer central directory,
  reads the local header to compute the first data byte, and returns a seekable
  window that `zipfile.ZipFile` can read. The 1.67 MB Norway central directory
  and the selected image/XML ranges are the only bytes fetched.
- The stable endpoint is **re-resolved on a 5 s TTL** and whenever a ranged GET
  returns 403. Directions: retry 403/429/5xx up to five times with exponential
  backoff; any other status is a `DatasetError` (exit 3).
- `RDV_OFFLINE=1` raises at construction, before the first request (S20).
- Downloads write `<Country>.zip.part`, resume from the existing part size,
  verify exact size, CRC32 and (when pinned) sha256, and only then rename to the
  final path. A failed check removes the part.
- `data/rdd2022.lock.json` pins the outer md5/size and, per country, the outer
  member name, local-header offset, size and CRC32. The first successful
  download records the inner archive sha256 in the same lock, making later
  verifications stronger without changing the acquisition path.
- `rdv data download --full` is the fallback: download the outer ZIP, verify
  its md5, and extract the needed inner members locally.

## Consequences

- A normal setup downloads roughly 1.9 GB instead of 13 GB, and the Norway
  sample costs roughly the size of the selected JPEGs plus range overhead.
- Zip64 members (Norway and United_States are larger than 4 GiB or sit past
  the 4 GiB offset) are handled by the standard `zipfile` parser; the lock
  stores the local-header offset, not a computed data offset, so a different
  extra-field length cannot silently shift the download.
- Tests use `httpx.MockTransport` with a synthetic nested ZIP, so no CI job
  touches Figshare and the resume/CRC/expiry paths are exercised without
  network access.
- The adapter lives in the driven-adapter layer and is constructed only by
  `composition.py`; the CLI never imports `httpx` or builds the archive itself.

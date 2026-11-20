# 0002 - Dataset subset, licensing stance and own splits

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D2, D3, D4, D15)
- **Task:** T5

## Context

RDD2022 (released through CRDDC'2022, Figshare article 21431547) is a
multi-national road-damage dataset with four classes (`D00` longitudinal crack,
`D10` transverse crack, `D20` alligator crack, `D40` pothole). It is large
(47,420 images in seven country folders) and heterogeneous: sensor, resolution
and capture viewpoint differ per country, and the test annotations were never
released.

The Figshare record declares **CC BY 4.0**. The arXiv paper page and some
mirrors state **CC BY-SA 4.0** (the license of the paper itself), and the
United States images come from Google Street View, whose terms are separate
from the dataset license. Redistributing a small image subset therefore needs
an explicit, conservative policy.

## Decision

- **Train on Japan, India and Czech only** (21,041 labelled train images:
  10,506 / 7,706 / 2,829). All three are road-level smartphone/dashcam captures
  with compatible viewpoints. China_Drone and China_MotorBike use different
  sensors and are future work.
- **Never train on United_States** because the Google Street View imagery has
  third-party rights that the Figshare license cannot settle. The folder is
  still pinned in `data/rdd2022.lock.json` for provenance but is not downloaded
  by default.
- **Use Norway only as an out-of-domain evaluation sample** (1,000 seeded train
  images, never trained on). Norway is ~3650x2044 and is a useful domain-shift
  probe.
- **License derivative fixtures as CC BY-SA 4.0.** The repository code stays
  MIT; `tests/fixtures/rdd2022-mini` and `src/rdv/assets/samples` carry a
  CC BY-SA 4.0 notice and an `ATTRIBUTION.md` naming the source images, DOI and
  paper. This satisfies both readings of the dataset license. The still-open
  legal question about models trained on CC data is stated in the model card
  (T10/T16).
- **Create our own splits.** The public test annotations do not exist, so using
  the official test folders would leave nothing to measure. Instead, sort the
  train ids per country, cut blocks of 100 consecutive ids, shuffle the blocks
  with `random.Random(42)`, and assign blocks greedily to 80/10/10. Blocks keep
  near-duplicate consecutive captures in the same partition as much as possible;
  residual leakage is a documented limitation (D3).
- **Keep empty annotations as negative images** (about 70% of Czech XMLs have
  no object), because false positives on intact road matter for a detector.

## Consequences

- The training set is homogeneous enough to fit one model while remaining
  multi-country, and every reported metric is measured on data the model never
  saw.
- The mini fixture and samples are legally redistributable with attribution,
  but they are a tiny subset: users must download the full dataset for training.
- The Norway number is an OOD signal, not part of the in-domain target metrics.
- Splits are reproducible from the downloaded tree and pinned by sha256, so a
  changed split is visible in review rather than silently altering results.

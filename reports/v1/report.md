# Evaluation report

- Model: `rdv-rtdetrv2-r18-rdd2022-v1` (device `cpu`)
- Split: `test`
- Images: 2100
- Ground-truth boxes: 2469
- Operating threshold: 0.0216 (val)
- mAP@0.5: 0.3794
- mAP@0.5:0.95: 0.1696

## Metrics per class

| Class | Name | Boxes | AP@0.5 | AP@0.5:0.95 |
|-------|------|-------|--------|-------------|
| D00 | Longitudinal crack | 649 | 0.3383 | 0.1434 |
| D10 | Transverse crack | 438 | 0.3089 | 0.1196 |
| D20 | Alligator crack | 797 | 0.4955 | 0.2527 |
| D40 | Pothole | 585 | 0.3749 | 0.1626 |

## Operating point

| Threshold | Precision | Recall | F1 | TP | FP |
|-----------|-----------|--------|----|----|----|
| 0.0216 | 0.4249 | 0.4484 | 0.4363 | 1172 | 1586 |

## Per country

| Country | Images | Boxes | mAP@0.5 | mAP@0.5:0.95 |
|---------|--------|-------|---------|--------------|
| Czech | 300 | 189 | 0.2780 | 0.1014 |
| India | 800 | 703 | 0.2666 | 0.1189 |
| Japan | 1000 | 1577 | 0.4261 | 0.1994 |

## Out of domain

| Images | Boxes | mAP@0.5 | mAP@0.5:0.95 |
|--------|-------|---------|--------------|
| 1000 | 1193 | 0.0588 | 0.0229 |

## Error taxonomy

| Category | Count |
|----------|-------|
| classification | 62 |
| localization | 378 |
| duplicate | 121 |
| background | 760 |
| missed | 1441 |
| **false positives** | 1321 |
| **missed** | 1441 |

## Confusion matrix

| Truth \ prediction | D00 | D10 | D20 | D40 | background |
|---|---|---|---|---|---|
| D00 | 241 | 2 | 11 | 3 | 392 |
| D10 | 4 | 184 | 1 | 1 | 248 |
| D20 | 15 | 3 | 319 | 4 | 456 |
| D40 | 0 | 0 | 1 | 276 | 308 |
| background | 304 | 337 | 168 | 475 | 0 |

## Artifacts

- [Precision-recall curves](pr_curves.png)
- [Confusion matrix](confusion_matrix.png)
- [Error analysis](error_analysis.json)
- [Error gallery](gallery/index.html)

## Error gallery

| Category | Image | Class | Score | IoU |
|----------|-------|-------|-------|-----|
| classification | gallery/classification/01_Japan_003964.jpg | D10 | 0.053 | 0.751 |
| localization | gallery/localization/01_India_004208.jpg | D40 | 0.088 | 0.450 |
| duplicate | gallery/duplicate/01_India_004196.jpg | D40 | 0.068 | 0.736 |
| background | gallery/background/01_Japan_004038.jpg | D10 | 0.113 | — |
| missed | gallery/missed/01_India_001776.jpg | D20 | — | — |
| classification | gallery/classification/02_Czech_002628.jpg | D00 | 0.048 | 0.819 |
| localization | gallery/localization/02_Japan_011031.jpg | D10 | 0.085 | 0.432 |
| duplicate | gallery/duplicate/02_Japan_012225.jpg | D10 | 0.052 | 0.608 |
| background | gallery/background/02_Japan_010983.jpg | D40 | 0.108 | — |
| missed | gallery/missed/02_Japan_012120.jpg | D20 | — | — |
| classification | gallery/classification/03_Japan_012780.jpg | D00 | 0.048 | 0.761 |
| localization | gallery/localization/03_India_005305.jpg | D40 | 0.084 | 0.420 |
| duplicate | gallery/duplicate/03_India_009097.jpg | D40 | 0.050 | 0.687 |
| background | gallery/background/03_Japan_004021.jpg | D10 | 0.105 | — |
| missed | gallery/missed/03_India_009750.jpg | D20 | — | — |
| classification | gallery/classification/04_Japan_004021.jpg | D00 | 0.046 | 0.773 |
| localization | gallery/localization/04_India_005286.jpg | D40 | 0.076 | 0.477 |
| duplicate | gallery/duplicate/04_Japan_000376.jpg | D10 | 0.048 | 0.874 |
| background | gallery/background/04_Japan_012920.jpg | D10 | 0.104 | — |
| missed | gallery/missed/04_Japan_003919.jpg | D20 | — | — |
| classification | gallery/classification/05_India_009009.jpg | D40 | 0.044 | 0.748 |
| localization | gallery/localization/05_India_005305.jpg | D40 | 0.075 | 0.250 |
| duplicate | gallery/duplicate/05_Japan_012225.jpg | D10 | 0.047 | 0.588 |
| background | gallery/background/05_Japan_012770.jpg | D40 | 0.095 | — |

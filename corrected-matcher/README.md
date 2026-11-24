# Corrected matcher and corrected TP-prefix records

This directory supports the major revision of manuscript `jimaging-4735760`
("Fixed-Budget Reallocation in YOLO Detection: An Oracle Diagnostic of
Worst-Image Recall").

## Contents

- `budget_scan_corrected.py` — corrected matcher implementation.
  SHA-256: `04C59D98860CB778221B0427D6D5632CA1D5B1878FA03B3337077D760DF959E8`.
  Semantics: unmatched-GT-only argmax, IoU >= 0.5, class-agnostic, canonical
  order (-confidence, tie = input index).
- `records/s42_o2m.json`, `s42_o2o.json`, `s43_o2m.json`, `s43_o2o.json`,
  `s44_o2m.json`, `s44_o2o.json` — per-image TP-prefix records for the 548
  VisDrone2019-DET validation images. Each row contains `name`, `n_gt`,
  `n_pred`, and `tp_prefix` (cumulative TP count for each confidence-ordered
  prefix up to 600).

## License separation

- **Code** (`budget_scan_corrected.py`, scripts in this directory): MIT,
  see `LICENSE`.
- **Records** (`records/*.json`): **not MIT**. They are derived from
  VisDrone2019-DET annotations and are made available under
  **CC BY-NC-SA 3.0**, with attribution; see `DATA-LICENSE.md`.
- No VisDrone images or raw annotation files are included.

## What can be verified

These records support **numerical reconstruction** of the corrected Table 1
values from archived TP-prefix records. They do **not** provide a complete
end-to-end replay from raw images and model weights; raw predictions and
annotations must be obtained independently under their own licenses.

The earlier matcher under `src/eval/budget_scan.py` is the historical
implementation and is not the corrected matcher used for the revised Table 1.

## Attribution

VisDrone2019-DET: Zhu, P.; Wen, L.; Du, D.; Bian, X.; Fan, H.; Hu, Q.;
Ling, H. Detection and Tracking Meet Drones Challenge. IEEE TPAMI 44 (2022)
7380-7399. DOI: 10.1109/TPAMI.2021.3119563. Dataset license: CC BY-NC-SA 3.0.

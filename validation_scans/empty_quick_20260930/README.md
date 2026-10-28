# Empty-bucket quick scans, 2026-09-30

Eight quick scans (strokes only, no tilting end scan) of the **empty**
bucket, taken on 2026-09-30 at 16:12. They used to be
`baseline_scans_quick/`, the quick scan's own baseline set
(dd08cf7).

**They are not baselines any more.** Since addfcf2 (the same day),
both kinds of scan are modelled from the one set of full baselines in
`baseline_scans/`. A quick scan is compared against those baselines'
stroke readings only (`scan/segments.py`). Nothing in the code reads
this folder.

**No longer valid as empty-bucket tests (2026-10-26).** They were taken with the old ToF mount. Against the 2026-10-22 baselines (new mount), all 8 report the same strong cluster (8-12σ, 56-190 cm³) at about (−0.04, −0.49, 0.37) on one side wall, a mount/geometry difference rather than noise. New empty scans with the current mount are needed for this role (issue #22).

They're kept as test data. The model was never built from them, so
detection on any of them should find **nothing**. That makes them an
independent check for false alarms on quick scans, like
`val_empty_repeat` in HARDWARE_TESTS.md section F, but eight times:

```bash
laundry detect validation_scans/empty_quick_20260930/baseline_20260930_161228_01.csv
laundry evaluate --repeat validation_scans/empty_quick_20260930/baseline_20260930_161228_01.csv
```

Caveat: they were recorded before daf0c00, which changed how the
stroke speed is chosen. At the default 3 cm step the path should be
the same, but nobody has checked. They also only match the bucket
as it was placed that day: once the bucket moves and the baselines
are re-collected, these no longer apply.

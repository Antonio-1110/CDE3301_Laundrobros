# Mouth anomaly ranked ahead of the laundry, 2026-10-21

`run_20261021_173027.csv` is the quick scan from a `laundry run` on the
rig at 17:30. The detector ranks an anomaly near the bucket **mouth**
ahead of the large cluster deeper in, so the grasp goes for the mouth
first.

## What the detector reports

Against `baseline_scans/` (the 8 full baselines in use that day,
recorded 2026-09-30),
with the default thresholds, ranked as `grasp.execute.rank_clusters`
orders them:

| rank | points | volume | peak | mean intrusion | from closed end |
|---|---|---|---|---|---|
| 1 | 16 | 23 cm³ | 7.2 σ | 6 mm | 47 cm (mouth) |
| 2 | 47 | 182 cm³ | 7.8 σ | 7 mm | 45 cm (mouth) |
| 3 | 730 | 2649 cm³ | 53.9 σ | 34 mm | 19 cm |

All three count as `confident`. `rank_clusters` puts confident
clusters mouth-first by position, whatever their size, so the two
shallow mouth clusters come before the item.

The 17:27 run (`scan_records/run_20261021_172752.csv`, not committed)
shows the same pattern: a 104-point, 55 cm³, 6 mm cluster is ranked
ahead of an 832-point cluster.

## Reproduce (offline, no arm)

```bash
laundry detect validation_scans/mouth_false_positive_20261021/run_20261021_173027.csv
```

## Expected once fixed

The 730-point cluster first, and the shallow mouth clusters either not
reported or ranked after it.

**Resolved by cause 1 (2026-10-26).** Against the baselines re-collected on
2026-10-22 with the new mount, this scan returns only the laundry
cluster (707 points, 44.8σ), even with the strong-or-large gate of
issue #22 switched off (`--min-peak-sigma 0`). The ranking (2.) is
unchanged.

Likely causes to check, in this order:

1. **Stale baselines.** They were recorded on 2026-09-30, but the ToF
   offset was changed for a new mount on 2026-10-19 (74488d3). Re-collect
   them (`laundry baseline collect --archive`, empty bucket) and detect
   again before changing any code.
2. **The ranking.** `rank_clusters` takes confident clusters mouth-first
   whatever their size. A minimum volume or intrusion before position
   counts would stop a 23 cm³, 6 mm cluster from jumping the queue.
3. **The mouth itself.** The rim is where the wall model is thinnest, so
   it may need a stricter gate.

The scan only matches the bucket as it was placed that day.

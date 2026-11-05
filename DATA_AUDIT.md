# Data audit — battery plate dataset

This is a record of a programmatic investigation of the images in `data/`
(performed with pixel-level comparison, feature-similarity clustering, and
metadata analysis — not manual visual inspection). The image files themselves
were **not modified**.

## 1. Synthetic vs real

| Set | Files | Resolution | Verdict |
|---|---|---|---|
| `bend/` (15) | `01_tab_tilt_*` … `15_corner_bend_*` | all 767×1023 | **synthetic** (generated) |
| `crack/` (10) | `01_upper_left_crack` … `10_multi_hairline_cracks` | all 767×1023 | **synthetic** (generated) |
| `good/` (14) | `1.png` … `14.jpg` | varied (640–1792 px) | real photos |
| `surface_error/` (24) | `1.jpg` … `24.jpg` | varied (338–640 px) | real photos |
| `crack/` (17) | `12.jpg` … `28.jpg` | varied (640–2048 px) | real photos, but see §2 |

The 25 synthetic files all share the unusual 767×1023 resolution and are
pixel-level near-identical to a **single source photo, `good/3.jpg`**.

## 2. The `crack` folder is derived from the `good` folder

Pixel comparison (256×256 RGB) shows the numeric `crack/` files are the same
photos as `good/` files:

| crack file(s) | matches | relationship |
|---|---|---|
| `crack/26.png` | `good/14.jpg` | identical (diff ≈ 0.01) |
| `crack/18.jpg`, `19.jpg` | `good/5.jpg` | identical (diff ≈ 0.2) |
| `crack/13,14,15,16,21,22` | `good/11,12,8,4` | identical or + thin crack line |
| `crack/17,23,24,27,28` | `good/6,9,13` | identical or + thin crack line |
| `crack/12,20,25` | `good/7,14` | near-identical |

So `crack/` contains (a) exact duplicates of `good/` photos with a conflicting
label, and (b) `good/` photos with a thin crack drawn on. It is not an
independent set of real crack photographs.

## 3. Plate grouping (proposed)

Real photos cluster into **5 physical plates** shared across `good` + `crack`:

```
Plate 1:  good/1.png, 2.png, 3.jpg
Plate 2:  good/4, 8, 11, 12  +  crack/13, 14, 15, 16, 21, 22
Plate 3:  good/6, 9, 13      +  crack/17, 23, 24, 27, 28
Plate 4:  good/7, 10, 14     +  crack/12, 20, 25, 26
Plate 5:  good/5             +  crack/18, 19
```

`surface_error/` clusters into ~10 groups:

```
SE1: 1, 3, 5, 6, 8, 10, 14     SE2: 2, 7, 9, 11, 13     SE3: 15, 22
SE4: 16, 20, 24                 SE5: 17, 21               SE6: 4
SE7: 12                         SE8: 18    SE9: 19    SE10: 23
```

> These groupings come from feature-similarity clustering and should be
> confirmed by eye before being used to compute final metrics.

## 4. Implications for evaluation

- The leave-one-plate-out split is currently **invalid**: the same physical
  plate (even the same pixels) appears in both `good/` and `crack/`, so the
  model can memorise a plate's appearance during training and "detect" it at
  test time. Reported scores are inflated.
- `bend/` has **zero real examples**; any bend metric is meaningless as-is.
- `crack/` is largely `good/` + digital overlays, so crack detection
  performance does not reflect real fracture detection.

## 5. Recommended actions (not yet applied)

1. Mark all 767×1023 files (`bend/*`, `crack/01–10`) as synthetic, and re-label
   the descriptive `crack/01–10` files (currently near-duplicates of each other
   and of `good/3.jpg`).
2. Remove the pixel-identical `crack` duplicates that carry a conflicting label,
   or correct their label.
3. Assign the plate IDs from §3 so `good` + `crack` share plate IDs, then re-run
   the cross-validation.

# Jetson Codec Phase 1 Winner Map

## Assumptions

- Winner map uses the largest `n_images` summary available for each resolution.
- That means `512x640` uses `baseline_21img`, not the earlier `baseline_5img` draft.
- Strict split consistency filter: `match_ratio_mean >= 0.999`, `precision_like_match_ratio_mean >= 0.999`, `class_agreement_ratio_mean >= 0.999`.
- `full_local` is treated as the local baseline, not a split candidate.

## Map 1: Global Fastest

| Resolution | Global Winner | Global ms | Best Split | Best Split ms | Gap vs Full Local |
| --- | --- | --- | --- | --- | --- |
| 384x480 | full_local | 31.127 | p5+fp16 | 34.088 | 2.961 ms |
| 512x640 | full_local | 31.769 | p5+fp16 | 41.206 | 9.437 ms |
| 640x640 | full_local | 39.465 | p5+fp16 | 59.094 | 19.630 ms |

结论：在 Jetson 本地 phase-1 数据里，`full_local` 在所有分辨率下都是总 winner。

## Map 2: Fastest Strict Split

| Resolution | Winner | Split ms | Payload Bytes | match_ratio | precision_like | mean_iou |
| --- | --- | --- | --- | --- | --- | --- |
| 384x480 | p5+fp16 | 34.088 | 645120 | 1.000 | 1.000 | 0.999999 |
| 512x640 | p5+fp16 | 41.206 | 1146880 | 1.000 | 1.000 | 0.999980 |
| 640x640 | p5+fp16 | 59.094 | 1433600 | 1.000 | 1.000 | 0.999993 |

结论：如果只看 split 候选且要求检测一致性不过线不进图，三档分辨率都是 `p5+fp16` 最快。

## Map 3: Smallest Strict-Pass Payload

| Resolution | Winner | Payload Bytes | Split ms | match_ratio | precision_like | mean_iou |
| --- | --- | --- | --- | --- | --- | --- |
| 384x480 | p5+fp16 | 645120 | 34.088 | 1.000 | 1.000 | 0.999999 |
| 512x640 | p5+int8 | 573452 | 43.587 | 1.000 | 1.000 | 0.998076 |
| 640x640 | p5+fp16 | 1433600 | 59.094 | 1.000 | 1.000 | 0.999993 |

补充：这个视角更像链路预算视角。`512x640` 下 `p5+int8` 仍然通过 strict filter，所以 payload 可以从 `1146880 B` 压到 `573452 B`；`384x480` 和 `640x640` 下，`int8` 的 `precision_like_match_ratio_mean` 没过 strict 线，所以 strict payload winner 仍是 `p5+fp16`。

## Per-Resolution Notes

### 384x480

| Candidate | Split ms | Payload Bytes | match_ratio | precision_like | mean_iou | Strict Pass |
| --- | --- | --- | --- | --- | --- | --- |
| p5+fp16 | 34.088 | 645120 | 1.000 | 1.000 | 0.999999 | yes |
| p4+fp16 | 34.330 | 645120 | 1.000 | 1.000 | 0.999996 | yes |
| p3+fp16 | 35.021 | 645120 | 1.000 | 1.000 | 0.999990 | yes |
| p5+int8 | 35.713 | 322572 | 1.000 | 0.972 | 0.996673 | no |
| p4+int8 | 35.891 | 322572 | 1.000 | 0.972 | 0.996559 | no |
| p3+int8 | 36.816 | 322572 | 1.000 | 0.972 | 0.998329 | no |

### 512x640

| Candidate | Split ms | Payload Bytes | match_ratio | precision_like | mean_iou | Strict Pass |
| --- | --- | --- | --- | --- | --- | --- |
| p5+fp16 | 41.206 | 1146880 | 1.000 | 1.000 | 0.999980 | yes |
| p4+fp16 | 41.979 | 1146880 | 1.000 | 1.000 | 0.999978 | yes |
| p3+fp16 | 42.472 | 1146880 | 1.000 | 1.000 | 0.999973 | yes |
| p5+int4 | 43.268 | 286732 | 0.643 | 0.827 | 0.796803 | no |
| p5+int8 | 43.587 | 573452 | 1.000 | 1.000 | 0.998076 | yes |
| p4+int4 | 44.280 | 286732 | 0.607 | 0.767 | 0.746088 | no |
| p4+int8 | 44.303 | 573452 | 1.000 | 1.000 | 0.998086 | yes |
| p3+int8 | 44.806 | 573452 | 1.000 | 1.000 | 0.998292 | yes |
| p3+int4 | 44.922 | 286732 | 0.509 | 0.738 | 0.736253 | no |

### 640x640

| Candidate | Split ms | Payload Bytes | match_ratio | precision_like | mean_iou | Strict Pass |
| --- | --- | --- | --- | --- | --- | --- |
| p5+fp16 | 59.094 | 1433600 | 1.000 | 1.000 | 0.999993 | yes |
| p4+fp16 | 59.964 | 1433600 | 1.000 | 1.000 | 0.999991 | yes |
| p3+fp16 | 60.216 | 1433600 | 1.000 | 1.000 | 0.999985 | yes |
| p5+int8 | 62.527 | 716812 | 1.000 | 0.988 | 0.999123 | no |
| p4+int8 | 63.391 | 716812 | 1.000 | 0.988 | 0.999093 | no |
| p3+int8 | 63.517 | 716812 | 1.000 | 0.976 | 0.999178 | no |


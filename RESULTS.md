# Results

## Scope

Local Jetson profiling with:
- model: `yolov8n.pt`
- input size: `512 x 640`
- splits: `p3 / p4 / p5`

## Prefix baseline

| Split | Prefix Mean (ms) | Frontend Total Mean (ms) |
|------|-------------------|--------------------------|
| p3   | 13.120            | 20.031                   |
| p4   | 13.985            | 20.761                   |
| p5   | 15.369            | 22.143                   |

## Suffix baseline v2

| Split | Edge Post Mean (ms) | Suffix Total Mean (ms) |
|------|----------------------|------------------------|
| p3   | 10.229               | 13.087                 |
| p4   | 8.503                | 11.246                 |
| p5   | 6.888                | 10.269                 |

## Payload size

At `512 x 640`, all three splits currently produce the same raw payload size:

- `2293760 B`

## Combined local profile

| Split | Prefix + Edge Post (ms) | Frontend Total + Suffix Total (ms) |
|------|--------------------------|------------------------------------|
| p3   | 23.349                   | 33.118                             |
| p4   | 22.487                   | 32.007                             |
| p5   | 22.257                   | 32.412                             |

## Takeaway

- `p3`: front-light, back-heavy
- `p5`: front-heavy, back-light
- `p4`: currently the most balanced split

## Next step

Add codec baselines on Jetson:
- `fp16`
- `int8`

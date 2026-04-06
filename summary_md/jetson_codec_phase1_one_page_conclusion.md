# Jetson Codec Phase 1 一页结论稿

## 研究目标

本轮 phase-1 的目标，是在 Jetson Orin NX 上建立 `full local` 与 `split + codec` 的本地画像，回答三个问题：

1. 在不同分辨率下，当前最优方案是谁。
2. `split` 位置与 `codec` 类型分别在时延、payload、检测一致性上起什么作用。
3. 哪个组合最值得进入下一轮真实链路实验。

当前结论基于本地 roundtrip 数据生成，不包含真实无线传输时延，因此它更适合回答“候选组合筛选”问题，而不是直接回答“真实系统端到端最优”问题。对应分析见 [jetson_codec_phase1_winner_map_summary.md](/home/nvidia/jetson_split/summary_md/jetson_codec_phase1_winner_map_summary.md)。

## 核心结论

### 1. 如果任务是“全程都在 Jetson 本地跑”，当前总 winner 始终是 `full_local`

在 `384x480 / 512x640 / 640x640` 三档分辨率下，`full_local` 都比最优 split 候选更快：

- `384x480`：`full_local = 31.13 ms`，最优 split `p5+fp16 = 34.09 ms`，慢 `+9.5%`
- `512x640`：`full_local = 31.77 ms`，最优 split `p5+fp16 = 41.21 ms`，慢 `+29.7%`
- `640x640`：`full_local = 39.46 ms`，最优 split `p5+fp16 = 59.09 ms`，慢 `+49.7%`

这说明：在不考虑传输收益的前提下，当前 split 路径不适合被解释成“本地加速方案”，它更像是为后续边云协同和链路受限场景准备的候选机制。

### 2. 在所有 split 候选里，`p5` 是稳定的一号位，`p4` 是稳定的二号位，`p3` 最慢

这个排序在 `fp16 / int8 / int4` 上都成立。原因不是 `p5` 前端更轻，而是它虽然让前端 prefix 更重，却让后端 suffix replay 降得更多，最终总时延最优。

以 `512x640 + fp16` 为例：

- `p3`：`frontend = 26.57 ms`，`edge_post = 12.43 ms`，总计 `42.47 ms`
- `p5`：`frontend = 29.09 ms`，`edge_post = 8.53 ms`，总计 `41.21 ms`

也就是说，`p5` 比 `p3` 多付出约 `2.5 ms` 前端成本，却换回约 `3.9 ms` 后端收益，净结果更好。  
这说明在当前 Jetson 侧画像下，后移 split 点是正确方向。

### 3. 当前 phase-1 中，`split` 决定计算划分，`codec` 决定带宽占用

在同一分辨率下，`p3 / p4 / p5` 的 payload bytes 基本一致，变化主要来自 codec：

- `fp16`：约 `2x` 压缩
- `int8`：约 `4x` 压缩，payload 约为 `fp16` 的一半
- `int4`：约 `8x` 压缩，payload 约为 `fp16` 的四分之一

因此，当前阶段里：

- 选 `split`，主要是在调前后端计算平衡
- 选 `codec`，主要是在调精度与链路负载平衡

### 4. `fp16` 是当前最稳的主 baseline，`p5+fp16` 是最适合写进主结论的 split winner

在三档分辨率下，`p5+fp16` 都满足 strict consistency 条件：

- `match_ratio_mean = 1.0`
- `precision_like_match_ratio_mean = 1.0`
- `class_agreement_ratio_mean = 1.0`
- `mean_iou_mean` 接近 `1.0`

因此，如果当前要给出一个“稳定、可解释、适合作为主线方案”的 split 结论，应该写成：

> 在 phase-1 Jetson 本地实验中，`p5+fp16` 是所有 split 候选中最稳且最快的组合。

### 5. `int8` 是最值得继续追的压缩方向，但它明显存在分辨率敏感性

`int8` 的优点非常明确：payload 直接减半，而本地 roundtrip 仅增加约 `1.6-3.4 ms`。  
但按当前 strict 口径，它只在 `512x640` 下通过，在 `384x480` 和 `640x640` 下未通过。

以 `p5` 为例：

- `384x480`：`fp16 -> int8` 后，时延 `+1.62 ms`，payload 减半，但 `precision_like` 从 `1.000` 降到 `0.972`
- `512x640`：`fp16 -> int8` 后，时延 `+2.38 ms`，payload 减半，同时仍通过 strict filter
- `640x640`：`fp16 -> int8` 后，时延 `+3.43 ms`，payload 减半，但 `precision_like` 从 `1.000` 降到 `0.988`

更重要的是，`int8` 在失败分辨率上的 `match_ratio` 仍接近 `1.0`，说明它不是严重漏检，更像是会引入额外框或置信度扰动。  
这意味着它仍是“可调优”的方向，而不是已经不可用的方向。

### 6. `int4` 当前不适合作为主线候选

虽然 `int4` 把 payload 压到了 `fp16` 的四分之一，但在 `512x640` 下，最佳的 `p5+int4` 仍出现：

- `match_ratio = 0.643`
- `precision_like = 0.827`
- `mean_iou = 0.7968`

而且它的本地 roundtrip 还比 `p5+fp16` 更慢。  
这说明当前版本的 `int4` 既没有守住检测一致性，也没有换来更好的本地时延，不应进入下一轮主线实验。

## 论文/汇报口径下的总体判断

如果把这轮实验压成一句话，可以写成：

> Phase-1 结果表明，`full_local` 仍是 Jetson 纯本地部署下的全局最优；但在所有 split 候选中，`p5+fp16` 稳定成为最优基线，而 `p5+int8` 在 `512x640` 下首次表现出“显著降低 payload 且保持检测一致性”的潜力，因此应成为下一轮真实链路实验的核心候选。

## 下一步实验建议

### 优先级 1：做真实链路端到端验证，而不是继续只看本地 roundtrip

下一轮最应该做的，不是再补更多本地 benchmark，而是把真实传输加回来，验证 split 是否能在真实系统里赢回来。

建议对比三组主候选：

- `full_local`
- `p5+fp16`
- `p5+int8`

建议在至少三类链路条件下测端到端时延：

- 高带宽低时延链路
- 中等带宽链路
- 低带宽或抖动明显的链路

输出指标建议包括：

- 端到端总时延
- 上传 payload 大小
- suffix 端时延
- 检测一致性指标

这一步会直接回答：`p5+int8` 是否能在真实链路里超过 `p5+fp16`，以及是否有机会接近甚至超过 `full_local`。

### 优先级 2：围绕 `p5+int8` 做定向稳定性优化

当前 `int8` 最大问题不是链路收益不够，而是跨分辨率稳定性不够。  
因此下一步不建议平均地调所有组合，而是只聚焦：

- `p5+int8 @ 384x480`
- `p5+int8 @ 640x640`

重点看三类改动是否能把 `precision_like_match_ratio_mean` 拉回 strict 阈值以上：

- 更好的量化校准样本
- 分辨率/分 split 的独立量化参数
- 解码后置信度阈值与 NMS 参数微调

目标不是把 `int8` 变成“理论最优”，而是先把它变成“跨分辨率稳定可用”。

### 优先级 3：补一轮带标注的小规模精度验证

当前 winner map 用的是 detection consistency proxy，而不是真实 mAP。  
它足够适合 phase-1 筛选，但如果要写论文或做正式结论，下一步最好补一个小规模带标注子集，验证：

- consistency proxy 与真实精度之间是否一致
- `p5+int8` 在真实检测质量上是否仍能接受

不需要一开始就上大规模评测，先做一个小而干净的验证集即可。

## 最推荐的执行顺序

如果只保留一句行动建议：

> 下一轮应以 `p5+fp16` 作为稳定基线，以 `p5+int8 @ 512x640` 作为主攻对象，优先开展“真实链路端到端实验 + int8 稳定性修正”，而不是继续扩展新的 split 或继续推进当前版本的 int4。

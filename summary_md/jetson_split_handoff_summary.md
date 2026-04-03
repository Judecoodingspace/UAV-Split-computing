# Jetson Split YOLO 实验续接文档（本对话 handoff）

## 0. 文档用途

这份文档用于把当前对话的研究 / 开发状态、核心结论、未解决问题、下一步任务，完整传递到下一个对话。

---

## 1. 当前核心上下文锚点

### 1.1 项目 / 任务核心目标

**把服务器上已验证的 Split YOLO 执行链迁移到 Jetson Orin NX 上，获得更真实的 UAV 前端 / 后端计算画像，并为后续 `split + codec` 联合实验打基础。**

### 1.2 必须知道的核心概念 / 定义

- **当前 Split 不是 single-tensor 切分，而是 multi-tensor payload split。**
- 当前三种 split 定义：
  - **`p3`**：`payload_layers = [9, 12, 15]`，`replay_start = 16`
  - **`p4`**：`payload_layers = [9, 15, 18]`，`replay_start = 19`
  - **`p5`**：`payload_layers = [15, 18, 21]`，`replay_start = 22`
- **`forward_to_split()`**：前端 prefix 执行，输出 `payload`、`uav_pre_ms`
- **`forward_from_split()`**：后端 suffix replay，输出 `raw_output`、`edge_post_ms`
- 当前 **raw payload 总字节数在 `512×640` 输入下三种 split 相同：`2293760 B`**
- 当前对 split 的理解：
  - **split 主要对应“算力划分”**
  - **codec 主要对应“链路预算划分”**
- 当前研究不再停留在“选 `fp16 / int8 / int4`”，而是逐步走向：
  - **`split + codec (+ bitrate)` 联合匹配**
  - 后续再考虑 **任务偏好 / 用户偏好 / 链路状态**

### 1.3 当前对话核心主线

- **完成 Jetson 环境搭建与可运行验证**
- **把 `yolo_split_executor_v1` 适配到 Jetson**
- **跑通 Jetson 端 prefix benchmark**
- **导出 payload 并跑通 Jetson 端 suffix replay**
- **得到 Jetson 本地 `p3 / p4 / p5` 的前后端计算画像**
- **确认下一步应进入 `fp16 / int8` codec 阶段**

---

## 2. 已完成 / 已确认的硬核信息

### 2.1 已跑通的代码 / 脚本

#### 已在 Jetson 上跑通
- **`jetson_split_executor.py`**
- **`benchmark_split_prefix_jetson.py`**
- **`benchmark_split_prefix_batch_jetson.py`**
- **`export_split_payload_jetson.py`**
- **`benchmark_split_suffix_jetson.py`**
- **`benchmark_split_suffix_jetson_v2.py`**（带 warmup 修正版）

#### 已生成的实验产物
- **Jetson 前端基线 detail / summary CSV**
- **payload_bank**
- **Jetson suffix detail / summary CSV**
- **Jetson 前后端合并计算画像表**
  - `jetson_split_compute_profile_summary.csv`
  - `jetson_split_compute_profile_summary_v2.csv`

---

### 2.2 已确认的环境 / 硬件 / 软件

#### Jetson 硬件环境
- **设备**：Jetson Orin NX 16GB
- **JetPack**：6.2
- **L4T**：36.4.3
- **Ubuntu**：22.04
- **CUDA**：12.6
- **cuDNN**：9.3
- **TensorRT**：10.3
- **OpenCV**：4.10（CUDA=YES）

#### Jetson Python / 深度学习环境
- **Python**：3.10.12
- **torch**：`2.5.0a0+8729d72e41.nv24.08`
- **torchvision**：`0.20.0a0+afc54f7`
- **ultralytics**：`8.3.154`
- **CUDA 可用**：`torch.cuda.is_available() == True`

#### 服务器情况
- **服务器旧实验确认是 CPU 路径**
- **原因**：服务器驱动太旧，装不了 GPU 版 torch
- **结论**：服务器旧时间不能与 Jetson 新时间做绝对值比较，只能作为早期趋势参考

---

### 2.3 已配置好的关键路径

#### Jetson 项目目录
- **项目根目录**：`/home/nvidia/jetson_split/`
- **权重目录**：`/home/nvidia/jetson_split/weights/`
- **权重文件**：`/home/nvidia/jetson_split/weights/yolov8n.pt`
- **源码目录**：`/home/nvidia/jetson_split/src/`
- **脚本目录**：`/home/nvidia/jetson_split/scripts/`
- **图片目录**：`/home/nvidia/jetson_split/data/`
- **输出目录**：`/home/nvidia/jetson_split/outputs/`

#### 虚拟环境
- **venv**：`/home/nvidia/venvs/jetson-split`
- 激活方式：
  ```bash
  source ~/venvs/jetson-split/bin/activate
  export PYTHONPATH=/home/nvidia/jetson_split/src:$PYTHONPATH
  ```

---

### 2.4 已确认的输入口径

- **Jetson benchmark 默认输入尺寸已对齐服务器：`512×640`**
- 之前出现 `640×640` 导致 payload shape / bytes 偏大，现已修正
- 在 `512×640` 下，payload shape 档位为：
  - `[1, 256, 16, 20]`
  - `[1, 128, 32, 40]`
  - `[1, 64, 64, 80]`
- 三种 split 的 raw payload bytes 均为：
  - **`2293760 B`**

---

### 2.5 Jetson 前端 21 张图基线（已确认）

#### 前端 prefix mean（21 张图）
- **`p3`**：`13.120 ms`
- **`p4`**：`13.985 ms`
- **`p5`**：`15.369 ms`

#### 前端 total mean（21 张图）
- **`p3`**：`20.031 ms`
- **`p4`**：`20.761 ms`
- **`p5`**：`22.143 ms`

#### 关键结论
- **前端计算趋势：`p3 < p4 < p5`**
- **split 越往后，Jetson 前缀计算越重**
- **三种 split 的 raw payload 大小相同，差异主要来自前端计算**

---

### 2.6 Jetson suffix v2（warmup 后）基线（已确认）

#### 后端 edge_post mean（warmup 修正后）
- **`p3`**：`10.229 ms`
- **`p4`**：`8.503 ms`
- **`p5`**：`6.888 ms`

#### suffix total mean（warmup 修正后）
- **`p3`**：`13.087 ms`
- **`p4`**：`11.246 ms`
- **`p5`**：`10.269 ms`

#### 关键结论
- **后端计算趋势：`p3 > p4 > p5`**
- **split 越靠前，后缀 replay 越重**
- **warmup 后，suffix 结果已回到合理区间**

---

### 2.7 当前最新 Jetson 本地完整计算画像（v2）

#### 纯计算总时间（`prefix + edge_post`）
- **`p3`**：`23.349 ms`
- **`p4`**：`22.487 ms`
- **`p5`**：`22.257 ms`

#### 全链路本地总时间（`frontend_total + suffix_total`）
- **`p3`**：`33.118 ms`
- **`p4`**：`32.007 ms`
- **`p5`**：`32.412 ms`

#### 前后计算占比
- **`p3`**：前端 `56.19%`，后端 `43.81%`
- **`p4`**：前端 `62.19%`，后端 `37.81%`
- **`p5`**：前端 `69.05%`，后端 `30.95%`

#### 当前阶段最重要的结论
- **`p3` 更像“前端轻、后端重”的 split**
- **`p5` 更像“前端重、后端轻”的 split**
- **`p4` 是当前 Jetson 本地计算画像里最均衡的一档**

---

### 2.8 当前项目结构（最小核心）

```text
/home/nvidia/jetson_split/
├── weights/
│   └── yolov8n.pt
├── data/
│   └── 00191.jpg ... 00211.jpg
├── src/
│   └── jetson_split_executor.py
├── scripts/
│   ├── benchmark_split_prefix_jetson.py
│   ├── benchmark_split_prefix_batch_jetson.py
│   ├── export_split_payload_jetson.py
│   ├── benchmark_split_suffix_jetson.py
│   └── benchmark_split_suffix_jetson_v2.py
└── outputs/
    ├── front_baseline_batch/
    ├── payload_bank/
    ├── suffix_baseline/
    └── suffix_baseline_v2/
```

---

## 3. 当前未解决的问题 / 卡点

### 3.1 当前无阻塞性报错

**当前没有阻塞性运行错误。**  
Jetson 侧 prefix、payload 导出、suffix replay 都已经跑通。

---

### 3.2 仍需注意的实验问题

#### 问题 A：suffix 仍存在少量离群值
虽然 warmup 后明显改善，但仍观察到个别 payload 的 `deserialize` 或 `edge_post` 异常偏高。

**已观察到的典型现象：**
- 旧版 suffix 无 warmup 时，`p3` 首样本曾出现异常大值，把均值严重拉高
- v2 后仍存在少量 `deserialize` 离群值，例如某个 `p5` payload 的 `deserialize_mean_ms` 明显偏高

**核心疑点：**
- 可能与 `torch.load` 首次文件读取 / 缓存状态 / 文件系统抖动有关
- 不是 split 定义错误，更像测量层面的离群点

---

### 3.3 已解决但必须记住的历史坑（避免下个对话重复试错）

#### 报错 1：device=auto 未被正确解析
关键报错：
```text
Expected one of cpu, cuda, ... device type at start of device string: auto
```
已解决：
- 修正 `jetson_split_executor.py` 的 `_resolve_device()`
- 或运行命令中显式传 `--device cuda:0`

#### 报错 2：benchmark 导入类名失败
关键报错：
```text
cannot import name 'YoloSplitExecutor' from 'jetson_split_executor'
```
已解决：
- 在 executor 末尾加入别名：
  ```python
  YoloSplitExecutor = YoloSplitExecutorJetson
  ```

#### 报错 3：权重自动下载失败（权限 / 路径）
关键报错：
```text
Permission denied
Failed to create the file yolov8n.pt
```
以及：
```text
FileNotFoundError: [Errno 2] No such file or directory: 'yolov8n.pt'
```
已解决：
- 不再依赖 Ultralytics 自动下载
- 固定本地绝对路径：
  `/home/nvidia/jetson_split/weights/yolov8n.pt`

#### 报错 4：虚拟环境创建失败
关键报错：
```text
The virtual environment was not created successfully because ensurepip is not available
```
已解决：
- 安装系统包：
  ```bash
  sudo apt install -y python3.10-venv
  ```

---

## 4. 下一步明确要做的任务

### 优先级 1：进入 `fp16 / int8` codec 阶段
目标：
- 在 Jetson 本地建立 **`split × codec`** 画像

具体动作：
1. **读取现有 `split_payload_codec_v1 / feature_codec_v3` 相关服务器代码**
2. **迁移 / 适配为 Jetson 可运行版本**
3. 写一个新的 roundtrip benchmark，测：
   - `compress_ms`
   - `decompress_ms`
   - `payload_bytes_after_codec`
   - `prefix + codec + suffix`
4. 先只做：
   - **`fp16`**
   - **`int8`**
5. **`int4` 暂不作为主线**

---

### 优先级 2：形成 `split × codec` 的 Jetson 本地总表
目标：
- 生成最终类似：
  - `p3 + fp16`
  - `p3 + int8`
  - `p4 + fp16`
  - `p4 + int8`
  - `p5 + fp16`
  - `p5 + int8`

核心表项：
- `uav_pre_ms`
- `compress_ms`
- `payload_bytes`
- `deserialize_ms`
- `decompress_ms`
- `edge_post_ms`
- `local_roundtrip_total_ms`

---

### 优先级 3：再做真实前后侧分离（Jetson -> 服务器）
目标：
- 让 Jetson 真正承担前端
- 让服务器承担后端 replay

建议顺序：
1. 先做 **payload 文件级交接**
2. 再做 **socket / TCP** 在线发送
3. 先用有线网络，不引入无线链路模拟

---

### 优先级 4：最后再进入 learned codec / 策略层
当前还**不应该**立刻做 learned codec 主体或 RL 策略学习。  
正确顺序是：

1. **先把 `fp16 / int8` 基线做扎实**
2. 再做 **轻量 learned codec**
3. 再做 **`split + codec + bitrate` 联合选择**
4. 最后再考虑：
   - 用户偏好
   - 小目标 / 大目标偏置
   - 链路 belief / 状态策略

---

## 5. 下一个对话续接建议

建议下一个对话直接这样开场：

> **我们已经完成了 Jetson 上 `p3/p4/p5` 的前后端本地计算画像，现在请继续帮我把服务器侧的 `fp16 / int8` payload codec 迁移到 Jetson 上，并生成 `split × codec` 的本地 roundtrip benchmark。**
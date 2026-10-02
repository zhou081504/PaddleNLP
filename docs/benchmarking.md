# 资源评估指南

现在推荐使用与 `speech_demo` 一致的目录入口：`python desensitize.py data` 普通运行，
`python evaluate.py data --devices cpu cuda:0 --batch-sizes 1 4 8 16 --repeats 3` 独立进程比较。
两者输出位于 `脱敏结果/` 和 `评测结果/`，详见根目录 README。
下文专门说明仍保留的 **单文件稳态压测**；其预热/加载时间口径与目录独立进程评测不同，不能直接混比。

## 如何跑一次可比较的测试

在 `PaddleNLP` 环境、项目根目录运行 `python -m benchmarks.run`。它与正式批处理共用同一个 `Pipeline`，只在外围增加采样、预热、重复计时，不另写一套脱敏逻辑。

1. 先用 `examples/demo.txt` 跑通功能；再选有代表性的业务文本。保留短/长文本、实体密度、语种和地址表达的实际比例。全空白/空文件拒绝用于压测。
2. 使用足够的数据，让正式处理至少持续数分钟；短 demo 只能说明本机是否跑通、资源的大致量级。`examples/make_dataset.py` 生成的重复文本用于压力验证，不能代表真实业务分布。
3. 预热至少 1 批，正式重复至少 3 轮；模型只加载一次，初始化单独计时。反复读同一文件可能命中系统缓存。
4. 对比 `--threads 1/2/4/8`、`--model-batch-size 1/4/8/16` 时逐项改变，每次使用新报告目录；`--batch-records` 是外层记录批次，`--model-batch-size` 是 UIE 内部推理批次，两者不是同一个参数。
5. 业务需要落盘时加入 `--write-output`。每轮均计入写文件、flush/fsync 和原子替换；最终文件只保留最后一轮结果。未启用时只读取和处理文本，丢弃结果，不测落盘吞吐。

```bash
conda activate PaddleNLP
python -m benchmarks.run -i data/benchmark.txt --threads 4 --model-batch-size 8 \
  --batch-records 32 --warmup 1 --repeats 3 --sample-interval 0.5 \
  --write-output --report-dir reports/cpu-t4-b8
```

未指定 `--report-dir` 时自动创建带时间戳的新目录。指定目录必须不存在，避免覆盖历史结果。指定 `-o` 时必须同时传 `--write-output`，输出路径也必须是新路径。输入和配置文件受到同路径、软/硬链接保护。

## 输出与指标口径

`report.json` 为完整汇总；`samples.csv` 边运行边写，可用表格工具打开或持续读取。启用写输出且未指定 `-o` 时，脱敏文件为报告目录的 `redacted.txt`。采样使用独立进程，避免模型的原生推理阻塞 Python 线程、导致资源采样失真；采样进程不加载模型，它自身的小量 CPU/内存也计入进程树。统计采用固定空间；延迟用最多 4096 个批次的 reservoir 样本，处理大文件不会保存全部文本、全部延迟或全部资源样本在内存中。

| 字段 | 用途与口径 |
| --- | --- |
| `input` | 输入字节数、记录数、字符数、空行数；字符数按 Python Unicode 字符计，含换行，不等于字节数 |
| `initialization.seconds` | 构造流水线和加载模型耗时；缓存不存在时可能包含下载，需单独识别 |
| `warmup` | 预热批次及耗时，不计入稳态吞吐 |
| `rounds` | 每次完整处理文件的速度、耗时与批延迟；空行计入条数但不做模型推理 |
| `summary.records_per_second` | 正式各轮总条数 / 总耗时，含读入、规则、模型、替换及可选写盘；非各轮速度的简单平均 |
| `summary.chars_per_second` | 各轮总字符数 / 总耗时，便于不同记录长度之间辅助比较 |
| `regex_seconds / model_seconds / replace_seconds` | 规则、模型、实体替换的累计耗时；GPU 计时等待推理完成 |
| `batch_latency_seconds` | 单批 `process()` 的均值、P50、P95、最大值；不含文件 I/O，不是单条请求延迟，`approximate` 标明抽样情况 |
| `resources.overall` | 覆盖初始化、预热和正式处理的采样均值/峰值；部署内存需检查初始化峰值 |
| `resources.phases / rounds` | 按阶段及正式轮次分别汇总，可重点比较 `measure` |
| `process_tree_cpu_percent` | 当前进程及被观测到的子进程 CPU，占满一个逻辑核为 100%；400% 约为同时使用 4 个逻辑核 |
| `process_tree_cpu_normalized_percent` | 上述 CPU 值除以本机逻辑核数，约表示占整机 CPU 的比例；容器 CPU 配额/绑定需另行考虑 |
| `host_cpu_percent` | 整机 CPU 利用率，范围 0–100%，包含其他程序；不等同于脱敏进程利用率 |
| `process_tree_rss_bytes` | 进程树 RSS 常驻内存，除以 1024³ 为 GiB；共享页可能重复计数，不等同于独占内存 |
| `host_memory_*` | 整机已用、可用、总内存和利用率，包含其他程序 |
| GPU `utilization_percent / memory_used_mib` | NVIDIA 整卡利用率、整卡已用显存，包含其他进程；不能直接当成当前脱敏进程独占值 |
| GPU `process_tree_memory_mib` | 当前进程树的显存占用；驱动不支持、设备不可用时为 null |

GPU 数据来自 `nvidia-smi`，带查询超时；缺工具、驱动异常、超时都有明确原因，缺值不填成 0。`--device-id` 是 Paddle 可见 GPU 的逻辑编号，CSV 保留整机 GPU UUID/编号；若使用 `CUDA_VISIBLE_DEVICES` 重映射，应按 UUID 对照。CPU 模式仍可观测机器上其他任务使用的 GPU，观测到 GPU 活动不表示本次使用了 GPU。

资源均值为采样点平均，含阶段切换快照；峰值是采样捕获值，可能漏掉瞬时尖峰。很短的子进程也可能未被捕获。CSV 每张 GPU 每次采样一行，CPU 字段会重复，多 GPU 分析时不能将重复行再相加。后台采样自身有少量开销，尤其极快的纯规则任务应增加运行时长，避免用几毫秒的结果定容。

## 将测试换算成部署资源

```bash
python -m benchmarks.run -i data/benchmark.txt --target-records 10000000 \
  --target-hours 24 --headroom 0.3 --report-dir reports/capacity
```

`--headroom 0.3` 表示保留 30% 处理能力。设实测吞吐为 R、总记录数为 N、时限为 H 小时，则报告估算：

- 单工作进程计划吞吐为 `R × 0.7`；单进程工期为 `N / (R × 0.7) / 3600` 小时。
- 并行工作进程需求为 `ceil(N / (R × 0.7 × H × 3600))`，最低 1 个。
- 内存参考值为“观测 RSS 峰值 × 并行进程数 / 0.7”，还需另外留出操作系统、文件缓存、数据接入和其他服务的空间。

这些是相同语料、相同机器和相同参数下的线性粗估，不是采购结论。多进程会复制模型，CPU核数、内存带宽、磁盘和 GPU 争用都会让吞吐无法线性提升。GPU 数量不能从整卡利用率直接换算；需在目标显卡上测显存峰值、批大小及多任务争用。用更长记录、更复杂 schema、更低识别阈值后也要重新测试。

当前工程适合离线分片：一个进程一次处理一个文件，输出按分片原子发布。定容时至少测单进程、目标并行数、持续运行后的 RSS 是否增长，再按实际峰值增加余量。无需一开始加入消息队列或分布式框架；业务接入后可在文件分片层调度，复用现有流水线。

## 质量与失败处理

性能报告不包含原文或实体；失败报告只记录异常类别和固定说明。失败停止采样并保留已观测数据，业务处理异常时不会发布当前轮次的半成品。若前一轮已成功输出而后一轮失败，前一轮完整文件仍可能存在，需以报告状态判断整次基准是否成功。

若后台采样发生异常，`resource_measurement_status` 标记为 `degraded`，保留吞吐结果但停用容量估算。查看 `resources.sampling_errors` 并修复后重测；正常的 GPU 不可用不会导致整次 CPU 测试降级。

吞吐与识别准确率需分别验收。UIE 使用概率阈值、长文本内部无重叠切分；每行上限不代表模型能整体看完该行。正式上线前请用标注数据统计姓名/地址等字段的漏检与误检，尤其覆盖长地址、跨窗口实体、口语和对话记录边界。

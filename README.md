# 文本批量脱敏与算力评测

运行和测评用法与 Desktop 下的 `speech_demo` 对齐：**普通运行传入文件夹；独立评测对比设备和 batch；结果使用中文目录与报告名称。**

规则识别手机号、身份证、银行卡、车牌、座机和连续数字；PaddleNLP UIE 识别姓名、地址、工作单位。每个运行进程复用一个模型，逐文件、逐批处理。

## 环境

```bash
cd /home/zhou/Desktop/GITHUB_CLONE/PaddleNLP
conda activate PaddleNLP
```

本机环境与 `uie-base` 缓存已经可用，无需重装。`paddlenlp/` 是当前 editable 安装依赖的运行时源码，需要保留；原有 `data/input.txt` 和历史结果未移动。

## 普通批量脱敏

把 UTF-8 TXT 文件放进 `data/`，可以包含多层子目录，然后执行：

```bash
python desensitize.py "data"
python desensitize.py "data" --batch-size 8 --threads 4
python desensitize.py "data" --device cpu
python desensitize.py "data" --device cuda:0 --output-dir "脱敏结果_GPU"
python desensitize.py "data" --no-benchmark
```

- 默认递归处理 `.txt` / `.TXT`，输出到项目下的 `脱敏结果/`，保留输入目录层次。
- 默认跳过已有非空结果；`--overwrite` 重新处理。输入或规则配置改变后也应加此参数。
- 单文件失败继续下一个；失败时保留该文件原有完整结果，`处理明细.csv` 标记本次失败。模型加载失败则终止整次任务。
- `--device` 默认 `auto`，自动选择可用 GPU，否则 CPU；支持 `cpu`、`gpu`、`gpu:0`、`cuda:0`。显式指定不可用 GPU 会失败，不静默切回 CPU。
- `--batch-size` / `--batch_size` / `--model-batch-size` 是同一个参数，默认 8，表示 UIE 内部推理批大小；`--batch-records` 是外层每批记录上限，默认 32，二者不同。
- `--no-benchmark` 关闭应用计时、资源采样及性能 JSON，仍输出脱敏文本与处理明细；旧的性能报告不会删除，须看修改时间。
- 当前输出目录自动从输入扫描中排除；可用 `--exclude-dir 路径` 排除其他目录，重复指定可排除多个。

输入输出示例：

```text
data/一月/a.txt  → 脱敏结果/一月/a.txt
data/二月/b.txt  → 脱敏结果/二月/b.txt
```

输入文本按行读取，**每行一条完整业务记录**。支持 BOM，输出无 BOM UTF-8，保留空行、换行形式和末行是否换行。当前不解析 Word、Excel、CSV 或 JSON。默认每行最多 16,384 字符，每批最多 65,536 字符（均含换行）；超过上限报错，不截断。长对话应先按业务整理记录边界。

输出先写临时文件，整文件成功后才原子替换；输入输出同路径、软硬链接覆盖和输出路径越界会被拒绝。目录中符号链接文件不参与扫描。数据量很大时按业务分片存放，每个文件是一个可独立重跑的单元。

## 普通运行的结果

```text
脱敏结果/
├── 一月/a.txt
├── 二月/b.txt
├── 处理明细.csv       每个文件的成功/失败/跳过、条数、字符数、处理耗时
├── 性能统计.json      本次模型加载、端到端耗时、吞吐、CPU/GPU/内存及环境
└── 资源采样.csv       分阶段资源时间线
```

默认启用独立进程监控，约每 0.5 秒采样；模型推理阻塞 Python 线程时仍能持续观测。CPU 100% 表示一个逻辑核，RSS 为进程树常驻内存（含监控进程）。GPU 整卡与进程树显存分别记录，设备/驱动不可用时指标为空并附原因；整机和整卡指标可能包含其他程序，采样可能漏过瞬时峰值。

普通运行的端到端时间包含模型加载、读入、脱敏、保存、失败文件和监控开销，排除初始 Python 导入及最终汇总报告写入；另列成功文件累计处理耗时。跳过的文件不计入本次处理条数，存在跳过或失败时不适合用作完整数据集的配置比较。

## 独立对比设备和 batch

和 `speech_demo/evaluate.py` 一样，每组启动一个新进程，从同一输入集完整重跑。默认自动选设备，对比模型 batch 1/4/8/16，每组一次：

```bash
python evaluate.py "data"

# 先用小样本验证
python evaluate.py "examples" --devices cpu --batch-sizes 4 8 --repeats 1

# CPU/GPU 对比，每组重复3次
python evaluate.py "data" --devices cpu cuda:0 --batch-sizes 1 4 8 16 --repeats 3

# 只对比规则模块，不加载模型（batch 对规则模式不产生推理效果）
python evaluate.py "data" --mode rules --devices cpu --batch-sizes 8
```

结果写入唯一的 `评测结果/run_.../`：

```text
评测结果/run_.../
├── 输入清单.json
├── 评测对比.csv
├── 评测汇总.json
├── 算力评估.md
├── case_001/          第一种设备/batch/重复轮次
│   ├── 脱敏后的文件与子目录
│   ├── 处理明细.csv
│   ├── 性能统计.json
│   ├── 资源采样.csv
│   └── 运行.log
└── case_002/ ...
```

每轮都包含模型加载，**没有额外预热**；评测端到端速度还包含子进程启动、依赖导入、最终报告和退出，与单模型预热后的稳态吞吐口径不同。先准备模型缓存，避免下载影响对比；文件系统缓存和其他进程负载仍会影响结果。

只有全部文件成功、无跳过、输入文件清单/大小/修改时间一致、无采样异常，且同一配置所有重复均成功时，才按端到端条/秒中位数排名。某一配置设备不可用或模型 OOM，保留日志并继续其他配置，不自动调小 batch。输入文件集改变会停止评测并保留已完成报告。全空白输入不能用于容量排名。

## 按业务规模估算节点

```bash
python evaluate.py "data" --devices cpu cuda:0 --repeats 3 \
  --history-records 10000000 --deadline-days 30 \
  --daily-records 100000 --hours-per-day 20 --utilization 0.8
```

不提供业务量时只报告实测速度和单节点每日条数/字符数，不编造采购数量。

```text
单节点日容量 = 实测条/秒 × 3600 × 每日运行小时 × 利用率
历史任务节点数 = ceil(历史条数 / 完成天数 / 单节点日容量)
日增任务节点数 = ceil(日增条数 / 单节点日容量)
同时处理节点数 = ceil((历史条数 / 完成天数 + 日增条数) / 单节点日容量)
```

这里的节点是同等硬件、同等语料下的独立工作节点，不是在当前机器上直接启动相同数量的进程。GPU 节点指同型号单卡和配套主机，CPU 节点指同等 CPU 主机；未验证多节点线性扩展。条数包含空行，容量语料应与业务长度、空行比例、实体密度一致，内存/显存还需留余量。

## 原有单文件与稳态压测

原入口继续可用，适合单文件调用和排除加载时间后的稳态测试：

```bash
python -m text_redaction -i data/input.txt -o outputs/result.txt --device cpu
python benchmark.py -i data/input.txt --device cpu --warmup 1 --repeats 3 --write-output
# 等价于 python -m benchmarks.run ...
```

稳态报告仍在 `reports/` 下，不移动旧报告。详细口径见 [资源评估指南](docs/benchmarking.md)，之前实测见 [本机验证](docs/local-validation.md)。

## 代码结构与调整

```text
desensitize.py          普通目录脱敏入口（对应 speech_demo 的 transcribe.py）
evaluate.py             独立设备/batch评测入口
benchmark.py            单文件稳态测评入口
resource_monitor.py     资源监控接口
text_redaction/         目录批处理、统一流水线、文件读写、模型适配与规则
benchmarks/             独立进程监控、稳态压测与目录评测实现
configs/                规则配置
examples/               合成demo和压测数据生成器
tests_app/              不下载模型的回归测试
paddlenlp/              当前环境依赖的第三方运行时
```

可用 `--config configs/rules.example.json` 指定字段别名和已知值；车牌词库相对配置文件目录解析。连续数字默认至少 7 位，可能覆盖普通订单号；号码是格式识别，不校验真实性。姓名默认全遮盖，`--name-mask keep-first` 可留姓；地址/单位使用类型标记。

`--schema 姓名 地址 工作单位`、`--position-prob 0.5`、`--max-seq-len 512`、`--model uie-base`、`--model-path /path/to/uie` 可调整模型。长文本内部无重叠切分，跨窗口实体可能漏检；性能测试不等于质量验收，应另用标注业务样本统计召回率。

```bash
python -m unittest discover -s tests_app -v
```

新机器先安装适合 CPU/CUDA 的 Paddle，再安装 `requirements.txt` 与本地源码 `python -m pip install -e . --no-deps`。本机依赖快照位于 `requirements/environment-tested.txt`；上次清理记录在 `docs/cleanup.json`。

"""统一批处理入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须为非负整数")
    return number


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("必须在 (0, 1] 内")
    return number


def build_parser(*, directory=False):
    p = argparse.ArgumentParser(description="流式文本脱敏：规则识别号码/车牌，PaddleNLP UIE 识别姓名/地址/单位")
    if directory:
        p.add_argument("input_dir", type=Path, help="输入 TXT 文件夹，递归处理子目录")
    else:
        p.add_argument("-i", "--input", type=Path, default=Path("examples/demo.txt"), help="UTF-8 文本，每行一条记录")
        p.add_argument("-o", "--output", type=Path, help="脱敏输出路径；成功后原子替换目标文件")
    p.add_argument("--mode", choices=["hybrid", "rules"], default="hybrid")
    p.add_argument("--batch-records", type=positive, default=32)
    p.add_argument("--batch-chars", type=positive, default=65536)
    p.add_argument("--max-record-chars", type=positive, default=16384, help="每行字符上限（含换行）；超过则失败")
    p.add_argument("--model-batch-size", "--batch-size", "--batch_size", type=positive, default=8)
    p.add_argument("--threads", type=positive, default=4)
    p.add_argument("--model", default="uie-base")
    p.add_argument("--model-path", type=Path, help="本地 UIE 模型目录；省略时使用 PaddleNLP 缓存")
    p.add_argument("--device", choices=None if directory else ["cpu", "gpu"],
                   default="auto" if directory else "cpu", help="auto/cpu/gpu/cuda:0" if directory else "cpu/gpu")
    p.add_argument("--device-id", type=nonnegative, default=0, help="Paddle 可见 GPU 的逻辑编号")
    p.add_argument("--max-seq-len", type=positive, default=512)
    p.add_argument("--position-prob", type=probability, default=.5)
    p.add_argument("--schema", nargs="+", default=["姓名", "地址", "工作单位"])
    p.add_argument("--name-mask", choices=["full", "keep-first"], default="full", help="姓名默认全遮盖；keep-first 兼容旧版留姓")
    p.add_argument("--config", type=Path, help="规则配置 JSON；不提供则使用内置规则")
    return p


def validate_args(args):
    if args.max_record_chars > args.batch_chars:
        raise ValueError("max-record-chars 不能大于 batch-chars")
    if args.max_seq_len > 512 or args.max_seq_len < max(map(len, args.schema)) + 4:
        raise ValueError("UIE max-seq-len 应大于最长 schema 长度 + 3，且不超过 512")
    if args.mode == "rules" and args.device != "cpu":
        raise ValueError("纯规则模式使用 CPU，请设置 --device cpu")
    if not args.input.is_file():
        raise ValueError("输入文件不存在")


def main():
    p = build_parser()
    args = p.parse_args()
    if args.output is None:
        p.error("请通过 -o/--output 指定脱敏结果路径")
    try:
        validate_args(args)
        from .io import run_file, same_file
        from .pipeline import Pipeline
        if same_file(args.input, args.output) or (args.config and same_file(args.config, args.output)):
            raise ValueError("输出不能覆盖输入或配置文件")
        started = time.perf_counter()
        engine = Pipeline(args)
        initialization = time.perf_counter() - started
        stats = run_file(engine, args, args.output)
        print(json.dumps(dict(device=engine.device, mode=args.mode, init_seconds=initialization, **stats),
                         ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        p.exit(130, "处理已中断；未发布未完成的输出。\n")
    except Exception as error:
        # 不打印异常参数或 traceback，避免模型/解析器把原文带入日志。
        print(f"处理失败（{type(error).__name__}）：请检查输入、配置、长度上限、模型及设备；未发布未完成的输出。", file=sys.stderr)
        raise SystemExit(1)

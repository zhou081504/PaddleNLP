"""PaddleNLP 的唯一适配入口；一个进程只加载一个模型实例。"""
from __future__ import annotations

import os


class UIEDetector:
    def __init__(self, args):
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        import paddle
        from paddlenlp import Taskflow

        if args.device == "gpu":
            if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() <= args.device_id:
                raise RuntimeError("请求的 GPU 不可用，请检查 Paddle GPU 版本、驱动和设备编号")
            expected = f"gpu:{args.device_id}"
        else:
            expected = "cpu"
        paddle.set_device(expected)
        options = dict(
            batch_size=args.model_batch_size,
            num_threads=args.threads,
            max_seq_len=args.max_seq_len,
            position_prob=args.position_prob,
            device_id=-1 if args.device == "cpu" else args.device_id,
        )
        if args.model_path is not None:
            if not args.model_path.is_dir():
                raise ValueError("本地模型目录不存在")
            options["task_path"] = str(args.model_path.resolve())
        self.task = Taskflow("information_extraction", schema=args.schema, model=args.model, **options)
        self.device = paddle.get_device()
        if self.device != expected:
            raise RuntimeError("实际推理设备与请求不一致")
        self._paddle = paddle
        self.measure = not getattr(args, "no_benchmark", False)

    def __call__(self, texts):
        results = self.task(texts)
        if self.measure and self.device.startswith("gpu"):
            # 让推理计时覆盖 GPU 完成时间。
            self._paddle.device.cuda.synchronize()
        return results

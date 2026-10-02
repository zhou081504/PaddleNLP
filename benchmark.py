"""保留单文件稳态压测入口；目录运行的测量由 text_redaction.batch 集成。"""
from benchmarks.run import main, run_benchmark

if __name__ == '__main__':
    raise SystemExit(main())

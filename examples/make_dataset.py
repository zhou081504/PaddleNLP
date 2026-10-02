"""将人工合成 demo 流式扩展为压力测试数据。"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=int, default=10000)
    p.add_argument("-o", "--output", type=Path, default=Path("data/benchmark.txt"))
    args = p.parse_args()
    if args.records < 1:
        p.error("records 必须大于 0")
    samples = (Path(__file__).parent / "demo.txt").read_text(encoding="utf-8").splitlines()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for index in range(args.records):
            stream.write(samples[index % len(samples)] + "\n")
    print(f"已生成 {args.records} 条合成记录")


if __name__ == "__main__":
    main()

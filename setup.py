"""保留本地 PaddleNLP editable 安装，同时打包脱敏业务及性能入口。"""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = (ROOT / "requirements/paddlenlp.txt").read_text().splitlines()
setup(
    name="paddlenlp",
    version="3.0.0b4.post20260921",
    description="PaddleNLP runtime with streaming text redaction and resource benchmarks",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["paddlenlp*", "text_redaction*", "benchmarks*"]),
    include_package_data=True,
    package_data={
        "text_redaction.rules": ["resources/*.json"],
        "paddlenlp": ["**/*.json", "**/*.yaml", "**/*.yml", "**/*.txt", "**/*.jinja", "**/*.model"],
    },
    install_requires=[line for line in requirements if line and not line.startswith("#")] + ["psutil>=5.9,<8"],
    python_requires=">=3.10",
    entry_points={"console_scripts": [
        "text-redact=text_redaction.cli:main",
        "text-redact-benchmark=benchmarks.run:main",
        "paddlenlp=paddlenlp.cli:main",
    ]},
    license_files=["LICENSE"],
)

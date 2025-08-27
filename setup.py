#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA GPU Metrics Logger 安裝腳本
"""

from setuptools import setup, find_packages
import os

# 讀取 README 檔案
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "NVIDIA GPU Metrics Logger - GPU 監控與資料記錄系統"

# 讀取需求檔案
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="nvidia-gpu-metrics-logger",
    version="1.0.0",
    author="GPU Metrics Logger 開發團隊",
    author_email="developer@example.com",
    description="基於 Python 的 NVIDIA GPU 監測系統，能夠持續監控 GPU 使用狀況並記錄至 CSV 檔案",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/nvidia-gpu-metrics-logger",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Hardware",
        "Topic :: System :: Monitoring",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "gpu-monitor=main:main",
            "nvidia-gpu-logger=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.ini"],
    },
    data_files=[
        ("config", ["config/config.ini"]),
    ],
    zip_safe=False,
    keywords="nvidia gpu monitoring metrics logging cuda python",
    project_urls={
        "Bug Reports": "https://github.com/example/nvidia-gpu-metrics-logger/issues",
        "Source": "https://github.com/example/nvidia-gpu-metrics-logger",
        "Documentation": "https://github.com/example/nvidia-gpu-metrics-logger/wiki",
    },
)

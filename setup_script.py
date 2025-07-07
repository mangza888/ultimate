#!/usr/bin/env python3
# setup.py - Ultimate Auto Trading System Setup

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ultimate-auto-trading",
    version="1.0.0",
    author="Ultimate AI Trading Team",
    author_email="support@ultimatetrading.ai",
    description="Ultimate Auto Trading System with AI, RL, and Advanced Backtesting",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ultimate-trading/ultimate-auto-trading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": [
            "torch>=1.12.0+cu117",
            "cupy-cuda11x>=11.0.0",
            "cudf-cu11>=22.08.0",
            "cuml-cu11>=22.08.0",
        ],
        "crypto": [
            "ccxt>=2.0.0",
            "python-binance>=1.0.0",
            "web3>=6.0.0",
        ],
        "cloud": [
            "boto3>=1.24.0",
            "google-cloud-storage>=2.5.0",
            "azure-storage-blob>=12.13.0",
        ],
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.971",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.19.0",
            "pytest-cov>=3.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.14.0",
            "grafana-api>=1.0.0",
            "slack-sdk>=3.18.0",
            "discord-webhook>=1.0.0",
        ],
        "experimental": [
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
            "mlflow>=1.28.0",
            "optuna>=3.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ultimate-trading=main:main",
            "ultimate-train=scripts.train:main",
            "ultimate-backtest=scripts.backtest:main",
            "ultimate-paper-trade=scripts.paper_trade:main",
            "ultimate-config=scripts.config:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
        "config": ["*.yaml", "*.yml"],
        "data": ["*.csv", "*.json"],
        "models": ["*.pkl", "*.pt", "*.pth"],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/ultimate-trading/ultimate-auto-trading/issues",
        "Source": "https://github.com/ultimate-trading/ultimate-auto-trading",
        "Documentation": "https://ultimate-trading.readthedocs.io/",
        "Funding": "https://github.com/sponsors/ultimate-trading",
    },
)
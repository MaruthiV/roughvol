"""
Setup script for roughvol package.
"""

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
    name="roughvol",
    version="1.0.0",
    author="Quantitative Finance Engineer",
    description="A self-contained Python research library for calibrating and testing the rough Bergomi stochastic-volatility model",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/roughvol",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "roughvol=roughvol.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "roughvol": ["config.yaml"],
    },
    keywords="quantitative finance, rough volatility, bergomi, options, calibration, monte carlo",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/roughvol/issues",
        "Source": "https://github.com/yourusername/roughvol",
        "Documentation": "https://github.com/yourusername/roughvol#readme",
    },
) 
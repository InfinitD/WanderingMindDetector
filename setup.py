#!/usr/bin/env python3
"""
Setup script for Cursey package installation.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cursey",
    version="2.0.0",
    author="Cursey Development Team",
    author_email="cursey@example.com",
    description="Multi-Person Face & Eye Tracking System with Detectron2",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/cursey-team/cursey",
    project_urls={
        "Bug Reports": "https://github.com/cursey-team/cursey/issues",
        "Source": "https://github.com/cursey-team/cursey",
        "Documentation": "https://github.com/cursey-team/cursey/blob/main/README.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cursey=examples.detectron_main_app:main",
            "cursey-legacy=examples.main_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cursey": [
            "models/*.xml",
            "*.md",
        ],
    },
    keywords=[
        "computer vision",
        "face detection", 
        "eye tracking",
        "detectron2",
        "pose estimation",
        "gaze tracking",
        "multi-person tracking",
        "real-time",
        "opencv",
        "pytorch",
    ],
    zip_safe=False,
)

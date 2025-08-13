#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read long description from README.md if available
script_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(script_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A comprehensive clustering toolkit with advanced tree cutting, visualization, and network analysis capabilities."

# Extract version from skclust/__init__.py
version = None
with open(os.path.join(script_directory, "skclust", "__init__.py")) as f:
    for line in f:
        if line.strip().startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"').strip("'")
            break

if version is None:
    raise RuntimeError("Unable to find __version__ in skclust/__init__.py")

# Parse requirements.txt
install_requires = []
with open(os.path.join(script_directory, "requirements.txt")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            install_requires.append(line)

setup(
    name="skclust",
    version=version,
    author="Josh L. Espinoza",
    author_email="jol.espinoz@gmail.com",
    description="A comprehensive clustering toolkit with advanced tree cutting and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jolespin/skclust",
    project_urls={
        "Bug Reports": "https://github.com/jolespin/skclust/issues",
        "Source": "https://github.com/jolespin/skclust",
    },
    license="MIT",
    packages=find_packages(),  # finds skclust automatically
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "fast": ["fastcluster>=1.2.0"],
        "tree": ["scikit-bio>=0.5.6"],
        "dynamic": ["dynamicTreeCut>=0.1.0"],
        "network": ["ensemble-networkx>=0.1.0"],
        "all": [
            "fastcluster>=1.2.0",
            "scikit-bio>=0.5.6",
            "dynamicTreeCut>=0.1.0",
            "ensemble-networkx>=0.1.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    keywords="clustering hierarchical-clustering dendrogram tree-cutting machine-learning data-analysis bioinformatics network-analysis visualization scikit-learn",
    zip_safe=False,
)

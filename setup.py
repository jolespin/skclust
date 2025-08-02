#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for skclust package.
"""

import os
import sys
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A comprehensive hierarchical clustering toolkit with advanced tree cutting, visualization, and network analysis capabilities."

# Package metadata
__version__ = "2025.7.26"
__author__ = "Josh L. Espinoza"
__email__ = "jol.espinoz@gmail.com" 

# Core requirements - these are essential for basic functionality
install_requires = [
    'numpy>=1.19.0',
    'pandas>=1.3.0',
    'scipy>=1.7.0',
    'scikit-learn>=1.0.0',
    'matplotlib>=3.3.0',
    'seaborn>=0.11.0',
    'networkx>=2.6.0',
]

# Optional dependencies for enhanced functionality
extras_require = {
    'fast': [
        'fastcluster>=1.2.0',  # Faster linkage computations
    ],
    'tree': [
        'scikit-bio>=0.5.6',  # Tree data structures and Newick export
    ],
    'dynamic': [
        'dynamicTreeCut>=0.1.0',  # Dynamic tree cutting algorithms
    ],
    'network': [
        'ensemble-networkx>=0.1.0',  # Enhanced network analysis
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'black>=21.0.0',
        'isort>=5.0.0',
        'flake8>=3.8.0',
        'mypy>=0.800',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'nbsphinx>=0.8.0',
        'jupyter>=1.0.0',
        'ipykernel>=6.0.0',
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'nbsphinx>=0.8.0',
        'myst-parser>=0.15.0',
    ],
    'examples': [
        'jupyter>=1.0.0',
        'ipykernel>=6.0.0',
        'plotly>=5.0.0',
        'dash>=2.0.0',
    ]
}

# Create 'all' extra that includes everything except dev
extras_require['all'] = list(set().union(
    extras_require['fast'],
    extras_require['tree'],
    extras_require['dynamic'],
    extras_require['network'],
    extras_require['examples'],
))

# Full installation including development tools
extras_require['full'] = list(set().union(
    extras_require['all'],
    extras_require['dev'],
    extras_require['docs'],
))

# Python version requirement
python_requires = '>=3.8'

# Classifiers for PyPI
classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Operating System :: OS Independent',
]

# Keywords for PyPI search
keywords = [
    'clustering',
    'hierarchical-clustering',
    'dendrogram',
    'tree-cutting',
    'machine-learning',
    'data-analysis',
    'bioinformatics',
    'network-analysis',
    'visualization',
    'scikit-learn',
]

setup(
    name='hierarchical-clustering',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='A comprehensive clustering toolkit with advanced tree cutting and visualization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jolespin/skclust',  
    project_urls={
        'Bug Reports': 'https://github.com/jolespin/skclust/issues',
        'Source': 'https://github.com/jolespin/skclust',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    py_modules=['skclust'],  # If it's a single module
    classifiers=classifiers,
    keywords=' '.join(keywords),
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points for command-line tools (if any)
    entry_points={
        'console_scripts': [
            # 'skclust=skclust.cli:main',
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        'hierarchical_clustering': [
            'data/*.csv',
            'data/*.json',
            'examples/*.py',
            'examples/*.ipynb',
        ],
    },
    
    # For data files outside of packages
    # data_files=[
    #     ('share/skclust/examples', ['examples/basic_example.py']),
    # ],
    
    # Testing
    test_suite='tests',
    tests_require=extras_require['dev'],
    
    # Zip safety
    zip_safe=False,
    
    # Minimum setuptools version
    setup_requires=['setuptools>=45', 'wheel'],
)

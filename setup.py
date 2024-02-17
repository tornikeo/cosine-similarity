#!/usr/bin/env python
import os
from setuptools import find_packages, setup


here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "cudams", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="cudams",
    version=version["__version__"],
    description="CUDA-accelerated cosine similarity measure for comparing MS/MS spectra.",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Tornike Onoprishvili",
    author_email="tornikeonoprishvili@gmail.com",
    url="https://github.com/tornikeo/cosine-similarity",
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    test_suite="tests",
    python_requires='>=3.8',
    install_requires=[
        "matchms>=0.23.1",
        "numba",
        "numpy>=1.20.3",
        "pandas",
        "joblib",
        "tqdm",
    ],
    extras_require={"train": ["rdkit"],
                    "dev": ["bump2version",
                            "isort>=4.2.5,<5",
                            "pytest",
                            "pytest-cov",
                            "yapf",],
                    }
)
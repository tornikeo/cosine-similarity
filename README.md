
# CudaMS

CUDA-accelerated mass-spectrum similarity kernels

<a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/samples/colab_tutorial_pesticide.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-quickstart-blue?logo=googlecolab">
</a>

Currently supported similarity kernels:
- `CudaCosineGreedy`, equivalent to [CosineGreedy](https://matchms.readthedocs.io/en/latest/_modules/matchms/similarity/CosineGreedy.html)
- `CudaFingerprintSimilarity`, equivalent to [FingerprintSimilarity](https://matchms.readthedocs.io/en/latest/_modules/matchms/similarity/FingerprintSimilarity.html) (`jaccard`, `cosine`, `dice`)
- More coming soon...

# How to use

The **easiest way** to get started is to use the <a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/samples/colab_tutorial_pesticide.ipynb">colab notebook
</a>  that has everything ready for you. Alternatively do any of the following:

## Run on Colab

### Colab samples:

<p> <a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/samples/upload_your_own_mgf.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-upload_your_mgf-blue?logo=googlecolab">
</a> files and get pairwise similarities quickly.
</p>

<p>
Run the
<a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/performance/default_params_on_colab_t4.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-speed_benchmark-blue?logo=googlecolab">
</a>
and replicate some of our performance results on free Colab GPU hardware.
</p>

<p>
See how accuracy depends on match limit in <a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/accuracy/accuracy_vs_match_limit.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-accuracy_vs_match_limit-blue?logo=googlecolab">
</a>
</p>

## Install locally

In case you experience slow installs, we recommend you switch from `conda` to [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). It is much faster. Total size of the environment will be around 7-8GB.

```bash
# Create clean python environment (we support python versions 3.9 - 3.11)
conda create -n cudams python=3.11 -y
# Install cudatoolkit
conda install nvidia::cuda-toolkit -y
# Install torch
# You **will most likely have to** follow official guide for torch (see here https://pytorch.org/get-started/locally/#start-locally)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install numba (if this fails, follow the official guide https://numba.pydata.org/numba-doc/latest/user/installing.html#installing-using-conda-on-x86-x86-64-power-platforms)
conda install numba -y

# Install matchms (if this fails, again, follow the official guide https://github.com/matchms/matchms?tab=readme-ov-file#installation)
pip install matchms[chemistry]

# Install this repository
pip install git+https://github.com/tornikeo/cosine-similarity
```

## Run in docker



## Run on vast.ai

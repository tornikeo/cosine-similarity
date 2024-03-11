
# CudaMS

CUDA-accelerated mass-spectrum similarity kernels

<a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/samples/colab_tutorial_pesticide.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-quickstart-blue?logo=googlecolab">
</a>


Currently supported similarity kernels:
- [CosineGreedy](https://matchms.readthedocs.io/en/latest/_modules/matchms/similarity/CosineGreedy.html)
- [FingerprintSimilarity](https://matchms.readthedocs.io/en/latest/_modules/matchms/similarity/FingerprintSimilarity.html) (`jaccard`, `cosine`, `dice`)
- More coming soon...

# How to use

The **easiest way** to get started is to use the <a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/samples/colab_tutorial_pesticide.ipynb">colab notebook
</a>  that has everything ready for you. Alternatively do any of the following:

## Run on Colab

### Colab samples:



<a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/samples/upload_your_own_mgf.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-upload_your_mgf-blue?logo=googlecolab">

</a>
Upload your two custom (possibly same) `.mgf` files and their similarities qucikly!

<a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/performance/performance_defaults.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-speed_benchmark-blue?logo=googlecolab">
</a>


Replicate some of our performance results on free Colab GPU hardware!

## Install locally

In case you experience slow installs, we recommend you switch from `conda` to [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). It is much faster. Total size of the environment will be around 7-8GB.

```bash
# Create clean python environment (we support python versions 3.9 - 3.11)
conda create -n cudams python=3.11 -y
# Install cudatoolkit
conda install nvidia::cuda-toolkit -y
# Install torch
# You **will most likley have to** follow official guide for torch (see here https://pytorch.org/get-started/locally/#start-locally)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install numba (if this fails, follow the official gude https://numba.pydata.org/numba-doc/latest/user/installing.html#installing-using-conda-on-x86-x86-64-power-platforms)
conda install numba -y

# Install matchms (if this fails, again, follow the official guide https://github.com/matchms/matchms?tab=readme-ov-file#installation)
pip install matchms[chemistry]

# Install this repository
pip install git+https://github.com/tornikeo/cosine-similarity
```

## Run in docker



## Run on vast.ai


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

Colab samples:

<a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/samples/upload_your_own_mgf.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-upload_your_mgf-blue?logo=googlecolab">
</a>

<a target="_blank" href="https://colab.research.google.com/github/tornikeo/cosine-similarity/blob/main/notebooks/performance/performance_defaults.ipynb">
  <img alt="Static Badge" src="https://img.shields.io/badge/colab-speed_benchmark-blue?logo=googlecolab">
</a>


## Install locally

```bash
# Install cudatoolkit
conda install cudatoolkit
# Install numba
conda install numba
# Install matchms
pip install matchms[chemistry]
# (Optional for very large, sparse outputs) install pytorch
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
# Install this repository
pip install git+https://github.com/tornikeo/cosine-similarity
```

## Run in docker



## Run on vast.ai




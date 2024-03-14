FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
RUN apt-get update && apt-get install -y --no-install-recommends git
COPY requirements.txt .
WORKDIR /workdir

COPY cudams/ cudams/
COPY setup.py .
COPY tests/ tests/
COPY requirements.txt .
COPY pyproject.toml .
COPY notebooks/ notebooks/

RUN pip install -e .[dev]

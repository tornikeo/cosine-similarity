FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
RUN apt-get update && apt-get install -y --no-install-recommends git
COPY requirements.txt .
WORKDIR /workdir
COPY . .
RUN pip install -e .

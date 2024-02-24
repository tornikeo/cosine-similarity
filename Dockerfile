FROM nvidia/cuda:12.3.1-devel-ubuntu20.04
RUN apt-get update && apt-get install -y --no-install-recommends git 
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

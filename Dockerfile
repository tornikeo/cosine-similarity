FROM tensorflow/tensorflow:2.11.0-gpu-jupyter
RUN apt-get update && apt-get install -y --no-install-recommends git 
WORKDIR /tf
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
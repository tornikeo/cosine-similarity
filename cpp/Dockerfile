FROM ubuntu:jammy as build
ENV HOME /root

RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    clang \
    cmake \
    gdb \
    wget \
    xtensor-dev \
    libomp-dev \
    libblas-dev \
    liblapack-dev

WORKDIR /workdir
COPY src src

COPY CMakeLists.txt .
RUN set -ex; \
    mkdir build; \
    cd build;  \
    cmake ..; \
    make;


FROM python:3.11
WORKDIR /workdir
COPY ./requirements.txt /workdir/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --from=build /workdir/build/cosine /workdir/cosine
COPY  server.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]


FROM ubuntu:jammy
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

WORKDIR /workdir/build
CMD ./cosine

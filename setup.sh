#!/bin/bash
rm -rf ./bin \
    && mkdir -p ./bin

ONNX2C_PATH=$(pwd)/bin
ESBMC_PATH=$(pwd)/bin

# Install libs
sudo apt install -y \
  curl wget git \
  build-essential cmake make \
  protobuf-compiler libprotobuf-dev \
  python3 python-pip python-setuptools

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

cd code \
  && poetry config virtualenvs.create true \
  && poetry install --only main --no-interaction --no-ansi -vvv


cd /tmp/

rm -rf /tmp/onnx2c \
 && git clone https://github.com/kraiskil/onnx2c.git \
  && cd onnx2c \
  && git checkout 1d637ab -b build-tmp \
  && git submodule update --init

mkdir build \
  && cd build \
  && cmake .. \
  && make onnx2c \
  && cp onnx2c $ONNX2C_PATH/. \
  && cd /tmp \
  && rm -rf onnx2c \

rm -rf /tmp/esbmc \
  && mkdir -p /tmp/esbmc \
  && cd /tmp/esbmc \
  && wget https://github.com/esbmc/esbmc/releases/download/v7.0/ESBMC-Linux.sh \
  && bash ESBMC-Linux.sh --exclude-subdir --skip-license \
  && cp ./bin/esbmc $ESBMC_PATH/. \
  && rm -rf /tmp/esbmc \
#!/bin/bash

# Configura variables
VENV_DIR="vllm_env"
PYTHON_VERSION="3.12"
MODEL_NAME="google/gemma-3-12b-it"
HOST="0.0.0.0"
PORT="8009"

# Crea y activa el entorno virtual
python$PYTHON_VERSION -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Instala vLLM y Transformers
pip install --upgrade pip
pip install vllm
pip install git+https://github.com/huggingface/transformers.git

vllm serve google/gemma-3-12b-it --host 0.0.0.0 --port 8009 --dtype bfloat16 --max-model-len 15000 --max-num-seqs 3
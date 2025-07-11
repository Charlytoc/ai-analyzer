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

# login in huggingface
# huggingface-cli login

# Download the model locally to 
# huggingface-cli download google/gemma-3-12b-it --local-dir ./server/ai/model/gemma-3-12b-it


# vllm serve google/gemma-3-12b-it --host 0.0.0.0 --port 8009 --dtype bfloat16 --max-model-len 25000 --max-num-seqs 3

# To run locally

# vllm serve ./server/ai/models/gemma3_12b --dtype bfloat16 --max-num-seqs 8 --max-model-len 25000 --host 0.0.0.0 --port 8009


vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-num-seqs 8 --max-model-len 25000 --host 0.0.0.0 --port 8009



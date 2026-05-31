#!/bin/bash
#SBATCH --job-name=fall-detect
#SBATCH --output=fall_detect_%j.log
#SBATCH --error=fall_detect_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=dgxh

# ── Environment ───────────────────────────────────────────────────────────────
# TF 2.17 was compiled against CUDA 12.3 + cuDNN 8.9
module load cuda/12.3
module load cudnn/8.9_cuda12

# Point TF/XLA to the correct CUDA toolkit so ptxas is found
export PATH=/usr/local/cuda-12.3/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-12.3"

# ── Install dependencies ──────────────────────────────────────────────────────
pip install --user --quiet \
    "tensorflow==2.17.0" \
    snowflake-connector-python \
    scikit-learn \
    scipy \
    numpy

# ── Run training ──────────────────────────────────────────────────────────────
python3 hpc_train.py \
    --account   UDLCYTH-GDB50567 \
    --user      martid24 \
    --password  '!999s10puKillerthedog' \
    --warehouse FALL_DETECTION \
    --epochs    100 \
    --batch     128 && echo "Training complete. Models saved to ./models/" || echo "Training FAILED — check .err file"

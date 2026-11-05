#!/bin/bash
#SBATCH --job-name=physjudge_prep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=bf16
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Decode clips, save 16+16 frames per clip, DINOv2 motion score.
set -euo pipefail
export WS=/scratch3/workspace/knaskar_umass_edu-judgecheck HF_HOME=/scratch3/workspace/knaskar_umass_edu-judgecheck/hf
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
hostname; date; module list 2>&1 || true; nvidia-smi --query-gpu=name,memory.total --format=csv
PY=$WS/env/bin/python; which $PY; $PY --version
cd src && $PY extract_frames.py
date

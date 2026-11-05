#!/bin/bash
#SBATCH --job-name=physjudge_qwen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=bf16&vram23
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
# Qwen2.5-VL-7B judge over all (clip, condition) keys; resumable. Pass e.g. "--limit 2" to smoke-test.
set -euo pipefail
export WS=/scratch3/workspace/knaskar_umass_edu-judgecheck HF_HOME=/scratch3/workspace/knaskar_umass_edu-judgecheck/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
hostname; date; module list 2>&1 || true; nvidia-smi --query-gpu=name,memory.total --format=csv
PY=$WS/env/bin/python; which $PY; $PY --version
cd src && $PY judge_qwen.py "$@"
date

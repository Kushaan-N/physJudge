#!/bin/bash
#SBATCH --job-name=physjudge_r2b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-preempt-highvram,gpu-highvram,gpu-preempt,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --exclude=gpu058
# Robustness R2b: rerun the R2 clips on an L40S (same GPU model as the main run).
set -euo pipefail
export WS=/scratch3/workspace/knaskar_umass_edu-judgecheck HF_HOME=/scratch3/workspace/knaskar_umass_edu-judgecheck/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
hostname; date; module list 2>&1 || true; nvidia-smi --query-gpu=name,memory.total --format=csv
cd src
$WS/env_owl/bin/python --version
$WS/env_owl/bin/python judge_autoeval.py --limit 10 --out ../results/robust/rerun_autoeval_l40s.jsonl
$WS/env/bin/python --version
$WS/env/bin/python judge_qwen.py --limit 10 --out ../results/robust/rerun_qwen_l40s.jsonl
date

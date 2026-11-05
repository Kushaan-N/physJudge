#!/bin/bash
#SBATCH --job-name=physjudge_robust
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu,gpu-preempt,gpu-preempt-highvram,gpu-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --exclude=gpu058
# Robustness checks R2-R4 from ROBUSTNESS.md. All outputs resumable.
set -euo pipefail
export WS=/scratch3/workspace/knaskar_umass_edu-judgecheck HF_HOME=/scratch3/workspace/knaskar_umass_edu-judgecheck/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
hostname; date; module list 2>&1 || true; nvidia-smi --query-gpu=name,memory.total --format=csv
cd src
OWL=$WS/env_owl/bin/python; MAIN=$WS/env/bin/python
$OWL --version; $MAIN --version
$OWL judge_autoeval.py --limit 10 --out ../results/robust/rerun_autoeval.jsonl       # R2
$OWL judge_autoeval.py --native32 --out ../results/robust/scores_autoeval_native32.jsonl  # R3
date
$MAIN judge_qwen.py --limit 10 --out ../results/robust/rerun_qwen.jsonl              # R2
$MAIN judge_qwen.py --intro no_order --out ../results/robust/scores_qwen_no_order.jsonl   # R4
date

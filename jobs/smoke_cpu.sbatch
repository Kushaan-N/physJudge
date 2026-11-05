#!/bin/bash
#SBATCH --job-name=physjudge_smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
# CPU smoke test of both judges on 1 plausible + 1 implausible clip (6 calls each).
# Writes to results/smoke_*.jsonl only. 80G: two 7B models loaded in bf16 one after another.
set -euo pipefail
export WS=/scratch3/workspace/knaskar_umass_edu-judgecheck HF_HOME=/scratch3/workspace/knaskar_umass_edu-judgecheck/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
hostname; date; module list 2>&1 || true
cd src
$WS/env_owl/bin/python --version
$WS/env_owl/bin/python judge_autoeval.py --device cpu --limit 1 --out ../results/smoke_autoeval.jsonl
date
$WS/env/bin/python --version
$WS/env/bin/python judge_qwen.py --device cpu --limit 1 --out ../results/smoke_qwen.jsonl
date

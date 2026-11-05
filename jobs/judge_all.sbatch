#!/bin/bash
#SBATCH --job-name=physjudge_all
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu,gpu-preempt,gpu-preempt-highvram,gpu-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --exclude=gpu058
# Both judges back to back on one GPU (the cluster is congested, so one allocation).
# gpu058 is excluded: Blackwell (sm_120) has no kernels in these torch builds.
# Each judge skips keys already in its jsonl, so a preempted/requeued job resumes.
set -euo pipefail
export WS=/scratch3/workspace/knaskar_umass_edu-judgecheck HF_HOME=/scratch3/workspace/knaskar_umass_edu-judgecheck/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
hostname; date; module list 2>&1 || true; nvidia-smi --query-gpu=name,memory.total --format=csv
cd src
which $WS/env/bin/python; $WS/env/bin/python --version
$WS/env/bin/python judge_qwen.py "$@"
date
which $WS/env_owl/bin/python; $WS/env_owl/bin/python --version
$WS/env_owl/bin/python judge_autoeval.py "$@"
date

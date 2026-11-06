#!/bin/bash
#SBATCH --job-name=physjudge_irrev
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu,gpu-preempt,gpu-preempt-highvram,gpu-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=bf16&vram23
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --exclude=gpu058
# Irreversible-action replication (PREREG_IRREVERSIBLE.md): frames, motion, then both
# judges (main settings) and AutoEval's native 32-frame path. All outputs resumable.
set -euo pipefail
export WS=/scratch3/workspace/knaskar_umass_edu-judgecheck HF_HOME=/scratch3/workspace/knaskar_umass_edu-judgecheck/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CLIPS=clips_irreversible.csv MOTION=irreversible/motion.csv
hostname; date; module list 2>&1 || true; nvidia-smi --query-gpu=name,memory.total --format=csv
cd src
OWL=$WS/env_owl/bin/python; MAIN=$WS/env/bin/python
$MAIN --version; $OWL --version
$MAIN extract_frames.py
date
$OWL judge_autoeval.py --out ../results/irreversible/scores_autoeval.jsonl
$OWL judge_autoeval.py --native32 --out ../results/irreversible/scores_autoeval_native32.jsonl
date
$MAIN judge_qwen.py --out ../results/irreversible/scores_qwen.jsonl
date

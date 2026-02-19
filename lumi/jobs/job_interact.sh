#!/bin/bash

# https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs

srun \
  --job-name interact \
  --cpus-per-task 56 \
  --gpus=8 \
  --mem=256G \
  --partition=small-g \
  --account=project_465001752 \
  --time=04:00:00 \
  --pty \
  singularity shell -B /project/project_465001752 /project/project_465002619/nanoproof-container.sif

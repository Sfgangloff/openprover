#!/bin/bash

# https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/container-jobs

srun \
  --job-name interact_cpu \
  --cpus-per-task 8 \
  --mem=16G \
  --partition=small \
  --account=project_465001752 \
  --time=04:00:00 \
  --pty \
  singularity shell -B /project/project_465001752 /project/project_465002619/nanoproof-container.sif

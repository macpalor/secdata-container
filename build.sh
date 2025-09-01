#!/bin/bash
#SBATCH --job-name=build_container
#SBATCH --time=02:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4

# Ensure env.yml is in the current directory
if [ ! -f env.yml ]; then
  echo "Error: env.yml not found in current directory"
  exit 1
fi

apptainer build secdata_container.sif secdata_container.def
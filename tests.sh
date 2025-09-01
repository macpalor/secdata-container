#!/bin/bash
#SBATCH --job-name=test-secdata-container
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100-80g
#SBATCH --time=00:10:00

SIF=secdata_container.sif

# Iterate over all .py files in the tests/ directory
for test_script in tests/*.py; do
    echo "Testing ${test_script}................"
    singularity run --nv --net --network none \
    $SIF \
    ${test_script}
done
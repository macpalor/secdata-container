#!/bin/bash
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100-80g
#SBATCH --time=02:00:00

SIF=secdata_container.sif

# Iterate over all .py files in the tests/ directory
for test_script in tests/*.py; do
    echo "Testing ${test_script}................"
    singularity run --nv --net --network none \
    #--bind /scratch/shareddata/dldata/huggingface-hub-cache:/models/huggingface-hub \
    $SIF \
    ${test_script}
done
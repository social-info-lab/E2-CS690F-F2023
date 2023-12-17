#!/bin/bash
#SBATCH --partition=cpu-long
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=BEUSDT-Query-Generation-GOSDT
#SBATCH --mem=256G
#SBATCH --output=./out/query-gen-aug-gosdt.%j.out
#SBATCH --error=./out/query-gen-aug-gosdt.%j.err
module load miniconda/22.11.1-1
conda activate gosdt
python query_gen_scripts/boolean_generation_gosdt.py
#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=fsl_firstLevel_log.txt
#SBATCH --error=fsl_firstLevel_error.err
#SBATCH --job-name=fsl_1st
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruonan.jia@yale.edu

echo "Running script"
module load miniconda
module load FSL
source activate py37_dev

python /home/rj299/project/mdm_analysis/mdm_imaging_analysis/fsl_glm_firstlevel.py

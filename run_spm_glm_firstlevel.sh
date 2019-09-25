#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=spm_1st_log_%J.txt
#SBATCH --error=spm_1st_error_%J.err
#SBATCH --job-name=spm_1st
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruonan.jia@yale.edu

module load miniconda
module load FSL
module load MATLAB/2018b
source activate py37_dev

python /home/rj299/project/mdm_analysis/mdm_imaging_analysis/spm_glm_firstlevel.py

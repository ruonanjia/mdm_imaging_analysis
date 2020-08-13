#!/bin/bash
#SBATCH --partition general
#SBATCH --nodes 1
#SBATCH --time 16:00:00
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-log-%J.txt

## get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    MacOS or linus terminal command to create your ssh tunnel:
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.hpc.yale.edu
    -----------------------------------------------------------------
    MobaXterm info:
    Forwarded port: same as remote port
    Remote server:${node}
    Remote port: ${port}
    SSH server: ${cluster}.hpc.yale.edu
    SSH login: $user
    SSH port: 22
    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:${port}  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

## start an ipcluster instance and launch jupyter server
## module load brainiak/0.8-Python-Anaconda3
## module load nilearn/0.5.0-Python-Anaconda3
## module load OpenMPI/2.1.2-GCC-6.4.0-2.28
module load miniconda
source activate py37_dev

jupyter-notebook --no-browser --port=${port} --ip=${node}

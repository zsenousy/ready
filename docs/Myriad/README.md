# MYRIAD - FEDERATED LEARNING

Instructions for connecting and training models for federated Learning with Myriad HPC Cluster

## Features
* HD: 5TB
* GPU: Tesla V100-PCIE-32GB or NVIDIA A100 80GB 
* STORAGE: send all datasets to Scratch and then run from there - /home/ccxxxxx/Scratch/scratch/ccxxxxx/datasets/

## Connect to Myriad
1. Connect to `vpn.ucl.ac.uk` using cisco https://www.ucl.ac.uk/isd/services/get-connected/ucl-virtual-private-network-vpn
2. Connect to the server
```bash
SSH TO MYRIAD
ssh ccxxxxx@myriad.rc.ucl.ac.uk
```

## ENVIRONMENT SETUP - IN ORDER

### CREATE VIRTUAL ENVIRONMENT
python3 -m venv $HOME/ready/venv_ready


### ACTIVATE VIRTUAL ENVIRONMENT
source /home/ccxxxxx/ready/venv_ready/bin/activate


### Load modules
* module unload compilers mpi gcc-libs
* module load gcc-libs/10.2.0
* module load python3/3.9-gnu-10.2.0
* module load pytorch/2.1.0/gpu
* pip3 install --user ".[dev]" --no-deps

### SET PYTHON PATH
export PYTHONPATH=/home/ccxxxxx/ready/venv_ready/lib/python3.9/site-packages:/home/ccxxxxx/.python3local/lib/python3.9/site-packages:/home/ccxxxxx/ready/src:$PYTHONPATH


### INSTALL READY PACKAGES
python -m pip install matplotlib scikit-image scikit-learn loguru omegaconf flwr "urllib3<2" tifffile imageio numpy==1.26.4

## DATASET PATHS
* MOBIOUS: /home/ccxxxxx/Scratch/scratch/ccxxxxx/datasets/mobious/mobious/MOBIOUS/train100per_1144
* OpenEDS: /home/ccxxxxx/Scratch/scratch/ccxxxxx/datasets/ready/ready/openEDS/openEDS/
* RIT-EYES: /home/ccxxxxx/Scratch/scratch/ccxxxxx/datasets/s-natural/s-natural/s-natural
* FEDERATED-WEIGHTS : /home/ccxxxxx/Scratch/scratch/ccxxxxx/datasets/ready/ready/federated
* INFERENCE: /home/ccxxxxx/Scratch/scratch/ccxxxxx/datasets/ready/ready/inference/inference_results
* GLOBAL MODEL: /home/ccxxxxx/Scratch/scratch/ccxxxxx/datasets/mobious/models/15-Jul-2026_12-01-38_cpu

## TRAINING FEDERATED ON MYRIAD
qsub run_federated.sh

## JOB SCRIPT (run_federated.sh)
* #!/bin/bash -l
* #$ -l h_rt=24:0:0
* #$ -l mem=16G
* #$ -l gpu=1
* #$ -N federated_unet
* #$ -cwd
* #$ -o federated_output_$JOB_ID.log
* #$ -e federated_error_$JOB_ID.log

* module unload compilers mpi gcc-libs
* module load gcc-libs/10.2.0
* module load python3/3.9-gnu-10.2.0
* module load pytorch/2.1.0/gpu

* export PYTHONPATH=/home/ccxxxxx/ready/venv_ready/lib/python3.9/site-packages:/home/ccxxxxx/.python3local/lib/python3.9/site-packages:/home/ccaxxxxx/ready/src:$PYTHONPATH

* source /home/ccxxxx/ready/venv_ready/bin/activate

* cd $HOME/ready

* python src/ready/apis/train_federated.py

## INFERENCE ON LOGIN NODE
* module unload compilers mpi gcc-libs
* module load gcc-libs/10.2.0
* module load python3/3.9-gnu-10.2.0
* module load pytorch/2.1.0/gpu
* export PYTHONPATH=/home/ccxxxxx/ready/venv_ready/lib/python3.9/site-packages:/home/ccxxxxx/.python3local/lib/python3.9/site-packages:/home/ccxxxxx/ready/src:$PYTHONPATH
* source /home/ccxxxxx/ready/venv_ready/bin/activate
* cd $HOME/ready
* python src/ready/apis/inference_federated.py -c configs/federated/config_inference_federated.yaml

## INFERENCE AS A JOB
qsub run_inference_federated.sh


### JOB SCRIPT (run_inference_federated.sh)
* #!/bin/bash -l
* #$ -l h_rt=06:0:0
* #$ -l mem=10G
* #$ -l gpu=1
* #$ -N inference_federated
* #$ -cwd
* #$ -o inference_output.log
* #$ -e inference_error.log

* module unload compilers mpi gcc-libs
* module load gcc-libs/10.2.0
* module load python3/3.9-gnu-10.2.0
* module load pytorch/2.1.0/gpu

* export PYTHONPATH=/home/ccxxxxx/ready/venv_ready/lib/python3.9/site-packages:/home/ccxxxxx/.python3local/lib/python3.9/site-packages:/home/ccxxxxx/ready/src:$PYTHONPATH

* source /home/ccxxxxx/ready/venv_ready/bin/activate

* cd $HOME/ready

* python src/ready/apis/inference_federated.py -c configs/federated/config_inference_federated.yaml

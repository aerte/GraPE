#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J adv
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=24GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u email
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o adv_%J.out
#BSUB -e adv_%J.err
# -- end of LSF options --

### nvidia-smi
source /zhome/4a/a/156124/GraPE/genv/bin/activate

python /zhome/4a/a/156124/GraPE/notebooks/optimization_test.py logp --samples 500
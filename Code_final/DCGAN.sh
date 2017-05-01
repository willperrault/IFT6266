# Example script of how to submit jobs on hades

#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -l nodes=1:ppn=1
#PBS -r n

module add python/3.5.1
module add openblas/0.2.18
module add CUDA/7.5
source p3.5/bin/activate

cd ~/Code
python Autoencoder.py

#######################
# Commands for hades ##
#######################

# To request in interactive mode:
# qsub -I -l nodes=1:ppn=1 -l walltime=02:00:00

# Copy repertory from computer to hades:
# scp -rp /Users/williamperrault/Github/H2017/IFT6266/Code/ ift6ed51@hades.calculquebec.ca:

# Submit job in the queue:
# qsub -q @hades DCGAN.sh

# Send file from hades to computer:
# scp ift6ed51@hades.calculquebec.ca:monfichier .

# scp norm_data_valid1.npy ift6ed51@hades.calculquebec.ca:Data/
# scp Autoencoder.py ift6ed51@hades.calculquebec.ca:Data/

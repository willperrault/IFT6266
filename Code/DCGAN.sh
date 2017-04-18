#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -l nodes=1:ppn=1
#PBS -r n
#PBS -l mem=10gb

source activate py35

#scp -rp /Users/williamperrault/Github/H2017/IFT6266/Code/. ift6ed51@hades.calculquebec.ca:

cd ~/Calcul_quebec
python DCGAN.py

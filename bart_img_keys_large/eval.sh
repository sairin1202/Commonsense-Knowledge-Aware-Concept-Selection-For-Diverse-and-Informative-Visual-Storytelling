#!/bin/bash
#$ -N eval
#$ -l rt_G.small=1
#$ -l h_rt=3:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
source ~/nlp/bin/activate
module load python/3.6/3.6.5
module load cuda/10.2/10.2.89

python -u evaluation.py
#qsub -g gcb50169 -l RESOURCE_TYPE=NUM_RESOURCE BATCH_FILE

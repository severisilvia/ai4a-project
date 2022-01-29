#!/bin/bash
#SBATCH --job-name=m3
#SBATCH --output=/homes/lbaraldi/m3_log
#SBATCH --error=/homes/lbaraldi/m3_log
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --partition=students-prod
#SBATCH --mail-user=lorenzo.baraldi@unimore.it
#SBATCH --mail-type=ALL

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate m3

cd /homes/lbaraldi/m3

export MASTER_ADDR="aimagelab-srv-00"
export MASTER_PORT=`comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export WORLD_SIZE=2
export LOCAL_RANK=0

export RANK=0
srun -N1 -n1 -w $MASTER_ADDR --gpus=1 --exclusive python -u train.py <args> &
sleep 5

for i in {1..1}; do
	  export RANK=$i
	  srun -N1 -n1 --gpus=1 --exclusive python -u train.py <args> &
done
wait


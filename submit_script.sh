#!/bin/bash
#SBATCH --job-name=open_clip
#SBATCH --account=<proj>
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --mem=448GB
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --partition=<partition>
#SBATCH --signal=USR2@120
#SBATCH --time=1440
#SBATCH --wckey=submitit
#SBATCH --output=slurm_output/%j_0_log.out
#SBATCH --error=slurm_output/%j_0_log.err


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12802

export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --unbuffered --output slurm_output/%j_%t_log.out --error slurm_output/%j_%t_log.err \
python -u src/training/main.py \
    --save-frequency 1 \
    --report-to wandb \
    --train-data="datasets/img2dataset/cc12m/{00000..00621}.tar" \
    --warmup 2000 \
    --batch-size=512 \
    --epochs=10 \
    --workers=7 \
    --model ViT-B-16 \
    --pretrained laion2b_s34b_b88k \
    --name "cc12m" \
    --seed 0 \
    --log-every-n-steps 2 \
    --loss_type main \
    --train-num-samples 8000000 \
     --local-loss \
     --gather-with-grad
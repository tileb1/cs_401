#!/bin/bash
#SBATCH --job-name=small_process_shards
#SBATCH --account=<project>
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=448GB
#SBATCH --ntasks-per-node=56
#SBATCH --nodes=<nodes>
#SBATCH --open-mode=append
#SBATCH --partition=<partition>
#SBATCH --time=180
#SBATCH --output=slurm_output/%j_0_log.out
#SBATCH --error=slurm_output/%j_0_log.err

# taskset -c 2 python make_shards_spacy.py --id 1 &
# taskset -c 3 python make_shards_spacy.py --id 2 &
# taskset -c 4 python make_shards_spacy.py --id 3 &
# taskset -c 5 python make_shards_spacy.py --id 4 &
# taskset -c 6 python make_shards_spacy.py --id 5 &
# taskset -c 7 python make_shards_spacy.py --id 6 &
# taskset -c 9 python make_shards_spacy.py --id 7 &
# taskset -c 10 python make_shards_spacy.py --id 8 &
# taskset -c 11 python make_shards_spacy.py --id 9 &
# taskset -c 12 python make_shards_spacy.py --id 10 &
# taskset -c 13 python make_shards_spacy.py --id 11 &
# taskset -c 14 python make_shards_spacy.py --id 12 &
# taskset -c 15 python make_shards_spacy.py --id 13 &
# taskset -c 17 python make_shards_spacy.py --id 14 &
# taskset -c 18 python make_shards_spacy.py --id 15 &
# taskset -c 19 python make_shards_spacy.py --id 16 &
# taskset -c 20 python make_shards_spacy.py --id 17 &
# taskset -c 21 python make_shards_spacy.py --id 18 &
# taskset -c 22 python make_shards_spacy.py --id 19 &
# taskset -c 23 python make_shards_spacy.py --id 20 &
# taskset -c 25 python make_shards_spacy.py --id 21 &
# taskset -c 26 python make_shards_spacy.py --id 22 &
# taskset -c 27 python make_shards_spacy.py --id 23 &
# taskset -c 28 python make_shards_spacy.py --id 24 &
# taskset -c 29 python make_shards_spacy.py --id 25 &
# taskset -c 30 python make_shards_spacy.py --id 26 &
# taskset -c 31 python make_shards_spacy.py --id 27 &
# taskset -c 33 python make_shards_spacy.py --id 28 &
# taskset -c 34 python make_shards_spacy.py --id 29 &
# taskset -c 35 python make_shards_spacy.py --id 30 &
# taskset -c 36 python make_shards_spacy.py --id 31 &
# taskset -c 37 python make_shards_spacy.py --id 32 &
# taskset -c 38 python make_shards_spacy.py --id 33 &
# taskset -c 39 python make_shards_spacy.py --id 34 &
# taskset -c 41 python make_shards_spacy.py --id 35 &
# taskset -c 42 python make_shards_spacy.py --id 36 &
# taskset -c 43 python make_shards_spacy.py --id 37 &
# taskset -c 44 python make_shards_spacy.py --id 38 &
# taskset -c 45 python make_shards_spacy.py --id 39 &
# taskset -c 46 python make_shards_spacy.py --id 40 &
# taskset -c 47 python make_shards_spacy.py --id 41 &
# taskset -c 49 python make_shards_spacy.py --id 42 &
# taskset -c 50 python make_shards_spacy.py --id 43 &
# taskset -c 51 python make_shards_spacy.py --id 44 &
# taskset -c 52 python make_shards_spacy.py --id 45 &
# taskset -c 53 python make_shards_spacy.py --id 46 &
# taskset -c 54 python make_shards_spacy.py --id 47 &
# taskset -c 55 python make_shards_spacy.py --id 48 &
# taskset -c 57 python make_shards_spacy.py --id 49 &
# taskset -c 58 python make_shards_spacy.py --id 50 &
# taskset -c 59 python make_shards_spacy.py --id 51 &
# taskset -c 60 python make_shards_spacy.py --id 52 &
# taskset -c 61 python make_shards_spacy.py --id 53 &
# taskset -c 62 python make_shards_spacy.py --id 54 &
# taskset -c 63 python make_shards_spacy.py --id 55 &

# taskset -c 1 python make_shards_spacy.py --id 0 && sleep 5m

srun python src/training/make_shards_spacy.py
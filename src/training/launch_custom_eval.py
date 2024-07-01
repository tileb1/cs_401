import subprocess
import os


def launch_custom_eval(
    checkpoint_path, model="ours_dinov2_ViT-B-14_reg", partition="small-g"
):
    script = """#!/bin/bash
#SBATCH --job-name=__eval__
#SBATCH --account=project_465000727
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --mem=464GB
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --partition={partition}
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --time=180
#SBATCH --output={output}
#SBATCH --error={error}


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=localhost
export MASTER_PORT=12804
export PYTHONPATH="$PYTHONPATH:/scratch/project_465000727/repos/Contextual-CLIP/src"

srun --unbuffered --output {output} --error {error} /scratch/project_465000727/images/pytorch_mmseg/bin/python src/evaluations/main_eval.py --model {model} --checkpoint_path {checkpoint_path} --dist_url {dist_url}"""

    def submit_job(formatted_script):
        TMP_SUBMISSION_PATH = "/tmp/tmp_submission.sh"
        with open(TMP_SUBMISSION_PATH, "w") as f:
            f.write(formatted_script)
        subprocess.run("sbatch {}".format(TMP_SUBMISSION_PATH), shell=True)

    formatted_script = script.format(
        model=model,
        checkpoint_path=checkpoint_path,
        dist_url="env://",
        partition=partition,
        output=str(os.path.join(checkpoint_path, "eval.out")),
        error=str(os.path.join(checkpoint_path, "eval.err")),
    )
    print(formatted_script)
    submit_job(formatted_script)


if __name__ == "__main__":
    pass

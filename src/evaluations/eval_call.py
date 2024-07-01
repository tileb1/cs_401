from training.distributed import init_distributed_device
import argparse
import subprocess
import os


def main(args):
    # Init
    _ = init_distributed_device(args)

    # Call the evaluation script
    python_path = "/scratch/project_465000727/thomas_envs/pytorch_mmseg/bin/python"
    script_path = "/pfs/lustrep4/scratch/project_465000727/thomas_repos/Contextual-CLIP_dev/src/evaluations/main_eval.py"
    command = [python_path, script_path]
    needed_args = ["config", "dist_url", "model", "checkpoint_path", "untar_path"]
    dict_args = vars(args)
    for arg in needed_args:
        if arg == "dist_url":
            command.extend(
                [
                    f"--{arg}",
                    "file:///pfs/lustrep4/scratch/project_465000727/thomas_repos/Contextual-CLIP_dev/init",
                ]
            )
        else:
            command.extend([f"--{arg}", str(dict_args[arg])])
    subprocess.run(command, env=os.environ, shell=False)


def parse_args():
    parser = argparse.ArgumentParser(description="mmseg test (and eval) a model")
    parser.add_argument("--config", default="ours.yaml", help="config file path")

    # Added by us
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ours_dinov2_ViT-B-14_reg",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/scratch/project_465000727/epoch_8.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--untar_path",
        type=str,
        default="/dev/shm",
        # default=None,
        help="Path where to untar.",
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

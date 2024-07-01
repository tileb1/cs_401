# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import uuid
from pathlib import Path
import pickle
import copy
import subprocess

GRIDSEARCH_DICT_LST = [{}]
import socket
import time


import submitit
from training.my_utils.run_manager import get_all_args_lst, get_parser, LOGS_ROOT_PATH


def parse_args():
    return get_parser().parse_args()


def get_shared_folder(args) -> Path:
    p = Path(os.path.join(LOGS_ROOT_PATH, args.name))
    p.mkdir(exist_ok=True)
    return p


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(args)), exist_ok=True)
    init_file = get_shared_folder(args) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.initial_args = copy.deepcopy(args)
        self.args = args
        self.job_releaser = False
        self.p = None
        self.p_pinner = None

    def __call__(self):
        import torch
        import time

        try:
            self.call()
        except RuntimeError as e:
            print(str(e))
            if (
                "Caught collective operation timeout" in str(e)
                or "Timed out initializing process" in str(e)
                or "DDP expects same model across all ranks" in str(e)
            ):
                # Timed out initializing process group in store based barrier on rank: 0, for key: store_based_barrier_key:1 (world_size=8, worker_count=16, timeout=0:15:00)
                self.args.dist_url = self.args.dist_url + "r"
                self.initial_args.dist_url = self.initial_args.dist_url + "r"
                print(
                    "torch.distributed.is_nccl_available():",
                    torch.distributed.is_nccl_available(),
                )
                torch.distributed.destroy_process_group()

                print(
                    "torch.distributed.is_nccl_available():",
                    torch.distributed.is_nccl_available(),
                )
                time.sleep(10)
                self.__call__()
            elif "miopenStatusInternalError" in str(e):
                time.sleep(10)
                self.__call__()
            else:
                raise e

    def call(self):
        import multiprocessing

        if submitit.JobEnvironment().global_rank == 0:
            if not self.job_releaser:
                self.job_releaser = True
        else:
            import time

            time.sleep(5)
        # TODO: remove this
        import signal

        signal.signal(signal.SIGBUS, self.signal_handler)
        # TODO: end
        import shutil
        import sys
        import training.main as main_cclip
        import os
        import multiprocessing
        import torch

        print("torch.distributed.is_initialized():", torch.distributed.is_initialized())

        if submitit.JobEnvironment().global_rank == 0:
            rank_path = os.path.join(
                self.args.output_dir,
                "global_rank_lst_{}".format(submitit.JobEnvironment().job_id),
            )
            if os.path.exists(rank_path):
                shutil.rmtree(rank_path)
            os.mkdir(rank_path)

        if "dodrio" in socket.gethostname():
            os.environ["NCCL_LL_THRESHOLD"] = "0"
        # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ["NCCL_IB_TIMEOUT"] = "24"

        self._setup_gpu_args()

        # Crash directly if error with MIOpen
        print("Will resnet now")
        import torchvision.models

        test_model = torchvision.models.__dict__["resnet50"]().cuda(self.args.gpu)
        print(test_model(torch.rand(4, 3, 224, 224).cuda(self.args.gpu)))

        print("DEFAULT AFFINITY")
        print("Affinity main process:", os.sched_getaffinity(0))
        print("multiprocessing.cpu_count():", multiprocessing.cpu_count())
        print("torch.get_num_threads():", torch.get_num_threads())

        if len(os.sched_getaffinity(0)) != self.args.cpu_per_task:
            first_cpu = list(sorted(os.sched_getaffinity(0)))[0]
            self.args.cpus = [
                i for i in range(first_cpu, first_cpu + self.args.cpu_per_task)
            ]
            os.system(
                "taskset -p -c %s %d" % (",".join(str(i) for i in self.args.cpus), 0)
            )
            print("AFTER CHANGES AFFINITY")
            print("Affinity main process:", os.sched_getaffinity(0))
            print("multiprocessing.cpu_count():", multiprocessing.cpu_count())
            print("torch.get_num_threads():", torch.get_num_threads())

        if self.args.run_training:
            print("#" * 50)
            print(os.environ)
            print("#" * 50)
            # Training
            main_cclip.main(self.args)

        # Needed for error handling/rescheduling
        Path(os.path.join(self.args.output_dir, "output_knn")).mkdir(
            exist_ok=True, parents=True
        )

        keys = [k for k in os.environ.keys()]
        for k in keys:
            del os.environ[k]

        if self.args.rank == 0:
            from training.launch_custom_eval import launch_custom_eval

            launch_custom_eval(
                checkpoint_path=os.path.join(
                    self.args.logs, self.args.name, "checkpoints"
                ),
                model=self.args.model,
                partition=self.args.slurm_partition,
            )

        print("#" * 50)
        print("End of submitit reached.")
        print("#" * 50)
        if self.p is not None:
            self.p.kill()
        if self.p_pinner is not None:
            self.p_pinner.kill()
        print("Killed handler process (if any).")

    def checkpoint(self):
        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.initial_args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def signal_handler(self, sig, frame):
        import submitit
        import datetime
        import time

        job_env = submitit.JobEnvironment()

        # Log when SIGBUS signal is caught (only for analysing crashes)
        with open(
            os.path.join(
                self.args.output_dir,
                "log_sigbus_{}_{}.txt".format(job_env.global_rank, job_env.job_id),
            ),
            "w",
        ) as f:
            f.write(
                "SIGBUS CAUGHT at {date} with {env}\n".format(
                    date=datetime.datetime.now(), env=job_env
                )
            )

        # Keep track of processes that caught the bus error
        with open(
            os.path.join(
                self.args.output_dir,
                "global_rank_lst_{}".format(submitit.JobEnvironment().job_id),
                str(job_env.global_rank),
            ),
            "w",
        ) as f:
            f.write("{}\n".format(job_env.global_rank))

        # Wait for all crashing processes to log their rank
        time.sleep(3)

        # Check if you are the lowest rank process and reschedule job
        if job_env.global_rank == min(
            [
                int(i)
                for i in os.listdir(
                    os.path.join(
                        self.args.output_dir,
                        "global_rank_lst_{}".format(submitit.JobEnvironment().job_id),
                    )
                )
            ]
        ):
            return_code = 1
            while return_code == 1:
                out = subprocess.run(
                    "scancel --signal=USR2 $SLURM_JOB_ID",
                    shell=True,
                    env=dict(os.environ),
                )
                return_code = out.returncode
                time.sleep(60)

        # Prevent other processes to alt the signaling of USR2
        time.sleep(600)
        raise RuntimeError("Job didnt get rescheduled but there was a SIGBUS")

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(
            str(self.args.output_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main(args, bad_nodes=""):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor_args = {"folder": args.output_dir, "slurm_max_num_timeout": 30}
    executor = submitit.AutoExecutor(**executor_args)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    kwargs = {}

    if args.comment:
        kwargs["slurm_comment"] = args.comment
    if args.slurm_exclusive is not None and args.ngpus == 8:
        kwargs["slurm_exclusive"] = True

    additional_slurm_params = {"account": args.slurm_account}
    if args.slurm_reservation is not None:
        additional_slurm_params["reservation"] = args.slurm_reservation
    if args.afterany:
        additional_slurm_params["dependency"] = "afterany:{}".format(args.afterany)
    if args.slurm_hint is not None:
        additional_slurm_params["hint"] = args.slurm_hint

    if args.slurm_exclude is None:
        tmp_lst = []
    else:
        tmp_lst = args.slurm_exclude.split(",")

    if bad_nodes is None:
        bad_lst = []
    else:
        bad_lst = bad_nodes.split(",")

    for n in bad_lst:
        if n not in tmp_lst:
            tmp_lst.append(n)

    final_bad_nodes = ",".join(tmp_lst)

    if len(final_bad_nodes) > 0:
        additional_slurm_params["exclude"] = final_bad_nodes

    mem = args.gb_ram_per_gpu * num_gpus_per_node
    executor.update_parameters(
        mem_gb=mem,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpu_per_task,
        nodes=nodes,
        slurm_qos=args.slurm_qos,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=args.slurm_partition,
        slurm_signal_delay_s=120,
        slurm_additional_parameters=additional_slurm_params,
        **kwargs,
    )

    name = args.output_dir.split("/")[-1]
    executor.update_parameters(name=name)

    args.dist_url = get_init_file(args).as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")
    print("Running with following args:")
    print(args, end="\n\n")


if __name__ == "__main__":
    # Set to False for conventional run
    # This setup is made for managing runs more easily

    FETCH_FROM_GOOGLE_SHEETS_BOOL = True
    if FETCH_FROM_GOOGLE_SHEETS_BOOL:
        args_lst = get_all_args_lst()
        folder_args = os.path.join(LOGS_ROOT_PATH, "args_lst")
        Path(folder_args).mkdir(parents=True, exist_ok=True)

        GS_args_lst = []
        # Running in gridsearch mode
        for GRIDSEARCH_DICT in GRIDSEARCH_DICT_LST:
            # The sheet should only have a single line which is the base config
            if len(GRIDSEARCH_DICT) > 0 and len(args_lst) == 1:
                base_args = copy.deepcopy(args_lst[0])
                base_dict = vars(base_args)
                output_dir = base_args.output_dir

                def nameify(o):
                    if type(o) == tuple or type(o) == list:
                        return "".join([str(i) for i in o])
                    return str(o)

                import itertools

                for elem in itertools.product(
                    *[[(k, elem) for elem in v] for (k, v) in GRIDSEARCH_DICT.items()]
                ):
                    base_dict["output_dir"] = (
                        output_dir
                        + "_GS_"
                        + "_".join([str(i[0]) + nameify(i[1]) for i in elem])
                    )
                    for k, v in elem:
                        base_dict[k] = v

                    gs_args = argparse.Namespace(**base_dict)
                    GS_args_lst.append(gs_args)
                    main(gs_args)

            # Running in standard mode from the sheet
            else:
                for index, args in enumerate(args_lst):
                    main(args)

                with open(
                    os.path.join(
                        folder_args, "args_lst_{}.pkl".format(uuid.uuid4().hex)
                    ),
                    "wb",
                ) as handle:
                    pickle.dump(
                        {"time": time.time(), "args_lst": args_lst},
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                for GRIDSEARCH_DICT in GRIDSEARCH_DICT_LST:
                    if len(GRIDSEARCH_DICT) > 0:
                        print("#" * 20)
                        print(
                            "len(GRIDSEARCH_DICT) > 0 but running in standard mode because multiple lines are defined in sheet."
                        )
                        print("#" * 20)
        if len(GS_args_lst) > 0:
            with open(
                os.path.join(folder_args, "args_lst_{}.pkl".format(uuid.uuid4().hex)),
                "wb",
            ) as handle:
                pickle.dump(
                    {"time": time.time(), "args_lst": GS_args_lst},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    else:
        args = parse_args()
        main(args)

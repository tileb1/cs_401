import os
import shutil
import socket
import time
import subprocess
import random
from training.my_utils.run_manager import LOGS_ROOT_PATH
from collections import defaultdict, Counter
import json
import sys
import pickle


def signal_handler(args, sig, frame):
    pass


def pin_workers_iterator(the_iterator, args):
    if "karolina" in socket.gethostname() or "ristoteles" in socket.gethostname():
        return
    try:
        print(args.cpus)
    except AttributeError:
        args.cpus = list(sorted(os.sched_getaffinity(0)))
        # os.system("taskset -p -c %d %d" % ((args.cpus[0]), os.getpid()))

    if args.num_workers > 0:
        for index, w in enumerate(the_iterator._workers):
            os.system(
                "taskset -p -c %d %d"
                % ((args.cpus[(index + 1) % len(args.cpus)]), w.pid)
            )


def clean_on_leave(args):
    if args.untar_path[:8] == "/dev/shm" and int(args.gpu) == 0:
        shutil.rmtree(args.untar_path)


def clear_shm():
    folder = "/dev/shm"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass


def get_args_lst(args):
    folder = os.path.join(LOGS_ROOT_PATH, "args_lst")
    current = {"time": 0, "args_lst": []}
    for f in os.listdir(folder):
        with open(os.path.join(folder, f), "rb") as handle:
            out = pickle.load(handle)
            for a in out["args_lst"]:
                if a.output_dir == args.output_dir and current["time"] < out["time"]:
                    current = out
    return current["args_lst"]

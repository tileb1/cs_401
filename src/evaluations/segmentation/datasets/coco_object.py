# ------------------------------------------------------------------------------
# Modified from GroupViT (https://github.com/NVlabs/GroupViT)
# Copyright (c) 2021-22, NVIDIA Corporation & affiliates. All Rights Reserved.
# ------------------------------------------------------------------------------
from mmseg.datasets import DATASETS, CustomDataset
import os
import time
import tarfile
import torch.distributed as dist


@DATASETS.register_module()
class COCOObjectDataset(CustomDataset):
    """COCO-Object dataset.

    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    CLASSES = (
        "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )

    PALETTE = [
        [0, 0, 0],
        [0, 192, 64],
        [0, 192, 64],
        [0, 64, 96],
        [128, 192, 192],
        [0, 64, 64],
        [0, 192, 224],
        [0, 192, 192],
        [128, 192, 64],
        [0, 192, 96],
        [128, 192, 64],
        [128, 32, 192],
        [0, 0, 224],
        [0, 0, 64],
        [0, 160, 192],
        [128, 0, 96],
        [128, 0, 192],
        [0, 32, 192],
        [128, 128, 224],
        [0, 0, 192],
        [128, 160, 192],
        [128, 128, 0],
        [128, 0, 32],
        [128, 32, 0],
        [128, 0, 128],
        [64, 128, 32],
        [0, 160, 0],
        [0, 0, 0],
        [192, 128, 160],
        [0, 32, 0],
        [0, 128, 128],
        [64, 128, 160],
        [128, 160, 0],
        [0, 128, 0],
        [192, 128, 32],
        [128, 96, 128],
        [0, 0, 128],
        [64, 0, 32],
        [0, 224, 128],
        [128, 0, 0],
        [192, 0, 160],
        [0, 96, 128],
        [128, 128, 128],
        [64, 0, 160],
        [128, 224, 128],
        [128, 128, 64],
        [192, 0, 32],
        [128, 96, 0],
        [128, 0, 192],
        [0, 128, 32],
        [64, 224, 0],
        [0, 0, 64],
        [128, 128, 160],
        [64, 96, 0],
        [0, 128, 192],
        [0, 128, 160],
        [192, 224, 0],
        [0, 128, 64],
        [128, 128, 32],
        [192, 32, 128],
        [0, 64, 192],
        [0, 0, 32],
        [64, 160, 128],
        [128, 64, 64],
        [128, 0, 160],
        [64, 32, 128],
        [128, 192, 192],
        [0, 0, 160],
        [192, 160, 128],
        [128, 192, 0],
        [128, 0, 96],
        [192, 32, 0],
        [128, 64, 128],
        [64, 128, 96],
        [64, 160, 0],
        [0, 64, 0],
        [192, 128, 224],
        [64, 32, 0],
        [0, 192, 128],
        [64, 128, 224],
        [192, 160, 0],
    ]

    def __init__(self, untar_path, local_rank, **kwargs):
        # Untar
        data_root = kwargs["data_root"]
        data_root = self.untar(untar_path, data_root, local_rank)
        kwargs["data_root"] = data_root
        dist.barrier()

        super(COCOObjectDataset, self).__init__(
            img_suffix=".jpg", seg_map_suffix="_instanceTrainIds.png", **kwargs
        )

    def untar(self, untar_path, data_root, local_rank):
        if not data_root.endswith(".tar"):
            return data_root

        # Untar the dataset
        if untar_path[0] == "$":
            untar_path = os.environ[untar_path[1:]]
        start_copy_time = time.time()

        new_data_root = os.path.join(untar_path, "coco_stuff164k")
        if not os.path.exists(new_data_root) and local_rank == 0:
            with tarfile.open(data_root, "r") as f:
                f.extractall(untar_path)
        print(f"Time taken to untar to {untar_path}:", time.time() - start_copy_time)

        # Wait
        time.sleep(5)

        # Update the data root dir
        data_root = new_data_root
        return data_root

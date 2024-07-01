import os.path as osp

from mmseg.datasets import DATASETS, CustomDataset

import os
import time
import tarfile
import torch.distributed as dist


@DATASETS.register_module(force=True)
class PascalContextDataset(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = (
        "background",
        "aeroplane",
        "bag",
        "bed",
        "bedclothes",
        "bench",
        "bicycle",
        "bird",
        "boat",
        "book",
        "bottle",
        "building",
        "bus",
        "cabinet",
        "car",
        "cat",
        "ceiling",
        "chair",
        "cloth",
        "computer",
        "cow",
        "cup",
        "curtain",
        "dog",
        "door",
        "fence",
        "floor",
        "flower",
        "food",
        "grass",
        "ground",
        "horse",
        "keyboard",
        "light",
        "motorbike",
        "mountain",
        "mouse",
        "person",
        "plate",
        "platform",
        "potted plant",
        "road",
        "rock",
        "sheep",
        "shelves",
        "sidewalk",
        "sign",
        "sky",
        "snow",
        "sofa",
        "table",
        "track",
        "train",
        "tree",
        "truck",
        "tvmonitor",
        "wall",
        "water",
        "window",
        "wood",
    )

    PALETTE = [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
    ]

    def __init__(self, untar_path, local_rank, split, **kwargs):
        # Untar
        data_root = kwargs["data_root"]
        data_root = self.untar(untar_path, data_root, local_rank)
        kwargs["data_root"] = data_root
        dist.barrier()

        super(PascalContextDataset, self).__init__(
            img_suffix=".jpg",
            seg_map_suffix=".png",
            split=split,
            reduce_zero_label=False,
            **kwargs,
        )
        assert osp.exists(self.img_dir) and self.split is not None

    def untar(self, untar_path, data_root, local_rank):
        if not data_root.endswith(".tar"):
            return data_root

        # Untar the dataset
        if untar_path[0] == "$":
            untar_path = os.environ[untar_path[1:]]
        start_copy_time = time.time()

        new_data_root = os.path.join(untar_path, "PContext/VOCdevkit/VOC2010")
        if not os.path.exists(new_data_root) and local_rank == 0:
            with tarfile.open(data_root, "r") as f:
                f.extractall(untar_path)
        print(f"Time taken to untar to {untar_path}:", time.time() - start_copy_time)

        # Wait
        time.sleep(5)

        # Update the data root dir
        data_root = new_data_root
        return data_root


@DATASETS.register_module(force=True)
class PascalContextDataset59(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = (
        "aeroplane",
        "bag",
        "bed",
        "bedclothes",
        "bench",
        "bicycle",
        "bird",
        "boat",
        "book",
        "bottle",
        "building",
        "bus",
        "cabinet",
        "car",
        "cat",
        "ceiling",
        "chair",
        "cloth",
        "computer",
        "cow",
        "cup",
        "curtain",
        "dog",
        "door",
        "fence",
        "floor",
        "flower",
        "food",
        "grass",
        "ground",
        "horse",
        "keyboard",
        "light",
        "motorbike",
        "mountain",
        "mouse",
        "person",
        "plate",
        "platform",
        "potted plant",
        "road",
        "rock",
        "sheep",
        "shelves",
        "sidewalk",
        "sign",
        "sky",
        "snow",
        "sofa",
        "table",
        "track",
        "train",
        "tree",
        "truck",
        "tvmonitor",
        "wall",
        "water",
        "window",
        "wood",
    )

    PALETTE = [
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
    ]

    def __init__(self, untar_path, local_rank, split, **kwargs):
        # Untar
        data_root = kwargs["data_root"]
        data_root = self.untar(untar_path, data_root, local_rank)
        kwargs["data_root"] = data_root
        dist.barrier()

        super(PascalContextDataset59, self).__init__(
            img_suffix=".jpg",
            seg_map_suffix=".png",
            split=split,
            reduce_zero_label=True,
            **kwargs,
        )
        assert osp.exists(self.img_dir) and self.split is not None

    def untar(self, untar_path, data_root, local_rank):
        if not data_root.endswith(".tar"):
            return data_root

        # Untar the dataset
        if untar_path[0] == "$":
            untar_path = os.environ[untar_path[1:]]
        start_copy_time = time.time()

        new_data_root = os.path.join(untar_path, "PContext/VOCdevkit/VOC2010")
        if not os.path.exists(new_data_root) and local_rank == 0:
            with tarfile.open(data_root, "r") as f:
                f.extractall(untar_path)
        print(f"Time taken to untar to {untar_path}:", time.time() - start_copy_time)

        # Wait
        time.sleep(5)

        # Update the data root dir
        data_root = new_data_root
        return data_root

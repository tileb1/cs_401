# ------------------------------------------------------------------------------
import os.path as osp

from mmseg.datasets import CustomDataset
from mmseg.datasets import DATASETS
import os
import tarfile
import time
import torch.distributed as dist


@DATASETS.register_module()
class PascalVOCDataset20(CustomDataset):
    """Pascal VOC dataset (the background class is ignored).

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    PALETTE = [
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]

    def __init__(self, untar_path, local_rank, split, **kwargs):
        # Untar
        data_root = kwargs["data_root"]
        data_root = self.untar(untar_path, data_root, local_rank)
        kwargs["data_root"] = data_root
        dist.barrier()

        super(PascalVOCDataset20, self).__init__(
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

        new_data_root = os.path.join(untar_path, "VOC12/VOCdevkit/VOC2012")
        if not os.path.exists(new_data_root) and local_rank == 0:
            with tarfile.open(data_root, "r") as f:
                f.extractall(untar_path)
        print(f"Time taken to untar to {untar_path}:", time.time() - start_copy_time)

        # Wait
        time.sleep(5)

        # Update the data root dir
        data_root = new_data_root
        return data_root

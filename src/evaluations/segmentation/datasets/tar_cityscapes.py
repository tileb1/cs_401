import os.path as osp

from mmseg.datasets import CityscapesDataset
from mmseg.datasets import DATASETS
import os
import tarfile
import time
import torch.distributed as dist


@DATASETS.register_module()
class TarCityscapesDataset(CityscapesDataset):
    def __init__(self, untar_path, local_rank, **kwargs):
        # Untar
        data_root = kwargs["data_root"]
        data_root = self.untar(untar_path, data_root, local_rank)
        kwargs["data_root"] = data_root
        dist.barrier()

        super(TarCityscapesDataset, self).__init__(**kwargs)

    def untar(self, untar_path, data_root, local_rank):
        if not data_root.endswith(".tar"):
            return data_root

        # Untar the dataset
        if untar_path[0] == "$":
            untar_path = os.environ[untar_path[1:]]
        start_copy_time = time.time()

        new_data_root = os.path.join(untar_path, "cityscapes")
        if not os.path.exists(new_data_root) and local_rank == 0:
            with tarfile.open(data_root, "r") as f:
                f.extractall(untar_path)
        print(f"Time taken to untar to {untar_path}:", time.time() - start_copy_time)

        # Wait
        time.sleep(5)

        # Update the data root dir
        data_root = new_data_root
        return data_root

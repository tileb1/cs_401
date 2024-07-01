import tempfile
import warnings

import mmcv
import numpy as np
import torch

# from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import pickle
import tempfile
from typing import Optional

import torch
import torch.distributed as dist
import time
from einops import rearrange


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False, dir=tmpdir
        ).name
    np.save(temp_file_name, array)
    return temp_file_name


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda"
    )
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[: shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None


def multi_gpu_test(
    model,
    data_loader,
    tmpdir=None,
    gpu_collect=False,
    efficient_test=False,
    pre_eval=False,
    format_only=False,
    format_args={},
):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            "DeprecationWarning: ``efficient_test`` will be deprecated, the "
            "evaluation is CPU memory friendly with pre_eval=True"
        )
        mmcv.mkdir_or_exist(".efficient_test")
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, (
        "``efficient_test``, ``pre_eval`` and ``format_only`` are mutually "
        "exclusive, only one of them could be true ."
    )

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            b = len(result)
            batch_indices *= b

        if efficient_test:
            result = [np2tmp(_, tmpdir=".efficient_test") for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args
            )
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size // b
            batch_size = len(result) * world_size // b
            for _ in range(batch_size):
                prog_bar.update()

    # Combine everything in a single tensor [(b * n_threshold), n_metrics, n_classes]
    results_tensor = torch.stack(
        [torch.stack(im_results) for im_results in results]
    ).cuda()

    # Gather across GPU
    gathered_results_tensor = [
        torch.zeros_like(results_tensor) for _ in range(world_size)
    ]
    dist.all_gather(gathered_results_tensor, results_tensor)
    gathered_results_tensor = torch.stack(gathered_results_tensor)

    if b == 1:
        gathered_results_tensor = rearrange(
            gathered_results_tensor, "g b m c -> (g b) m c"
        )
        return gathered_results_tensor.cpu()

    # Process as desired
    gathered_results_tensor = rearrange(
        gathered_results_tensor,
        "g (b t) m c -> t (g b) m c",
        t=len(model.background_thresholds),
    )
    dict_results = {
        k: v.cpu()
        for k, v in zip(model.background_thresholds, gathered_results_tensor.unbind())
    }
    return dict_results

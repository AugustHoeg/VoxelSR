import monai
from monai.data import SmartCacheDataset, CacheDataset

class Dataset:
    def __init__(self, opt, apply_split=True):
        self.opt = opt
        self.apply_split = apply_split
        self.dataset_opt = opt['datasets']
        self.dataset_name = self.dataset_opt['name']
        self.dataset_type = self.dataset_opt['dataset_type']
        self.dataset_params_train = self.dataset_opt['train']['dataset_params']
        self.dataset_params_test = self.dataset_opt['test']['dataset_params']
        self.dataset = self.get_dataset_class()

    def get_dataset_class(self):
        if self.dataset_name == "Synthetic_2022_QIM_52_Bone":
            from data.Dataset_2022_QIM_52_Bone import Dataset_2022_QIM_52_Bone as D
        elif self.dataset_name == "2022_QIM_52_Bone":
            from data.Dataset_2022_QIM_52_Bone import Dataset_2022_QIM_52_Bone as D
        elif self.dataset_name == "HCP_1200":
            from data.Dataset_HCP_1200 import Dataset_HCP_1200 as D
        elif self.dataset_name == "IXI":
            from data.Dataset_IXI import Dataset_IXI as D
        elif self.dataset_name == "BRATS2023":
            from data.Dataset_BRATS2023 import Dataset_BRATS2023 as D
        elif self.dataset_name == "KIRBY21":
            from data.Dataset_KIRBY21 import Dataset_KIRBY21 as D
        else:
            raise NotImplementedError(f'Dataset {self.dataset_name} not implemented.')
        # Initialize dataset
        dataset = D(self.opt)
        return dataset

    def get_files(self):
        train_files, test_files = self.dataset.get_file_paths()
        return train_files, test_files, self.dataset.data_path

    def get_transforms(self):
        transforms = self.dataset.get_transforms(mode="train")
        test_transforms = self.dataset.get_transforms(mode="test")
        baseline_transforms = self.dataset.get_baseline_transforms(mode="test")
        return transforms, test_transforms, baseline_transforms

    def define_datasets(self):
        train_files, test_files, data_path = self.get_files()
        transforms, test_transforms, baseline_transforms = self.get_transforms()

        dataset_mapping = {
            "MonaiSmartCacheDataset": SmartCacheDataset,
            "CacheDataset": CacheDataset,
            "MonaiDataset": monai.data.Dataset,
            "MonaiIterableDataset": monai.data.IterableDataset,
            "MonaiCacheDataset": monai.data.CacheDataset
        }

        if self.dataset_type not in dataset_mapping:
            raise NotImplementedError(f'Dataset type {self.dataset_type} is not found.')

        DatasetType = dataset_mapping[self.dataset_type]

        train_dataset = DatasetType(train_files, transform=transforms, **self.dataset_params_train)
        test_dataset = DatasetType(test_files, transform=test_transforms, **self.dataset_params_test)
        baseline_dataset = monai.data.Dataset(test_files, transform=baseline_transforms)

        print(f'Dataset {self.dataset_name} of type {self.dataset_type} initialized.')
        return train_dataset, test_dataset, baseline_dataset


import torch.distributed as dist
from monai.data import CacheDataset, partition_dataset


class DDP_CacheDataset(CacheDataset):
    """CacheDataset for DistributedDataParallel training.
    From this discussion "I solved this issue by setting Trainer(use_distributed_sampler=False) in lightning 2.0+ and using a manually split CacheDataset."
    https://github.com/Lightning-AI/pytorch-lightning/discussions/11763
    """

    def __init__(self, data_list: list, transform, cache_rate: float = 0, num_workers: int = 0):
        """
        Args:
            data: data to cache.
            transform: transform to apply to data.
            cache_rate: cache rate for data.
            num_workers: number of workers for data.
        """
        if dist.is_initialized():
            part_data = self._get_part_data_list(data_list)
            print(
                f"Data count: {len(part_data)} for local rank {dist.get_rank()}, world size: {dist.get_world_size()}, total number: {len(data_list)}")
        else:
            part_data = data_list

        super().__init__(part_data,
                         transform,
                         cache_rate=cache_rate,
                         num_workers=num_workers,
                         copy_cache=False,
                         as_contiguous=True,
                         )

    def _get_part_data_list(self, data_list: list) -> list:
        """
        Get part of data list for each rank.
        Args:
            data_list: list of data.
        """
        return partition_dataset(
            data=data_list,
            num_partitions=dist.get_world_size(),
            shuffle=False,
            seed=42,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]


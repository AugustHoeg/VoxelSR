import os
import copy
import glob
import queue
from tqdm import tqdm
import sys
import random
import numpy as np
import torch
import monai.data
import monai.transforms as mt
import zarr
from zarr.storage import LRUStoreCache, FSStore, MemoryStore, DirectoryStore
from ome_zarr.io import parse_url
from monai.data import SmartCacheDataset, DataLoader, IterableDataset
from time import sleep
from time import perf_counter as time
from multiprocessing import Process, Queue, Event
from queue import Empty
#from torch.multiprocessing import Process, Queue, Event
from threading import Thread


def extract_patch(data, group_name, ome_level, patch_size=(32, 32, 32)):
    # TODO: Fix this method
    # We start with the first level
    volume = data[group_name][ome_level]
    start = np.random.randint(0, np.array(volume.shape) - patch_size)  # (0,0,0)
    end = start + patch_size

    patch = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    out_dict = {ome_level: patch}
    return out_dict

def extract_patch_levels(data, group_pair, patch_size=(32, 32, 32), f=4, metadata=None):
    volume_L = data[group_pair['L']]
    volume_H = data[group_pair['H']]

    patch_size_hr = np.multiply(patch_size, f)
    valid_shape = np.maximum(np.subtract(volume_L.shape, patch_size), (1, 1, 1))
    d0, h0, w0 = np.random.randint(0, valid_shape)
    d1, h1, w1 = np.add((d0, h0, w0), patch_size)

    patch_L = volume_L[d0:d1, h0:h1, w0:w1]
    patch_L = np.pad(patch_L, ((0, patch_size[0] - patch_L.shape[0]),
                               (0, patch_size[1] - patch_L.shape[1]),
                               (0, patch_size[2] - patch_L.shape[2])))

    patch_H = volume_H[d0*f:d1*f, h0*f:h1*f, w0*f:w1*f]
    patch_H = np.pad(patch_H, ((0, patch_size_hr[0] - patch_H.shape[0]),
                               (0, patch_size_hr[1] - patch_H.shape[1]),
                               (0, patch_size_hr[2] - patch_H.shape[2])))

    out_dict = {'L': patch_L, 'H': patch_H}

    if metadata:
        for key, val in metadata.items():
            out_dict[key] = val

    return out_dict

def extract_patch_levels_prealloc(data, group_pair, patch_size=(32, 32, 32), f=4, metadata=None):
    volume_L = data[group_pair['L']]
    volume_H = data[group_pair['H']]

    patch_size_hr = np.multiply(patch_size, f)
    patch_L = np.zeros(patch_size)
    patch_H = np.zeros(patch_size_hr)

    valid_shape = np.maximum(np.subtract(volume_L.shape, patch_size), (1, 1, 1))
    d0, h0, w0 = np.random.randint(0, valid_shape)
    d1, h1, w1 = np.add((d0, h0, w0), patch_size)

    D0, H0, W0 = np.multiply((d0, h0, w0), f)
    D1, H1, W1 = np.multiply((d1, h1, w1), f)

    data_L = volume_L[d0:d1, h0:h1, w0:w1]
    data_H = volume_H[D0:D1, H0:H1, W0:W1]

    patch_L[:data_L.shape[0], :data_L.shape[1], :data_L.shape[2]] = data_L
    patch_H[:data_H.shape[0], :data_H.shape[1], :data_H.shape[2]] = data_H

    out_dict = {'L': patch_L, 'H': patch_H}

    if metadata:
        for key, val in metadata.items():
            out_dict[key] = val

    return out_dict


def extract_patch_levels_from_chunk(data, group_name, ome_levels, patch_size=(32, 32, 32)):
    # TODO: Fix this method
    volume = data[group_name][ome_levels[-1]]
    c_shape = volume.cdata_shape
    c_idx = tuple(np.random.randint(c_shape))

    valid_range_lr = np.array(volume.chunks) - patch_size + 1
    start = np.random.randint(0, valid_range_lr)
    end = start + patch_size

    patch = volume.blocks[c_idx][start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    out_dict = {ome_levels[-1]: patch}

    for level in ome_levels[:-1]:
        level_diff = int(ome_levels[-1]) - int(level)
        start = start * 2 ** level_diff
        end = end * 2 ** level_diff
        volume = data[group_name][level]
        patch = volume.blocks[c_idx][start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        out_dict[level] = patch

    return out_dict


class ZarrProducer():
    def __init__(self, worker_data, patch_shape, patch_transform, up_factor, queue_size: int = 64, num_workers: int = 1, sampling_method='random', seed=8338):
        super().__init__()

        # Define data
        self.worker_data = worker_data
        self.patch_shape = patch_shape
        self.patch_transform = patch_transform
        self.up_factor = up_factor

        self.num_workers = num_workers
        self.queues = []  # Each worker will have its own queue
        for _ in range(num_workers):
            self.queues.append(Queue(maxsize=queue_size))

        self.workers = []
        self.stop_event = Event()  # Event to signal workers to stop

        self.seed = seed

        if sampling_method == 'in_chunk':
            self._sample_data = extract_patch_levels_from_chunk
        else:
            self._sample_data = extract_patch_levels

    def _worker_process(self, id):

        self._set_random_seed(self.seed + id)  # Set random seed for each worker
        # print("Worker seed set to: ", self.seed + id)

        w = [self.worker_data[key]['sampling_weight'] for key in self.worker_data]  # sampling weights
        p = w / np.sum(w)
        while not self.stop_event.is_set():
            name = np.random.choice(list(self.worker_data.keys()), p=p)
            z = random.choice(self.worker_data[name]['zarr_data'])  # Randomly select a zarr file in dataset
            group_pair = random.choice(self.worker_data[name]['group_pairs'][f'{self.up_factor}'])  # random group pair
            patch = self._sample_data(z, group_pair, self.patch_shape, metadata=None)  # metadata={'name': name, 'group_pair': group_pair})
            if self.patch_transform:
                patch = self.patch_transform(patch)
            try:
                self.queues[id].put(patch)  # block for time out space is available
            except queue.Full:
                sleep(0.2)  # Sleep for a short time if the queue is full
                continue

    def get(self):
        while True:
            for queue in self.queues:
                try:
                    return queue.get_nowait()
                except Empty:
                    continue
            sleep(0.001)  # Sleep for 1 ms to avoid CPU starvation

    def set_workers(self):

        for id in range(self.num_workers):
            worker = Thread(target=self._worker_process, args=(id,))
            #worker = Process(target=self._worker_process, args=(id,))
            #worker.daemon = True
            self.workers.append(worker)

    def start_workers(self):
        # Start worker processes
        for worker in self.workers:
            worker.start()

        print(f"Started Producer with {self.num_workers} worker(s)")

    def stop_workers(self):
        # Stop the worker processes by setting stop event
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=2)

    def _set_random_seed(self, seed):
        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        monai.utils.misc.set_determinism(seed)
        np.random.RandomState(seed)


class ZarrIterableDataset(IterableDataset):

    def __init__(self, dataset_dict, patch_shape, patch_transform, up_factor, num_workers, queue_size, base_seed=8338, store_type='Numpy', num_samples=1000, sampling_method='random', print_metadata=False):
        self.dataset_dict = dataset_dict
        self.patch_shape = patch_shape
        self.patch_transform = patch_transform
        self.up_factor = up_factor
        self.num_workers = num_workers  # Number of worker processes per queue
        self.queue_size = queue_size
        self.base_seed = base_seed
        self.num_samples = num_samples
        self.producer = None
        self.sampling_method = sampling_method  # Method to sample patches, e.g., 'random', 'in_chunk'
        self.store_type = store_type
        self.print_metadata = print_metadata  # Print metadata of the Zarr group

        self.dataset_names = list(dataset_dict.keys())

        self.paths = []
        for key in dataset_dict.keys():
            self.paths += dataset_dict[key]['paths']
        super().__init__(self.paths, patch_transform)

        # Check if the paths are valid
        for path in self.paths:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist.")

        if sampling_method == 'in_chunk':
            self._sample_data = extract_patch_levels_from_chunk
        else:
            self._sample_data = extract_patch_levels

    def _load_data(self, worker_id, num_workers):

        worker_data = copy.deepcopy(self.dataset_dict)
        for name, dataset in self.dataset_dict.items():
            paths = dataset['paths'][worker_id::num_workers]
            if paths:
                worker_data[name]['paths'] = dataset['paths'][worker_id::num_workers]
            else:
                del worker_data[name]
                continue

            worker_data[name]['zarr_data'] = []  # Create field for zarr file handles

            for path in paths:
                if dataset['store_type'] == 'Numpy':
                    # TODO: fix NumPy method here.
                    data = zarr.open(path, mode='r', cache_attrs=True)
                    z = {self.group_name: {level: np.array(data[self.group_name][level]) for level in self.ome_levels}}
                elif dataset['store_type'] == 'MemoryStore':
                    disk_store = DirectoryStore(path)
                    memory_store = MemoryStore()
                    zarr.copy_store(disk_store, memory_store)
                    z = zarr.open(memory_store, mode='r')
                elif dataset['store_type'] == 'LRUStoreCache':
                    store_size = 2 ** 28  # 256 MB
                    cached_store = LRUStoreCache(FSStore(path), max_size=store_size)
                    z = zarr.open(store=cached_store, mode='r', cache_attrs=True)
                else:
                    z = zarr.open(path, mode='r', cache_attrs=True)

                # TODO fix check for in-chunk sampling
                #if self.sampling_method == 'in_chunk' and self.store_type != 'Numpy':
                #    self._assert_chunk_sampling(z, self.patch_shape)

                worker_data[name]['zarr_data'].append(z)

                if self.print_metadata:
                    store = parse_url(path, mode="r").store
                    root = zarr.group(store=store)

                    print(root.info)  # Print the metadata of the Zarr group
                    print(root.tree())  # Print the structure of the Zarr group

        return worker_data

    def _assert_chunk_sampling(self, root, patch_shape):
        # TODO: Fix this method
        chunk_shape = root[self.group_name][self.ome_levels[-1]].chunks
        if any(ps > cs for ps, cs in zip(patch_shape, chunk_shape)):
            raise ValueError(
                f"Patch shape {patch_shape} is larger than chunk shape {chunk_shape} in {root.store.path}.")

    def _init_producer(self, id, worker_data):

        self.producer = ZarrProducer(worker_data=worker_data,
                                     patch_shape=self.patch_shape,
                                     patch_transform=self.patch_transform,
                                     up_factor=self.up_factor,
                                     queue_size=self.queue_size,
                                     num_workers=self.num_workers,
                                     sampling_method=self.sampling_method,
                                     seed=self.base_seed + id*self.num_workers)  # Use a different seed for each producer
        self.producer.set_workers()
        self.producer.start_workers()

        # wait for all queues to fill up
        print(f"Waiting for producer queues to be {50}% full...", end='\r')
        while np.sum([queue.qsize() for queue in self.producer.queues]) < int(self.queue_size * self.num_workers) // 2:
            sleep(1)
        # print(f"Waiting for producer queues to be {100}% full...")
        # while self.producer.queue.qsize() < int(self.queue_size):
        #     continue

    # def stop_workers(self):
    #     # Stop the worker processes by setting stop event
    #     self.stop_event.set()
    #     for worker in self.workers:
    #         worker.join(timeout=2)


    def _generate_patch(self, worker_data):

        # Sample random dataset, volume and group pair
        w = [worker_data[key]['sampling_weight'] for key in worker_data]  # sampling weights
        p = w / np.sum(w)
        name = np.random.choice(list(worker_data.keys()), p=p)  # random name
        z = random.choice(worker_data[name]['zarr_data'])  # Randomly select a zarr file in dataset
        group_pair = random.choice(worker_data[name]['group_pairs'][f'{self.up_factor}'])  # random group pair

        # Extract a patch from the selected dataset
        patch = self._sample_data(z, group_pair, self.patch_shape, metadata=None)  # metadata={'name': name, 'group_pair': group_pair})

        if self.patch_transform:
            patch = self.patch_transform(patch)

        return patch

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_id = 0
            num_workers = 1
            samples_per_worker = self.num_samples
        else:  # in a worker process
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            samples_per_worker = int(np.ceil(self.num_samples / float(worker_info.num_workers)))

        # Load data
        worker_data = self._load_data(worker_id, num_workers)

        if self.producer is None and self.num_workers > 0:
            self._init_producer(worker_id, worker_data)

        # Generate patches
        if self.num_workers == 0:
            for _ in range(samples_per_worker):
                yield self._generate_patch(worker_data)
        else:
            for _ in range(samples_per_worker):
                yield self.producer.get()  # block for time out space is available

    def __len__(self):
        # Return a large number to ensure the dataset is infinite
        return self.num_samples


def main():

    # Example usage
    batch_size = 8
    patch_shape = (32, 32, 32)

    HCP_1200_train_paths = glob.glob("../../Vedrana_master_project/3D_datasets/datasets/HCP_1200/ome/train/*.zarr")
    HCP_1200_test_paths = glob.glob("../../Vedrana_master_project/3D_datasets/datasets/HCP_1200/ome/test/*.zarr")

    IXI_train_paths = glob.glob("../../Vedrana_master_project/3D_datasets/datasets/IXI/ome/train/*.zarr")
    IXI_test_paths = glob.glob("../../Vedrana_master_project/3D_datasets/datasets/IXI/ome/test/*.zarr")

    dataset_dict = {
        "HCP_1200": {
            "paths": HCP_1200_train_paths,
            "group_pairs": {
                "4": [{"H": "HR/1", "L": "HR/3"}],  # {"H": "HR/1", "L": "HR/3"}
            },
            "sampling_weight": 1,
            "store_type": "DirectoryStore"
        },
        "IXI": {
            "paths": IXI_train_paths,
            "group_pairs": {
                "4": [{"H": "HR/1", "L": "HR/3"}],  # {"H": "HR/1", "L": "HR/3"}
            },
            "sampling_weight": 1,
            "store_type": "DirectoryStore"
        }
    }

    seed = 8883
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define patch transforms
    patch_transform = mt.Compose([
        mt.Identityd(keys=['H', 'L'], allow_missing_keys=True),
        mt.EnsureChannelFirstd(keys=['H', 'L'], channel_dim='no_channel'),
        # mt.SignalFillEmptyd(keys=['H', 'L'], replacement=0),  # Remove any NaNs
        # mt.ScaleIntensityd(keys=ome_levels, minv=0.0, maxv=1.0),
        # #mt.Rand3DElasticd(keys=ome_levels, prob=0.5, sigma_range=(5, 10), magnitude_range=(0.1, 0.2), mode='bilinear'),
        #mt.RandFlipd(keys=['H', 'L'], prob=0.5, spatial_axis=0),
        #mt.RandFlipd(keys=['H', 'L'], prob=0.5, spatial_axis=1),
        #mt.RandFlipd(keys=['H', 'L'], prob=0.5, spatial_axis=2)
    ])

    dataset = ZarrIterableDataset(dataset_dict,
                                  patch_shape,
                                  patch_transform,
                                  up_factor=4,
                                  num_workers=4,
                                  queue_size=128,
                                  store_type='DirectoryStore',
                                  num_samples=1000,
                                  sampling_method='random'  # 'random' or 'in_chunk'
                                  )

    num_workers = 4
    persistent_workers = True if num_workers > 0 else False
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=False,
                                            persistent_workers=persistent_workers)

    no_epochs = 10
    plot_counter = 0
    plot_interval = 1000000
    start_time = time()
    for i in range(no_epochs):
        print(f"Epoch {i + 1}/{no_epochs}")
        for batch in tqdm(dataloader, desc='Reconstructing patches\n', mininterval=2):
            # for batch in dataloader:
            pass
            # sleep(0.1)  # Assuming some processing time
            # print("Loaded batch...")
            # for key in batch.keys():
            #     print(f"Key: {key}, Shape: {batch[key].shape}")
            if plot_counter % plot_interval == 0:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(batch['L'][0, 0, 16, :, :])
                plt.subplot(1, 2, 2)
                plt.imshow(batch['H'][0, 0, 64, :, :])
                plt.show()
                plot_counter = 0
            plot_counter += 1


    time_elapsed = time() - start_time
    print(f"Time taken {time_elapsed} sec.")
    print(f"Time taken per patch {time_elapsed / no_epochs / batch_size} sec. (average)")

    print(f"Loaded {len(dataset)} items from Zarr dataset.")


if __name__ == "__main__":
    main()
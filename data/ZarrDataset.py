import os
import queue
from tqdm import tqdm
import sys
import random
import numpy as np
import torch
import monai.data
import monai.transforms as mt
import zarr
from zarr.storage import LRUStoreCache, FSStore
from ome_zarr.io import parse_url
from monai.data import SmartCacheDataset, DataLoader
from time import sleep
from time import perf_counter as time
from multiprocessing import Process, Queue, Event


class ZarrProducer():
    def __init__(self, zarr_data, group_name, ome_levels, patch_shape, patch_transform, queue_size: int = 64, num_workers: int = 1, seed=8338):
        super().__init__()

        # Define data
        self.zarr_data = zarr_data
        self.patch_shape = patch_shape
        self.patch_transform = patch_transform

        # Each worker will have its own queue
        self.queue = Queue(maxsize=queue_size)
        self.num_workers = num_workers
        self.workers = []
        self.stop_event = Event()  # Event to signal workers to stop

        self.ome_levels = ome_levels  # levels in the OME-Zarr dataset
        self.group_name = group_name  # Name of the group in the Zarr file

        self.seed = seed

    def _worker_process(self, id):

        self.set_random_seed(self.seed + id)  # Set random seed for each worker
        # print("Worker seed set to: ", self.seed + id)

        while not self.stop_event.is_set():
            z = random.choice(self.zarr_data)  # Randomly select a zarr dataset
            patch = self._extract_patch_levels(z, self.patch_shape)
            if self.patch_transform:
                patch = self.patch_transform(patch)
            try:
                self.queue.put(patch)  # block for time out space is available
            except queue.Full:
                sleep(0.2)  # Sleep for a short time if the queue is full
                continue

    def _extract_patch(self, data, patch_size=(32, 32, 32)):

        # We start with the first level
        volume = data[self.group_name][self.ome_levels[0]]
        start = np.random.randint(0, np.array(volume.shape) - patch_size)  # (0,0,0)
        end = start + patch_size

        patch = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        out_dict = {self.ome_levels[0]: patch}
        return out_dict

    def _extract_patch_levels(self, data, patch_size=(32, 32, 32)):

        volume = data[self.group_name][self.ome_levels[-1]]
        start = np.random.randint(0, np.array(volume.shape) - patch_size)
        end = start + patch_size
        out_dict = {self.ome_levels[-1]: volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]}

        for i in range(len(self.ome_levels) - 2, -1, -1):  # reverse order
            volume = data[self.group_name][self.ome_levels[i]]
            start = start * 2
            end = end * 2
            out_dict[self.ome_levels[i]] = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        return out_dict

    def set_workers(self):

        for id in range(self.num_workers):
            worker = Process(target=self._worker_process, args=(id,))
            worker.daemon = True
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

    def set_random_seed(self, seed):
        if seed is None:
            seed = random.randint(1, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        monai.utils.misc.set_determinism(seed)
        np.random.RandomState(seed)


class ZarrDataset(monai.data.Dataset):
    def __init__(self, ome_levels, group_name, paths, patch_shape, patch_transform, num_producers: int = 1, num_workers: int = 1, queue_size: int = 64, base_seed=8338, use_LRU_cache=False, start_percent=1.0):

        self.group_name = group_name
        self.ome_levels = ome_levels  # Number of levels in the Zarr dataset
        self.paths = paths
        self.patch_shape = patch_shape
        self.patch_transform = patch_transform
        self.num_producers = num_producers
        self.num_workers = num_workers  # Number of worker processes per queue
        self.queue_size = queue_size
        self.base_seed = base_seed
        self.start_percent = start_percent

        super().__init__(paths, patch_transform)

        # Check if the paths are valid
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist.")

        self.zarr_data = []
        for path in paths:
            if use_LRU_cache:
                store_size = 2**28  # 256 MB
                cached_store = LRUStoreCache(FSStore(path), max_size=store_size)
                self.zarr_data.append(zarr.open(store=cached_store, mode='r', cache_attrs=True))
            else:
                self.zarr_data.append(zarr.open(path, mode='r', cache_attrs=True))

            store = parse_url(path, mode="r").store
            root = zarr.group(store=store)

            print(root.info)  # Print the metadata of the Zarr group
            print(root.tree())  # Print the structure of the Zarr group

        # Estimated RAM usage
        usage = 0.0
        bytes_per_ele = 4  # Assuming float32
        for level in range(len(ome_levels)):
            shape = np.array(self.patch_shape) * (2**level)
            usage += (np.prod(shape) * bytes_per_ele / 1024**2) * num_producers * queue_size  # in MB
        print(f"Estimated RAM usage: {usage:.2f} MB")

        # Start producer processes
        self._init_producers()

    def _init_producers(self):

        # Start worker processes
        self.producers = []
        for id in range(self.num_producers):
            producer = ZarrProducer(self.zarr_data,
                                    group_name=self.group_name,  # Use the path as the group name
                                    ome_levels=self.ome_levels,
                                    patch_shape=self.patch_shape,
                                    patch_transform=self.patch_transform,
                                    queue_size=self.queue_size,
                                    num_workers=self.num_workers,
                                    seed=self.base_seed + id*self.num_workers)  # Use a different seed for each producer
            self.producers.append(producer)

            producer.set_workers()
            producer.start_workers()

        # wait for all queues to fill up
        print(f"Waiting for producer queues to be {self.start_percent*100}% full...")
        for producer in self.producers:
            while producer.queue.qsize() < int(self.queue_size * self.start_percent):
                continue

    def stop_producers(self):
        # Stop the producer processes
        for producer in self.producers:
            producer.stop_workers()

    def __getitem__(self, index):

        while True:
            # Select a random producer
            producer = random.choice(self.producers)
            if producer.queue.empty():
                continue
            else:
                patch = producer.queue.get(timeout=0.1)  # block for time out space is available
                #patch = producer.queue.get()  # block for time out space is available
                return patch

    def __len__(self):
        return len(self.paths)
        #return self.num_producers * self.queue_size


if __name__ == "__main__":

    # Example usage
    ome_levels = ["0", "1", "2"]  # Example OME levels
    paths = ["path/to/zarr1", "path/to/zarr2"]  # Replace with actual Zarr paths
    patch_shape = (64, 64, 64)  # Example patch shape
    patch_transform = mt.Compose([
        mt.EnsureChannelFirstd(keys=ome_levels, channel_dim='no_channel')
    ])

    dataset = ZarrDataset(ome_levels=ome_levels,
                          group_name='HR',
                          paths=paths,
                          patch_shape=patch_shape,
                          patch_transform=patch_transform,
                          num_producers=4,
                          num_workers=1,
                          queue_size=64,
                          use_LRU_cache=False)

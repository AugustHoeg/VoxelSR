import zarr
import numpy as np
from zarr.storage import DirectoryStore
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale, write_multiscale_labels
from numcodecs import Blosc

def write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=(648, 648, 648), cname='lz4'):

    # Define the chunk sizes for each level
    chunk_sizes = [np.array(chunk_size) // (2**i) for i in range(len(image_pyramid))]
    print("Chunk sizes: ", chunk_sizes)

    # Define storage options for each level
    # Compressions: LZ4(), Zstd(level=3)
    # for Blosc, use cname='zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy'
    storage_opts = [
        {"chunks": chunk_sizes[i], "compression": Blosc(cname=cname, clevel=3, shuffle=Blosc.BITSHUFFLE)}
        for i in range(len(image_pyramid))
    ]

    # Write the image data to the Zarr group
    write_multiscale(
            image_pyramid,
            group=image_group,
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

    if label_pyramid is not None:
        # Now write the label pyramid under /volume/labels/mask/
        write_multiscale_labels(
            label_pyramid,
            group=image_group,
            name="mask",
            axes=["z", "y", "x"],
            storage_options=storage_opts
        )

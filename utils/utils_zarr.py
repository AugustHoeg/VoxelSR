import os
import numpy as np
from dask.diagnostics import ProgressBar
from ome_zarr.writer import write_image, write_multiscale, write_multiscale_labels, write_multiscales_metadata, write_label_metadata
from zarr.codecs import BloscCodec, BloscCname, BloscShuffle

def write_ome_pyramid(image_group, image_pyramid, label_pyramid, chunk_size=(648, 648, 648), shard_size=None, cname=None):

    # Define the chunk sizes for each level
    #chunk_shapes = [np.array(chunk_size) // (2**i) for i in range(len(image_pyramid))]
    chunk_shapes = [np.array(chunk_size) for _ in range(len(image_pyramid))]
    print("Chunk shapes: ", chunk_shapes)

    # Define the shard sizes for each level
    shard_shapes = None
    if shard_size is not None:
        shard_shapes = [chunk_shapes[i] * np.array(shard_size) * (2 ** i) for i in range(len(image_pyramid))]
        print("Shard shapes: ", shard_shapes)

    # Define storage options for each level
    # Compressions: LZ4(), Zstd(level=3)
    storage_opts = [
        {
            "chunks": chunk_shapes[i].tolist(),
            # "compressor": BloscCodec(cname=BloscCname[cname], clevel=3, shuffle=BloscShuffle.bitshuffle)
        }
        for i in range(len(image_pyramid))
    ]

    if shard_shapes is not None:
        for i in range(len(storage_opts)):
            storage_opts[i]["shards"] = shard_shapes[i].tolist()

    if cname is not None:
        for i in range(len(storage_opts)):
            storage_opts[i]["compressor"] = BloscCodec(cname=BloscCname[cname], clevel=3, shuffle=BloscShuffle.bitshuffle)

    with ProgressBar(dt=1.0):
        # Write the image data to the Zarr group
        write_multiscale(
                image_pyramid,
                group=image_group,
                axes=["z", "y", "x"],
                storage_options=storage_opts
            )

    with ProgressBar(dt=1.0):
        if label_pyramid is not None:
            # Now write the label pyramid under /volume/labels/mask/
            write_multiscale_labels(
                label_pyramid,
                group=image_group,
                name="mask",
                axes=["z", "y", "x"],
                storage_options=storage_opts
            )

    print("Done writing multiscale data to OME-Zarr group")
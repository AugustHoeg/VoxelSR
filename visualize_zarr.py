import zarr
import vizarr

if __name__ == "__main__":

    zarr_path = "../PyHPC/ome_array_pyramid_inference.zarr"
    store = zarr.open(zarr_path, mode='r')

    viewer = vizarr.Viewer()
    viewer.add_image(source=store, name="volume")
    viewer

    print("Done")
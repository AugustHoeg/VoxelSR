import os
import numpy as np
import glob
import tifffile
import monai
import monai.transforms as mt

if __name__ == "__main__":

    data_path = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/CAD*"
    images_HR = sorted(glob.glob(os.path.join(data_path, "HR/", "CAD*.tif")))

    for input_path in images_HR:
        print(input_path)
        npy_path = os.path.basename(input_path).replace(".tif", ".npy")
        new_path = os.path.join(os.path.dirname(input_path), npy_path)
        print(new_path)
        image = tifffile.imread(input_path)
        np.save(new_path, image)

    if False:
        load_dir = "/dtu/3d-imaging-center/projects/2022_QIM_52_Bone/analysis/LUND_data/CAD060/HR/CAD060.tif"
        loader = mt.LoadImage(reader="ITKReader", virtual_stack=True)
        image = loader(load_dir)
        print("image shape", image.shape)
        print("Done")


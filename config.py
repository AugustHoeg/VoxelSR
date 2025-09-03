import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is the Project Root directory

# ----------------------------------------
# Available options: MODEL_ARCHITECTURE
# ----------------------------------------
# 2D Models:
# "RCAN"
# "SwinIR"
# "HAT"
# "DRCT"

# 3D Models:
# "ArSSR"
# "mDCSRN_GAN"
# "mDCSRN"
# "SuperFormer"
# "ESRGAN3D"
# "RRDBNet3D"
# "EDDSR"
# "MFER"
# "MTVNet"
# "ConvNeXtSR"
# ----------------------------------------

MODEL_ARCHITECTURE = "ConvNeXtSR_2x_laptop"  # Flag for selecting model architecture if configuration file is not provided

# ----------------------------------------

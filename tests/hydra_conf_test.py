import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf

import config

from utils.load_options import init_options

@hydra.main(version_base=None, config_path="options", config_name=config.MODEL_ARCHITECTURE)
def main(opt: DictConfig):
    # Enable changes to the configuration
    OmegaConf.set_struct(opt, False)

    # Ensure Hydra does not change the working directory
    print(f"Current Working Directory: {os.getcwd()}")

    opt_path = os.path.join(config.ROOT_DIR, 'options', f'{HydraConfig.get().job.config_name}.yaml')

    # Initialize options
    init_options(opt, opt_path)

    # Print configuration (for debugging)
    print(f"configuration: \n {OmegaConf.to_yaml(opt)}")

if __name__ == "__main__":
    main()

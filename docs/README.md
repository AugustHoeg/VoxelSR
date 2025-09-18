# VoxelSR: A Deep Learning Framework for Super-Resolution of Volumetric Data 
[August Leander Høeg](https://github.com/AugustHoeg)

This repository contains tools and scripts for training of volumetric super-resolution methods on large-scale 3D datasets, as well as several baseline model implementations.

## Updates
- ✅ 2025-08-25: Added OME-Zarr data loading support for training and inference.
- **(To do)** Add dataset guide 

## Environment

### Installation
1. Clone the repository.
2. Create virtual environment.
3. Install requirements
```sh
pip install -r requirements.txt
```

## Training / Testing

- Please refer to the configuration files for each model located in ```/options```. These contain infomation regarding the model architecture to be trained/tested, the dataset and the SR scale.
- We provide separate configurations files for HCP 1200, IXI, BraTS 2023, Kirby 21 and FACTS datasets. 
- Note that the training procedure by default logs training statistics using [Weights and Biases](https://wandb.ai/).

### Training from scratch
To run the training procedure, in this case ```RRDBNet3D``` on dataset ```HCP_1200``` using, run the command: 
```python
invoke train RRDBNet3D HCP_1200  
```
- Trained models will be saved in ```/logs``` under the appropriate dataset and run name.   

## Testing
To run the test procedure, run the command:
```python
invoke testzarr <model_name> <dataset_name>   
```
- Performance statistics will also be saved in  ```/logs``` in the same location as the trained model parameters.

### Issues
Please contact: aulho@dtu.dk


# VoxelSR: A Deep Learning Framework for Super-Resolution of Volumetric Data 
[August Leander Høeg](https://github.com/AugustHoeg)

This repository contains tools and scripts for training of volumetric super-resolution methods on large-scale 3D datasets, as well as several baseline model implementations.

## Updates
- ✅ 2026-03-13: Added OME-Zarr data loading support for training and inference.

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
- We provide separate configurations files for multiple volumetric image datasets, including VoDaSuRe, CTSpine1K, LIDC-IDRI, LITS, and more. 
- Note that the training procedure by default logs training statistics using [Weights and Biases](https://wandb.ai/).

### Training from scratch
To run the SR training procedure, run the command: 
```python
invoke trainid <model_name> <dataset_name> <experiment_id> 
```
For example ```RRDBNet3D``` on dataset ```HCP_1200``` with experiment id ```ÌD000000```:
```python
invoke trainid RRDBNet3D HCP_1200 ID000000 
```
- Trained models will be saved in ```/logs``` under the appropriate dataset and run name.   

### Testing
To run the test procedure, run the following command with a completed experiment's id:
```python
invoke testzarrid <experiment_id> 
```
For example testing model run with experiment id ```ÌD000000```:
```python
invoke testzarrid ID000000 
```
- Performance statistics will also be saved in  ```/logs``` in the same location as the trained model parameters.

### Issues
Please contact: aulho@dtu.dk

### Note
This repository is based on [KAIR](https://github.com/cszn/KAIR).

